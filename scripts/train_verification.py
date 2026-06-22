"""
Subject-independent face verification for EmotionAR.

This script is separate from train_identification.py. It evaluates an
enrollment/probe authentication protocol:
  - leave one subject out
  - train an identity feature extractor with all remaining subjects
  - average K enrollment embeddings from the held-out subject into a template
  - compare probe embeddings with the template using cosine similarity
  - report AUC, EER, FAR, and FRR

The training stage uses a lightweight identity-classifier proxy to learn the
feature extractor. The verification stage is non-learned template matching.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_identification import (  # noqa: E402
    EMOTIONS,
    TRAIN_TRANSFORM,
    VAL_TRANSFORM,
    collect_records,
    maybe_limit_per_user,
    numeric_sort_key,
    parse_csv_option,
    set_seed,
)


DEFAULT_DATA_DIR = PROJECT_DIR / "data_ori" / "emoji-hero-vr-db-si"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "verification"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class IdentityDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(record["label"])


class ImageRecordDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_run_dir(results_dir, run_name, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = run_name.strip().replace(" ", "_") if run_name else "verification"
    safe_model_name = model_name.replace("/", "_")
    run_dir = Path(results_dir) / f"{timestamp}_{safe_run_name}_{safe_model_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_train_label_map(records, heldout_user):
    train_user_ids = sorted(
        {record["user_id"] for record in records if record["user_id"] != heldout_user},
        key=numeric_sort_key,
    )
    user_to_label = {user_id: idx for idx, user_id in enumerate(train_user_ids)}
    label_to_user = {idx: user_id for user_id, idx in user_to_label.items()}
    return user_to_label, label_to_user


def split_user_capture_groups(records, val_ratio, rng):
    by_capture = defaultdict(list)
    for record in records:
        by_capture[record["capture_key"]].append(record)

    groups = list(by_capture.values())
    rng.shuffle(groups)
    n_val_target = max(1, int(round(len(records) * val_ratio)))
    train_records, val_records = [], []
    for group in groups:
        if len(val_records) < n_val_target:
            val_records.extend(group)
        else:
            train_records.extend(group)

    if not train_records and val_records:
        train_records.append(val_records.pop())
    if not train_records or not val_records:
        raise ValueError("Could not create non-empty train/val split")
    return train_records, val_records


def create_loso_training_split(records, heldout_user, val_ratio, seed):
    rng = random.Random(seed + int(heldout_user) * 1009)
    by_user = defaultdict(list)
    for record in records:
        if record["user_id"] != heldout_user:
            by_user[record["user_id"]].append(record)

    train_rows, val_rows = [], []
    user_to_label, label_to_user = build_train_label_map(records, heldout_user)
    for user_id in sorted(by_user, key=numeric_sort_key):
        user_records = list(by_user[user_id])
        user_train, user_val = split_user_capture_groups(user_records, val_ratio, rng)
        for split_name, rows, target in (
            ("train", user_train, train_rows),
            ("val", user_val, val_rows),
        ):
            for record in rows:
                row = dict(record)
                row["split"] = split_name
                row["label"] = user_to_label[row["user_id"]]
                target.append(row)

    heldout_rows = [dict(record, split="heldout") for record in records if record["user_id"] == heldout_user]
    return train_rows, val_rows, heldout_rows, user_to_label, label_to_user


def class_weights(labels, num_classes, device):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for label in range(num_classes):
        count = counts.get(label, 0)
        weights.append(total / (num_classes * count) if count else 1.0)
    return torch.FloatTensor(weights).to(device)


def train_sampler(records, enabled):
    if not enabled:
        return None
    user_counts = Counter(record["user_id"] for record in records)
    emotion_counts = Counter(record["emotion"] for record in records)
    weights = []
    for record in records:
        # Balance identity and emotion softly without discarding data.
        user_weight = 1.0 / math.sqrt(user_counts[record["user_id"]])
        emotion_weight = 1.0 / math.sqrt(emotion_counts[record["emotion"]])
        weights.append(user_weight * emotion_weight)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_identity_loader(records, batch_size, transform, shuffle, num_workers, balance_emotion=False):
    sampler = train_sampler(records, enabled=balance_emotion)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(IdentityDataset(records, transform=transform), **kwargs)


def create_model(model_name, num_classes, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def evaluate_classifier(model, loader, device, num_classes):
    model.eval()
    labels_all, preds_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()
            labels_all.extend(labels.numpy().tolist())
            preds_all.extend(predictions.tolist())
    if not labels_all:
        return {"top1_accuracy": 0.0, "macro_f1": 0.0}
    return {
        "top1_accuracy": float(accuracy_score(labels_all, preds_all)),
        "macro_f1": float(
            f1_score(labels_all, preds_all, labels=list(range(num_classes)), average="macro", zero_division=0)
        ),
    }


def train_feature_extractor(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience, num_classes):
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            predictions = logits.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += predictions.eq(labels).sum().item()

        train_loss /= max(1, len(train_loader))
        train_acc = train_correct / train_total if train_total else 0.0
        val_metrics = evaluate_classifier(model, val_loader, device, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_top1_acc": val_metrics["top1_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_top1={val_metrics['top1_accuracy']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["top1_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["top1_accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {
        "best_val_top1_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_time_sec": time.time() - start_time,
    }


def prepare_embedding_model(model):
    if not hasattr(model, "reset_classifier"):
        raise RuntimeError("This timm model does not expose reset_classifier; choose another backbone.")
    model.reset_classifier(0)
    model.eval()
    return model


def make_record_loader(records, batch_size, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(ImageRecordDataset(records, transform=VAL_TRANSFORM), **kwargs)


def flatten_features(features):
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim > 2:
        features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)
    return features


def extract_embeddings(model, records, batch_size, num_workers, device):
    unique_records = []
    seen_paths = set()
    for record in records:
        if record["path"] not in seen_paths:
            unique_records.append(record)
            seen_paths.add(record["path"])

    loader = make_record_loader(unique_records, batch_size, num_workers)
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            features = flatten_features(model(images))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            features_np = features.cpu().numpy()
            for row_idx, embedding in zip(indices.numpy().tolist(), features_np):
                embeddings[unique_records[row_idx]["path"]] = embedding.astype(np.float32)
    return embeddings


def expand_eval_modes(value):
    modes = []
    for item in parse_csv_option(value):
        if item == "same_camera":
            modes.extend(["same_camera_0", "same_camera_1"])
        elif item == "cross_camera":
            modes.extend(["cross_camera_0to1", "cross_camera_1to0"])
        else:
            modes.append(item)
    return modes


def candidate_records_for_mode(heldout_rows, negative_pool, mode):
    if mode == "all":
        return heldout_rows, heldout_rows, negative_pool
    if mode == "same_camera_0":
        return (
            [row for row in heldout_rows if row["camera_index"] == "0"],
            [row for row in heldout_rows if row["camera_index"] == "0"],
            [row for row in negative_pool if row["camera_index"] == "0"],
        )
    if mode == "same_camera_1":
        return (
            [row for row in heldout_rows if row["camera_index"] == "1"],
            [row for row in heldout_rows if row["camera_index"] == "1"],
            [row for row in negative_pool if row["camera_index"] == "1"],
        )
    if mode == "cross_camera_0to1":
        return (
            [row for row in heldout_rows if row["camera_index"] == "0"],
            [row for row in heldout_rows if row["camera_index"] == "1"],
            [row for row in negative_pool if row["camera_index"] == "1"],
        )
    if mode == "cross_camera_1to0":
        return (
            [row for row in heldout_rows if row["camera_index"] == "1"],
            [row for row in heldout_rows if row["camera_index"] == "0"],
            [row for row in negative_pool if row["camera_index"] == "0"],
        )
    raise ValueError(f"Unknown eval mode: {mode}")


def balanced_sample(records, n, rng):
    if n <= 0:
        return []
    if len(records) < n:
        raise ValueError(f"Need {n} records but only found {len(records)}")
    by_emotion = defaultdict(list)
    for record in records:
        by_emotion[record["emotion"]].append(record)
    for rows in by_emotion.values():
        rng.shuffle(rows)

    selected = []
    selected_paths = set()
    emotions = list(by_emotion.keys())
    rng.shuffle(emotions)
    while len(selected) < n:
        made_progress = False
        emotions.sort(key=lambda emotion: sum(1 for row in selected if row["emotion"] == emotion))
        for emotion in emotions:
            while by_emotion[emotion] and by_emotion[emotion][-1]["path"] in selected_paths:
                by_emotion[emotion].pop()
            if by_emotion[emotion]:
                row = by_emotion[emotion].pop()
                selected.append(row)
                selected_paths.add(row["path"])
                made_progress = True
                if len(selected) == n:
                    break
        if not made_progress:
            break
    if len(selected) < n:
        remaining = [record for record in records if record["path"] not in selected_paths]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    return selected


def maybe_limit_balanced(records, max_records, rng):
    if not max_records or len(records) <= max_records:
        return list(records)
    return balanced_sample(list(records), max_records, rng)


def sample_negatives(negative_pool, positives, rng, negatives_per_positive):
    by_emotion = defaultdict(list)
    for record in negative_pool:
        by_emotion[record["emotion"]].append(record)
    for rows in by_emotion.values():
        rng.shuffle(rows)

    selected = []
    used_paths = set()
    for positive in positives:
        emotion = positive["emotion"]
        candidates = by_emotion.get(emotion, [])
        for _ in range(negatives_per_positive):
            while candidates and candidates[-1]["path"] in used_paths:
                candidates.pop()
            if candidates:
                row = candidates.pop()
            else:
                fallback = [record for record in negative_pool if record["path"] not in used_paths]
                if not fallback:
                    break
                row = rng.choice(fallback)
            selected.append(row)
            used_paths.add(row["path"])
    return selected


def template_from_records(enrollment_records, embeddings):
    vectors = np.stack([embeddings[record["path"]] for record in enrollment_records], axis=0)
    template = vectors.mean(axis=0)
    norm = np.linalg.norm(template)
    if norm > 0:
        template = template / norm
    return template.astype(np.float32)


def score_records(template, records, embeddings):
    scores = []
    for record in records:
        score = float(np.dot(template, embeddings[record["path"]]))
        scores.append(score)
    return scores


def compute_verification_metrics(labels, scores):
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    if len(np.unique(labels_arr)) < 2:
        return {
            "auc": float("nan"),
            "eer": float("nan"),
            "far_at_eer": float("nan"),
            "frr_at_eer": float("nan"),
            "threshold_eer": float("nan"),
            "accuracy_at_eer": float("nan"),
        }

    auc = float(roc_auc_score(labels_arr, scores_arr))
    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    predictions = (scores_arr > threshold).astype(int)
    accuracy = float((predictions == labels_arr).mean())
    return {
        "auc": auc,
        "eer": eer,
        "far_at_eer": float(fpr[idx]),
        "frr_at_eer": float(fnr[idx]),
        "threshold_eer": threshold,
        "accuracy_at_eer": accuracy,
    }


def evaluate_fold_verification(
    heldout_user,
    heldout_rows,
    negative_pool,
    embeddings,
    enrollment_sizes,
    eval_modes,
    enrollment_repeats,
    negatives_per_positive,
    max_positive_probes,
    seed,
):
    fold_metrics = []
    pair_rows = []
    for mode_index, eval_mode in enumerate(eval_modes):
        enrollment_candidates, positive_candidates, negative_candidates = candidate_records_for_mode(
            heldout_rows, negative_pool, eval_mode
        )
        for k in enrollment_sizes:
            if len(enrollment_candidates) < k:
                continue
            for repeat in range(enrollment_repeats):
                rng = random.Random(seed + int(heldout_user) * 100003 + k * 1009 + repeat * 97 + mode_index)
                enrollment_records = balanced_sample(enrollment_candidates, k, rng)
                enrollment_paths = {record["path"] for record in enrollment_records}
                positives = [record for record in positive_candidates if record["path"] not in enrollment_paths]
                positives = maybe_limit_balanced(positives, max_positive_probes, rng)
                negatives = sample_negatives(negative_candidates, positives, rng, negatives_per_positive)
                if not positives or not negatives:
                    continue

                template = template_from_records(enrollment_records, embeddings)
                positive_scores = score_records(template, positives, embeddings)
                negative_scores = score_records(template, negatives, embeddings)
                labels = [1] * len(positive_scores) + [0] * len(negative_scores)
                scores = positive_scores + negative_scores
                metrics = compute_verification_metrics(labels, scores)
                emotion_counts_positive = Counter(record["emotion"] for record in positives)
                emotion_counts_negative = Counter(record["emotion"] for record in negatives)
                camera_counts_positive = Counter(record["camera_index"] for record in positives)
                camera_counts_negative = Counter(record["camera_index"] for record in negatives)

                metric_row = {
                    "fold_user": heldout_user,
                    "k": k,
                    "repeat": repeat,
                    "eval_mode": eval_mode,
                    "n_enrollment": len(enrollment_records),
                    "n_positive": len(positives),
                    "n_negative": len(negatives),
                    "auc": metrics["auc"],
                    "eer": metrics["eer"],
                    "far_at_eer": metrics["far_at_eer"],
                    "frr_at_eer": metrics["frr_at_eer"],
                    "threshold_eer": metrics["threshold_eer"],
                    "accuracy_at_eer": metrics["accuracy_at_eer"],
                    "positive_emotion_counts": json.dumps(dict(sorted(emotion_counts_positive.items()))),
                    "negative_emotion_counts": json.dumps(dict(sorted(emotion_counts_negative.items()))),
                    "positive_camera_counts": json.dumps(dict(sorted(camera_counts_positive.items()))),
                    "negative_camera_counts": json.dumps(dict(sorted(camera_counts_negative.items()))),
                    "enrollment_filenames": ";".join(record["filename"] for record in enrollment_records),
                }
                fold_metrics.append(metric_row)

                for label, score, record in (
                    [(1, score, record) for score, record in zip(positive_scores, positives)]
                    + [(0, score, record) for score, record in zip(negative_scores, negatives)]
                ):
                    pair_rows.append(
                        {
                            "fold_user": heldout_user,
                            "k": k,
                            "repeat": repeat,
                            "eval_mode": eval_mode,
                            "label": label,
                            "score": score,
                            "probe_user": record["user_id"],
                            "probe_emotion": record["emotion"],
                            "probe_camera": record["camera_index"],
                            "probe_filename": record["filename"],
                            "enrollment_filenames": metric_row["enrollment_filenames"],
                        }
                    )
    return fold_metrics, pair_rows


def aggregate_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["k"], row["eval_mode"])].append(row)

    summary_rows = []
    metric_names = ["auc", "eer", "far_at_eer", "frr_at_eer", "accuracy_at_eer"]
    for (k, eval_mode), group in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        summary = {"k": k, "eval_mode": eval_mode, "n_rows": len(group)}
        fold_ids = sorted({row["fold_user"] for row in group}, key=numeric_sort_key)
        summary["n_folds"] = len(fold_ids)
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in group], dtype=float)
            values = values[~np.isnan(values)]
            if values.size:
                summary[f"{metric_name}_mean"] = float(values.mean())
                summary[f"{metric_name}_std"] = float(values.std(ddof=0))
            else:
                summary[f"{metric_name}_mean"] = float("nan")
                summary[f"{metric_name}_std"] = float("nan")
        summary_rows.append(summary)
    return summary_rows


def save_history(path, rows):
    fieldnames = [
        "model",
        "fold_user",
        "epoch",
        "train_loss",
        "train_acc",
        "val_top1_acc",
        "val_macro_f1",
    ]
    save_csv(path, rows, fieldnames)


def save_fold_metrics(path, rows):
    fieldnames = [
        "model",
        "fold_user",
        "k",
        "repeat",
        "eval_mode",
        "n_enrollment",
        "n_positive",
        "n_negative",
        "auc",
        "eer",
        "far_at_eer",
        "frr_at_eer",
        "threshold_eer",
        "accuracy_at_eer",
        "positive_emotion_counts",
        "negative_emotion_counts",
        "positive_camera_counts",
        "negative_camera_counts",
        "enrollment_filenames",
        "train_time_sec",
        "best_epoch",
        "best_val_top1_acc",
    ]
    save_csv(path, rows, fieldnames)


def save_pair_scores(path, rows):
    fieldnames = [
        "model",
        "fold_user",
        "k",
        "repeat",
        "eval_mode",
        "label",
        "score",
        "probe_user",
        "probe_emotion",
        "probe_camera",
        "probe_filename",
        "enrollment_filenames",
    ]
    save_csv(path, rows, fieldnames)


def save_summary_rows(path, rows):
    fieldnames = [
        "model",
        "k",
        "eval_mode",
        "n_rows",
        "n_folds",
        "auc_mean",
        "auc_std",
        "eer_mean",
        "eer_std",
        "far_at_eer_mean",
        "far_at_eer_std",
        "frr_at_eer_mean",
        "frr_at_eer_std",
        "accuracy_at_eer_mean",
        "accuracy_at_eer_std",
    ]
    save_csv(path, rows, fieldnames)


def parse_enrollment_sizes(value):
    sizes = [int(item) for item in parse_csv_option(value)]
    if not sizes:
        raise ValueError("At least one enrollment size is required")
    return sizes


def choose_folds(records, folds_value, max_folds):
    users = sorted({record["user_id"] for record in records}, key=numeric_sort_key)
    if folds_value and folds_value.lower() != "all":
        requested = {str(int(item)) if item.isdigit() else item for item in parse_csv_option(folds_value)}
        users = [user for user in users if user in requested]
    if max_folds:
        users = users[:max_folds]
    return users


def run_one_model(args, model_name):
    set_seed(args.seed)
    records = collect_records(args.data_dir, args.data_layout)
    records = maybe_limit_per_user(records, args.limit_per_user, args.seed)
    if not records:
        raise RuntimeError(f"No images found under {args.data_dir}")

    fold_users = choose_folds(records, args.folds, args.max_folds)
    enrollment_sizes = parse_enrollment_sizes(args.enrollment_sizes)
    eval_modes = expand_eval_modes(args.eval_modes)
    run_dir = create_run_dir(args.results_dir, args.run_name, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "pretrained": not args.no_pretrained,
        "data_dir": str(Path(args.data_dir).resolve()),
        "data_layout": args.data_layout,
        "n_images": len(records),
        "n_users": len({record["user_id"] for record in records}),
        "fold_users": fold_users,
        "enrollment_sizes": enrollment_sizes,
        "eval_modes": eval_modes,
        "enrollment_repeats": args.enrollment_repeats,
        "negatives_per_positive": args.negatives_per_positive,
        "max_positive_probes": args.max_positive_probes,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "train_emotion_balanced_sampler": not args.no_train_emotion_balance,
        "limit_per_user": args.limit_per_user,
        "seed": args.seed,
    }
    write_json(run_dir / "run_config.json", config)

    print(f"Run directory: {run_dir}")
    print(f"Images: {len(records)}")
    print(f"Users: {config['n_users']}")
    print(f"Folds: {len(fold_users)} ({', '.join(fold_users)})")
    print(f"Enrollment sizes: {enrollment_sizes}")
    print(f"Eval modes: {eval_modes}")

    if args.dry_run:
        summary = {"config": config, "dry_run": True}
        write_json(run_dir / "summary.json", summary)
        print("Dry run complete. No model was trained.")
        return run_dir

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    all_fold_metrics = []
    all_pair_rows = []
    all_history = []

    for fold_index, heldout_user in enumerate(fold_users, start=1):
        print(f"\n=== Fold {fold_index}/{len(fold_users)}: held-out P{int(heldout_user):02d} ===")
        train_rows, val_rows, heldout_rows, user_to_label, label_to_user = create_loso_training_split(
            records,
            heldout_user=heldout_user,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        num_classes = len(user_to_label)
        print(f"Train images: {len(train_rows)} | Val images: {len(val_rows)} | Held-out images: {len(heldout_rows)}")
        print(f"Training identity classes: {num_classes}")

        train_loader = make_identity_loader(
            train_rows,
            args.batch_size,
            TRAIN_TRANSFORM,
            shuffle=True,
            num_workers=args.num_workers,
            balance_emotion=not args.no_train_emotion_balance,
        )
        val_loader = make_identity_loader(
            val_rows,
            args.batch_size,
            VAL_TRANSFORM,
            shuffle=False,
            num_workers=args.num_workers,
            balance_emotion=False,
        )

        model = create_model(model_name, num_classes=num_classes, pretrained=not args.no_pretrained).to(device)
        labels = [row["label"] for row in train_rows]
        criterion = nn.CrossEntropyLoss(weight=class_weights(labels, num_classes, device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model, history, training_summary = train_feature_extractor(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=args.epochs,
            patience=args.patience,
            num_classes=num_classes,
        )
        for row in history:
            all_history.append({"model": model_name, "fold_user": heldout_user, **row})

        embedding_model = prepare_embedding_model(model)
        embedding_records = heldout_rows + val_rows
        embeddings = extract_embeddings(embedding_model, embedding_records, args.batch_size, args.num_workers, device)
        fold_metrics, pair_rows = evaluate_fold_verification(
            heldout_user=heldout_user,
            heldout_rows=heldout_rows,
            negative_pool=val_rows,
            embeddings=embeddings,
            enrollment_sizes=enrollment_sizes,
            eval_modes=eval_modes,
            enrollment_repeats=args.enrollment_repeats,
            negatives_per_positive=args.negatives_per_positive,
            max_positive_probes=args.max_positive_probes,
            seed=args.seed,
        )
        for row in fold_metrics:
            row["model"] = model_name
            row["train_time_sec"] = training_summary["train_time_sec"]
            row["best_epoch"] = training_summary["best_epoch"]
            row["best_val_top1_acc"] = training_summary["best_val_top1_acc"]
            all_fold_metrics.append(row)
        for row in pair_rows:
            row["model"] = model_name
            all_pair_rows.append(row)

        if args.save_fold_models:
            torch.save(
                {
                    "model_name": model_name,
                    "heldout_user": heldout_user,
                    "model_state_dict": model.state_dict(),
                    "user_to_label": user_to_label,
                    "label_to_user": label_to_user,
                    "training_summary": training_summary,
                    "config": config,
                },
                run_dir / f"fold_P{int(heldout_user):02d}_feature_extractor.pt",
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = aggregate_metrics(all_fold_metrics)
    for row in summary_rows:
        row["model"] = model_name
    save_history(run_dir / "training_history.csv", all_history)
    save_fold_metrics(run_dir / "fold_metrics.csv", all_fold_metrics)
    save_pair_scores(run_dir / "pair_scores.csv", all_pair_rows)
    save_summary_rows(run_dir / "summary_by_k_mode.csv", summary_rows)

    summary = {
        "config": config,
        "n_fold_metric_rows": len(all_fold_metrics),
        "n_pair_rows": len(all_pair_rows),
        "summary_by_k_mode": summary_rows,
    }
    write_json(run_dir / "summary.json", summary)

    print("\nVerification summary")
    for row in summary_rows:
        print(
            f"  K={row['k']} {row['eval_mode']}: "
            f"AUC={row['auc_mean']:.4f}+/-{row['auc_std']:.4f}, "
            f"EER={row['eer_mean']:.4f}+/-{row['eer_std']:.4f}, "
            f"Acc@EER={row['accuracy_at_eer_mean']:.4f}"
        )
    print(f"Summary: {run_dir / 'summary.json'}")
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="LOSO enrollment/probe face verification for EmotionAR")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-layout", type=str, default="auto", choices=["auto", "flat", "original"])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models", type=str, default="efficientnet_b0")
    parser.add_argument("--run-name", type=str, default="verification")
    parser.add_argument("--folds", type=str, default="all", help="Comma-separated held-out users, e.g. 1,2,3, or all")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional cap for quick pilots")
    parser.add_argument("--enrollment-sizes", type=str, default="1,3,5")
    parser.add_argument(
        "--eval-modes",
        type=str,
        default="all,same_camera,cross_camera",
        help="Comma-separated: all, same_camera, cross_camera, same_camera_0, same_camera_1, cross_camera_0to1, cross_camera_1to0",
    )
    parser.add_argument("--enrollment-repeats", type=int, default=1)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-positive-probes", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-user", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-train-emotion-balance", action="store_true")
    parser.add_argument("--save-fold-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = [model.strip() for model in args.models.split(",") if model.strip()]
    if not model_names:
        raise ValueError("At least one model name is required")
    run_dirs = []
    for model_name in model_names:
        print(f"\n=== Running verification model: {model_name} ===")
        run_dirs.append(str(run_one_model(args, model_name)))
    print("\nCompleted runs:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")


if __name__ == "__main__":
    main()
