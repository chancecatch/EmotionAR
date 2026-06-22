"""
Leave-One-Out (LOO) Cross-Validation for EmotionAR Personalization Study
2-Stage Pipeline with Early Stopping and MLflow Tracking

Following EmojiHeroVR paper methodology:
  - EfficientNet-B0 with ImageNet pretrained weights
  - User-based validation split (8 users for validation)
  - 2-Stage Training: Base -> Personalize

Stages:
  Stage 1 (Base): Train on 28 users (8 users for validation, 1 LOO user)
  Stage 2 (Personalize): Fine-tune with Base + Personal data
    - Option A: Full layers trainable (default)
    - Option B: Classifier only (--classifier-only flag)
"""

import os
import sys
import argparse
import copy
import csv
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import torchvision
from mlflow.models.signature import infer_signature

# MLflow
import mlflow
from mlflow import pytorch as mlflow_pytorch

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # RTX 4090 optimization: enable cuDNN auto-tuner for fastest convolution algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "emoji-hero-vr-db-si"
RESULTS_DIR = PROJECT_DIR / "results" / "loo_cv"

# Emotion labels
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
ID_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}
NEUTRAL_LABEL = EMOTION_TO_ID['Neutral']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ORIGINAL_SETS = ['training_set', 'validation_set', 'test_set']
SET_ID_TO_NAME = {'0': 'training_set', '1': 'validation_set', '2': 'test_set'}
SIMILAR_EMOTION_FALLBACKS = {
    EMOTION_TO_ID['Anger']: [EMOTION_TO_ID[e] for e in ['Disgust', 'Fear', 'Sadness', 'Surprise', 'Neutral', 'Happiness']],
    EMOTION_TO_ID['Disgust']: [EMOTION_TO_ID[e] for e in ['Anger', 'Sadness', 'Fear', 'Neutral', 'Surprise', 'Happiness']],
    EMOTION_TO_ID['Fear']: [EMOTION_TO_ID[e] for e in ['Surprise', 'Anger', 'Disgust', 'Sadness', 'Neutral', 'Happiness']],
    EMOTION_TO_ID['Happiness']: [EMOTION_TO_ID[e] for e in ['Surprise', 'Neutral', 'Anger', 'Disgust', 'Sadness', 'Fear']],
    EMOTION_TO_ID['Neutral']: [EMOTION_TO_ID[e] for e in ['Sadness', 'Happiness', 'Disgust', 'Anger', 'Fear', 'Surprise']],
    EMOTION_TO_ID['Sadness']: [EMOTION_TO_ID[e] for e in ['Neutral', 'Disgust', 'Anger', 'Fear', 'Happiness', 'Surprise']],
    EMOTION_TO_ID['Surprise']: [EMOTION_TO_ID[e] for e in ['Fear', 'Happiness', 'Anger', 'Disgust', 'Neutral', 'Sadness']],
}

# Paper-matched augmentations (translation/zoom via affine)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def numeric_sort_key(value):
    value = str(value)
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def parse_csv_option(value):
    if not value or value == 'all':
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]


def parse_int_csv_option(value):
    items = parse_csv_option(value)
    if not items:
        return []
    parsed = sorted({int(item) for item in items})
    if any(item <= 0 for item in parsed):
        raise ValueError(f"Expected positive integers, got: {value}")
    return parsed


def make_data_loader(dataset, batch_size=64, shuffle=False, num_workers=8,
                     pin_memory=True, persistent_workers=True):
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
    return DataLoader(dataset, **loader_kwargs)


def parse_filename_metadata(filename):
    """Parse EmoHeVRDB-SI file names.

    Expected format:
    <timestamp>-<set-id>-<participant-id>-<level-id>-<emoji-id>-<emotion-id>-<camera-index>.jpg
    """
    stem = os.path.basename(filename).rsplit('.', 1)[0]
    parts = stem.split('-')
    if len(parts) != 7:
        raise ValueError(
            "Expected filename format "
            "<timestamp>-<set-id>-<participant-id>-<level-id>-<emoji-id>-<emotion-id>-<camera-index>.jpg; "
            f"got: {filename}"
        )
    return {
        'timestamp': parts[0],
        'set_id': parts[1],
        'participant_id': parts[2],
        'user_id': parts[2],
        'level_id': parts[3],
        'emoji_id': parts[4],
        'emotion_id': parts[5],
        'camera_index': parts[6],
        'capture_key': '-'.join(parts[:-1]),
    }


def resolve_si_root(data_dir):
    data_dir = Path(data_dir)
    nested_si = data_dir / "emoji-hero-vr-db-si"
    if nested_si.exists():
        return nested_si
    return data_dir


def infer_data_layout(data_dir):
    data_dir = resolve_si_root(data_dir)
    if any((data_dir / set_name).is_dir() for set_name in ORIGINAL_SETS):
        return 'original'
    if any((data_dir / emotion).is_dir() for emotion in EMOTIONS):
        return 'flat'
    raise RuntimeError(f"Could not infer data layout under {data_dir}")


def collect_records(data_dir=DATA_DIR, data_layout='auto'):
    """Collect image records with participant, emotion, camera, and source-set metadata."""
    records = []
    data_dir = resolve_si_root(data_dir)
    if data_layout == 'auto':
        data_layout = infer_data_layout(data_dir)

    def append_record(image_path, emotion, original_set):
        metadata = parse_filename_metadata(image_path.name)
        label = EMOTION_TO_ID[emotion]
        if int(metadata['emotion_id']) != label:
            raise ValueError(
                f"Emotion folder/name mismatch for {image_path}: "
                f"folder={emotion} filename_emotion_id={metadata['emotion_id']}"
            )
        records.append({
            'path': str(image_path),
            'filename': image_path.name,
            'label': label,
            'emotion': emotion,
            'original_set': original_set,
            **metadata,
        })

    if data_layout == 'original':
        for original_set in ORIGINAL_SETS:
            set_dir = data_dir / original_set
            if not set_dir.exists():
                continue
            for emotion in EMOTIONS:
                emotion_dir = set_dir / emotion
                if not emotion_dir.exists():
                    continue
                for image_path in sorted(emotion_dir.iterdir()):
                    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        append_record(image_path, emotion, original_set)
    elif data_layout == 'flat':
        for emotion in EMOTIONS:
            emotion_dir = data_dir / emotion
            if not emotion_dir.exists():
                continue
            for image_path in sorted(emotion_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    metadata = parse_filename_metadata(image_path.name)
                    append_record(image_path, emotion, SET_ID_TO_NAME.get(metadata['set_id'], 'unknown'))
    else:
        raise ValueError(f"Unknown data_layout: {data_layout}")

    if not records:
        raise RuntimeError(f"No image records found under {data_dir} with layout={data_layout}")
    return records


def records_by_user(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['user_id']].append(record)
    return dict(grouped)


def records_to_data(records):
    return {
        'images': [record['path'] for record in records],
        'labels': [record['label'] for record in records],
        'records': list(records),
    }


def balance_records(records, seed=42):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['label']].append(record)
    if not grouped:
        return []
    min_count = min(len(grouped[label]) for label in grouped)
    rng = random.Random(seed)
    balanced = []
    for label in range(len(EMOTIONS)):
        samples = list(grouped.get(label, []))
        rng.shuffle(samples)
        balanced.extend(samples[:min_count])
    rng.shuffle(balanced)
    return balanced


class EmojiHeroDataset(Dataset):
    """Custom dataset for EmojiHeroVR images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or val_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_user_id_from_filename(filename):
    """Extract user ID from filename format: timestamp-0-USER_ID-session-..."""
    parts = os.path.basename(filename).split('-')
    return parts[2]


def collect_all_data():
    """Collect all image paths organized by user and emotion."""
    import glob
    user_data = defaultdict(lambda: {'images': [], 'labels': []})
    
    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = DATA_DIR / emotion
        if emotion_dir.exists():
            for img_path in glob.glob(str(emotion_dir / "*.jpg")) + glob.glob(str(emotion_dir / "*.png")):
                user_id = get_user_id_from_filename(img_path)
                user_data[user_id]['images'].append(img_path)
                user_data[user_id]['labels'].append(emotion_idx)
    
    return dict(user_data)


def balance_dataset(images, labels, seed=42):
    """Balance dataset by undersampling to minimum class frequency."""
    from collections import Counter
    
    rng = random.Random(seed)
    class_counts = Counter(labels)
    if not class_counts:
        return [], []
    min_count = min(class_counts.values())
    
    class_samples = {c: [] for c in range(7)}
    for img, lbl in zip(images, labels):
        class_samples[lbl].append(img)
    
    balanced_images, balanced_labels = [], []
    for class_idx in range(7):
        samples = class_samples[class_idx]
        if len(samples) > min_count:
            rng.shuffle(samples)
            samples = samples[:min_count]
        balanced_images.extend(samples)
        balanced_labels.extend([class_idx] * len(samples))
    
    combined = list(zip(balanced_images, balanced_labels))
    rng.shuffle(combined)
    if not combined:
        return [], []
    balanced_images, balanced_labels = zip(*combined)
    
    return list(balanced_images), list(balanced_labels)


def create_loo_splits(user_data, leave_out_user, seed=42, personal_test_ratio=0.2):
    """
    Create train/val/test splits for 2-Stage LOO pipeline.
    
    Split: 28 train / 8 val / 1 LOO user (total 37 users)
    - Stage 1: Train on 28 users, validate on 8 users, test on LOO user (20% holdout)
    - Stage 2: Add LOO user calibration data (80%) to training, test on LOO user (20% holdout)
    
    NOTE: Personal data is split 80/20 to prevent data leakage between train and test.
    """
    all_users = [u for u in user_data.keys() if u != leave_out_user]
    
    rng = random.Random(seed)
    rng.shuffle(all_users)
    
    n_train = 28
    n_val = 8
    
    train_users = set(all_users[:n_train])
    val_users = set(all_users[n_train:n_train + n_val])
    
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    personal_images, personal_labels = [], []
    
    for user_id, data in user_data.items():
        if user_id == leave_out_user:
            personal_images.extend(data['images'])
            personal_labels.extend(data['labels'])
        elif user_id in train_users:
            train_images.extend(data['images'])
            train_labels.extend(data['labels'])
        elif user_id in val_users:
            val_images.extend(data['images'])
            val_labels.extend(data['labels'])
    
    # Split personal data 80/20 for train/test to prevent data leakage
    personal_combined = list(zip(personal_images, personal_labels))
    rng.shuffle(personal_combined)
    
    n_test = max(1, int(len(personal_combined) * personal_test_ratio))
    test_data = personal_combined[:n_test]
    calibration_data = personal_combined[n_test:]
    
    if test_data:
        test_images_split, test_labels_split = zip(*test_data)
        test_images_split, test_labels_split = list(test_images_split), list(test_labels_split)
    else:
        test_images_split, test_labels_split = [], []
    
    if calibration_data:
        calibration_images, calibration_labels = zip(*calibration_data)
        calibration_images, calibration_labels = list(calibration_images), list(calibration_labels)
    else:
        calibration_images, calibration_labels = [], []
    
    # Balance validation set (test set uses all available samples for reliability)
    val_images_balanced, val_labels_balanced = balance_dataset(val_images, val_labels, seed)
    
    print(f"  [Split] Train: {len(train_users)} users ({len(train_images)} images)")
    print(f"  [Split] Val: {len(val_users)} users ({len(val_images_balanced)} images, balanced)")
    print(f"  [Split] Personal (LOO User {leave_out_user}): {len(personal_images)} total images")
    print(f"         └── Calibration (80%): {len(calibration_images)} images (for Stage 2 training)")
    print(f"         └── Test (20%): {len(test_images_split)} images (held out, never seen during training)")
    
    return {
        'train': {'images': train_images, 'labels': train_labels},
        'val': {'images': val_images_balanced, 'labels': val_labels_balanced},
        'test': {'images': test_images_split, 'labels': test_labels_split},
        'personal': {'images': calibration_images, 'labels': calibration_labels},  # Now only calibration portion
        'train_users': len(train_users),
        'val_users': len(val_users),
        'personal_total': len(personal_images),
        'personal_calibration': len(calibration_images),
        'personal_test': len(test_images_split),
    }


def paired_captures(records, label=None, excluded_paths=None):
    excluded_paths = excluded_paths or set()
    grouped = defaultdict(dict)
    for record in records:
        if label is not None and record['label'] != label:
            continue
        if record['path'] in excluded_paths:
            continue
        grouped[record['capture_key']][record['camera_index']] = record

    pairs = []
    for capture_key, by_camera in grouped.items():
        if '0' in by_camera and '1' in by_camera:
            pairs.append((capture_key, [by_camera['0'], by_camera['1']]))
    return sorted(pairs, key=lambda item: item[0])


def select_neutral_auth_like_records(personal_records, shots=14, seed=42):
    """Select authentication-like samples: neutral central/side pairs only."""
    rng = random.Random(seed)
    pairs = paired_captures(personal_records, label=NEUTRAL_LABEL)
    rng.shuffle(pairs)

    selected = []
    manifest = []
    for capture_key, pair_records in pairs:
        if len(selected) >= shots:
            break
        for record in sorted(pair_records, key=lambda r: r['camera_index']):
            if len(selected) >= shots:
                break
            selected.append(record)
            manifest.append({
                **manifest_row(record, 'neutral_auth_like'),
                'selection_note': 'neutral_pair',
                'replacement_for_emotion': '',
            })

    if len(selected) < shots:
        selected_paths = {record['path'] for record in selected}
        fallback = [record for record in personal_records
                    if record['label'] == NEUTRAL_LABEL and record['path'] not in selected_paths]
        rng.shuffle(fallback)
        for record in fallback:
            if len(selected) >= shots:
                break
            selected.append(record)
            manifest.append({
                **manifest_row(record, 'neutral_auth_like'),
                'selection_note': 'neutral_unpaired_fallback',
                'replacement_for_emotion': '',
            })

    if len(selected) < shots:
        raise RuntimeError(f"Only found {len(selected)} neutral records for requested {shots} shots")
    return selected[:shots], manifest[:shots]


def select_balanced14_records(personal_records, seed=42, replacement_policy='keep_14'):
    """Select one central/side pair per emotion; fill missing emotions if requested."""
    rng = random.Random(seed)
    selected = []
    manifest = []
    missing_labels = []
    selected_paths = set()

    for label in range(len(EMOTIONS)):
        pairs = paired_captures(personal_records, label=label, excluded_paths=selected_paths)
        rng.shuffle(pairs)
        if not pairs:
            missing_labels.append(label)
            continue
        capture_key, pair_records = pairs[0]
        for record in sorted(pair_records, key=lambda r: r['camera_index']):
            selected.append(record)
            selected_paths.add(record['path'])
            manifest.append({
                **manifest_row(record, 'balanced14'),
                'selection_note': 'emotion_camera_pair',
                'replacement_for_emotion': '',
            })

    if missing_labels and replacement_policy == 'strict':
        missing = ','.join(ID_TO_EMOTION[label] for label in missing_labels)
        raise RuntimeError(f"Missing required emotion pairs for balanced14: {missing}")

    if missing_labels and replacement_policy == 'keep_14':
        for replacement_for in missing_labels:
            replacement_pair = None
            priority = SIMILAR_EMOTION_FALLBACKS.get(replacement_for, [])
            fallback_priority = priority + [
                label for label in range(len(EMOTIONS))
                if label != replacement_for and label not in priority
            ]
            for fallback_label in fallback_priority:
                candidate_pairs = paired_captures(
                    personal_records, label=fallback_label, excluded_paths=selected_paths
                )
                rng.shuffle(candidate_pairs)
                if candidate_pairs:
                    replacement_pair = candidate_pairs[0]
                    break
            if replacement_pair is None:
                continue
            capture_key, pair_records = replacement_pair
            for record in sorted(pair_records, key=lambda r: r['camera_index']):
                if len(selected) >= 14:
                    break
                selected.append(record)
                selected_paths.add(record['path'])
                manifest.append({
                    **manifest_row(record, 'balanced14'),
                    'selection_note': 'replacement_pair',
                    'replacement_for_emotion': ID_TO_EMOTION[replacement_for],
                })

    if len(selected) < 14:
        raise RuntimeError(f"Only found {len(selected)} balanced14 records")
    return selected[:14], manifest[:14], missing_labels


def manifest_row(record, sampler):
    return {
        'sampler': sampler,
        'user_id': record['user_id'],
        'emotion': record['emotion'],
        'label': record['label'],
        'camera_index': record['camera_index'],
        'capture_key': record['capture_key'],
        'original_set': record.get('original_set', ''),
        'filename': record['filename'],
        'path': record['path'],
    }


def create_abc14_splits(user_records, leave_out_user, seed=42, neutral_shots=12,
                        balanced_replacement_policy='keep_14'):
    """Create matched LOO splits for A/base, B/neutral-auth-like, and C/balanced14."""
    all_users = [u for u in user_records.keys() if u != str(leave_out_user)]
    rng = random.Random(seed)
    rng.shuffle(all_users)

    train_users = set(all_users[:28])
    val_users = set(all_users[28:36])

    train_records, val_records, personal_records = [], [], []
    for user_id, records in user_records.items():
        if user_id == str(leave_out_user):
            personal_records.extend(records)
        elif user_id in train_users:
            train_records.extend(records)
        elif user_id in val_users:
            val_records.extend(records)

    val_records_balanced = balance_records(val_records, seed)
    neutral_records, neutral_manifest = select_neutral_auth_like_records(
        personal_records, shots=neutral_shots, seed=seed
    )
    balanced_records, balanced_manifest, missing_labels = select_balanced14_records(
        personal_records, seed=seed, replacement_policy=balanced_replacement_policy
    )

    calibration_paths = {record['path'] for record in neutral_records + balanced_records}
    common_test_records = [record for record in personal_records if record['path'] not in calibration_paths]
    other_emotion_test_records = [record for record in common_test_records if record['label'] != NEUTRAL_LABEL]

    print(f"  [Split] Train: {len(train_users)} users ({len(train_records)} images)")
    print(f"  [Split] Val: {len(val_users)} users ({len(val_records_balanced)} images, balanced)")
    print(f"  [Split] LOO User {leave_out_user}: {len(personal_records)} total images")
    print(f"         neutral_auth_like: {len(neutral_records)} calibration images")
    print(f"         balanced14: {len(balanced_records)} calibration images")
    print(f"         common test: {len(common_test_records)} images ({len(other_emotion_test_records)} non-neutral)")
    if missing_labels:
        missing = ', '.join(ID_TO_EMOTION[label] for label in missing_labels)
        print(f"         balanced14 missing emotions filled by replacement: {missing}")

    return {
        'train': records_to_data(train_records),
        'val': records_to_data(val_records_balanced),
        'test_common': records_to_data(common_test_records),
        'test_other': records_to_data(other_emotion_test_records),
        'neutral_personal': records_to_data(neutral_records),
        'balanced_personal': records_to_data(balanced_records),
        'neutral_manifest': neutral_manifest,
        'balanced_manifest': balanced_manifest,
        'train_users': len(train_users),
        'val_users': len(val_users),
        'personal_total': len(personal_records),
        'neutral_calibration': len(neutral_records),
        'balanced_calibration': len(balanced_records),
        'personal_test': len(common_test_records),
        'missing_balanced_labels': missing_labels,
        'manifest_rows': neutral_manifest + balanced_manifest,
    }


def create_model(num_classes=7, pretrained=True, model_name='efficientnet_b0'):
    """Create a timm image classifier with a 7-emotion head."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def is_classifier_parameter(name):
    name_parts = name.split('.')
    return (
        'classifier' in name_parts
        or name_parts[0] in {'fc', 'head'}
        or name.endswith('fc.weight')
        or name.endswith('fc.bias')
    )


def get_classifier_parameters(model):
    params = [param for name, param in model.named_parameters() if is_classifier_parameter(name)]
    if params:
        return params
    named_params = list(model.named_parameters())
    return [named_params[-1][1]] if named_params else []


def setup_partial_unfreeze(model, unfreeze_ratio='third'):
    """
    Freeze layers based on unfreeze ratio.
    
    EfficientNet-B0 keeps the previous block-based behavior. Other backbones
    fall back to unfreezing the last fraction of named parameters plus the
    classifier/head.
    """
    if unfreeze_ratio == 'full':
        for param in model.parameters():
            param.requires_grad = True
        print(f"  [Unfreeze: full] All layers trainable")
        return model
    
    if unfreeze_ratio not in {'third', 'half'}:
        raise ValueError(f"Unknown unfreeze_ratio: {unfreeze_ratio}")
    
    for param in model.parameters():
        param.requires_grad = False

    named_params = list(model.named_parameters())
    has_efficientnet_blocks = any(name.startswith('blocks.') for name, _ in named_params)
    if has_efficientnet_blocks:
        if unfreeze_ratio == 'third':
            unfreeze_blocks = [5, 6]
        else:
            unfreeze_blocks = [3, 4, 5, 6]
        for name, param in named_params:
            if is_classifier_parameter(name):
                param.requires_grad = True
            for block_num in unfreeze_blocks:
                if f'blocks.{block_num}' in name:
                    param.requires_grad = True
        print(f"  [Unfreeze: {unfreeze_ratio}] EfficientNet blocks {unfreeze_blocks} + classifier")
    else:
        fraction = 1 / 3 if unfreeze_ratio == 'third' else 1 / 2
        start_idx = max(0, int(len(named_params) * (1 - fraction)))
        for idx, (name, param) in enumerate(named_params):
            if idx >= start_idx or is_classifier_parameter(name):
                param.requires_grad = True
        print(f"  [Unfreeze: {unfreeze_ratio}] Last {fraction:.0%} of parameters + classifier/head")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [Unfreeze: {unfreeze_ratio}] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return model


def calculate_class_weights(labels, device):
    """Calculate inverse class frequencies for weighted loss."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    num_classes = 7
    
    weights = []
    for i in range(num_classes):
        count = counts[i] if i in counts else 0
        if count > 0:
            weights.append(total / (num_classes * count))
        else:
            weights.append(1.0)
            
    return torch.FloatTensor(weights).to(device)


def get_pip_requirements():
    return [
        f"torch=={torch.__version__}",
        f"torchvision=={torchvision.__version__}",
        f"timm=={timm.__version__}",
        f"numpy=={np.__version__}",
    ]


def get_input_example(val_loader, device):
    for images, _ in val_loader:
        return images[:1].to(device)
    return None


def build_signature(model, input_example):
    if input_example is None:
        return None, None
    model.eval()
    with torch.no_grad():
        output = model(input_example)
    input_example_np = input_example.cpu().numpy()
    output_np = output.cpu().numpy()
    signature = infer_signature(input_example_np, output_np)
    return input_example_np, signature


def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, 
                               device, epochs, patience=10, stage_name="Training"):
    """Training loop with early stopping, Mixed Precision, and MLflow logging."""
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    # Mixed Precision for faster training on RTX 4090
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixed Precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
        val_f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0
        
        mlflow.log_metrics({
            f"{stage_name}_train_loss": avg_loss,
            f"{stage_name}_train_acc": train_acc,
            f"{stage_name}_val_acc": val_acc,
            f"{stage_name}_val_f1": val_f1,
        }, step=epoch)
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping] at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


def stage1_base_training(train_data, val_data, device, epochs=50, batch_size=64, lr=1e-4,
                         patience=10, model_name='efficientnet_b0',
                         num_workers=8, persistent_workers=True):
    """Stage 1: Train base model on 28 users. Returns model and training time."""
    start_time = time.time()
    print(f"\n[Stage 1] Training Base Model ({model_name}, 28 users)...")
    
    train_dataset = EmojiHeroDataset(train_data['images'], train_data['labels'], transform=train_transform)
    val_dataset = EmojiHeroDataset(val_data['images'], val_data['labels'], transform=val_transform)
    
    train_loader = make_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=persistent_workers
    )
    val_loader = make_data_loader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent_workers
    )
    
    model = create_model(model_name=model_name).to(device)
    
    class_weights = calculate_class_weights(train_data['labels'], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model, best_val_acc = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion,
        device, epochs=epochs, patience=patience, stage_name="Stage1"
    )
    
    training_time = time.time() - start_time
    mlflow.log_metric("stage1_best_val_acc", best_val_acc)
    mlflow.log_metric("stage1_training_time_sec", training_time)
    print(f"  [Stage 1 Complete] Best Val Acc: {best_val_acc*100:.2f}%, Time: {training_time:.1f}s")
    
    return model, training_time


def stage2_personalize(base_model, train_data, val_data, personal_data, device, 
                       epochs=50, batch_size=64, lr=1e-5, patience=10, 
                       classifier_only=False, unfreeze_ratio='full',
                       train_source='base_plus_personal',
                       num_workers=8, persistent_workers=True):
    """
    Stage 2: Personalize model with Base + Personal data. Returns model and training time.
    
    Args:
        classifier_only: If True, only train classifier layer.
        unfreeze_ratio: 'full', 'half' (2/3), or 'third' (1/3) of layers to unfreeze.
    """
    start_time = time.time()
    
    if classifier_only:
        mode = "Classifier Only"
    elif unfreeze_ratio == 'third':
        mode = "Unfreeze 1/3 (blocks.5, 6)"
    elif unfreeze_ratio == 'half':
        mode = "Unfreeze 2/3 (blocks.3-6)"
    else:
        mode = "Full Layers"
    print(f"\n[Stage 2] Personalization ({mode})...")
    
    if train_source == 'base_plus_personal':
        combined_images = train_data['images'] + personal_data['images']
        combined_labels = train_data['labels'] + personal_data['labels']
    elif train_source == 'personal_only':
        combined_images = personal_data['images']
        combined_labels = personal_data['labels']
    else:
        raise ValueError(f"Unknown Stage 2 train_source: {train_source}")
    print(f"  [Stage 2 Data] train_source={train_source}, personal={len(personal_data['images'])}, total={len(combined_images)}")
    
    train_dataset = EmojiHeroDataset(combined_images, combined_labels, transform=train_transform)
    val_dataset = EmojiHeroDataset(val_data['images'], val_data['labels'], transform=val_transform)
    
    train_loader = make_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=persistent_workers
    )
    val_loader = make_data_loader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent_workers
    )
    
    model = base_model.to(device)
    
    if classifier_only:
        # Freeze backbone, train only classifier
        for param in model.parameters():
            param.requires_grad = False
        classifier_params = get_classifier_parameters(model)
        for param in classifier_params:
            param.requires_grad = True
        optimizer = optim.Adam(classifier_params, lr=lr)
    elif unfreeze_ratio in ['third', 'half']:
        # Train only partial layers based on ratio
        model = setup_partial_unfreeze(model, unfreeze_ratio=unfreeze_ratio)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)
    else:
        # Train all layers
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Keep class weights for methodological consistency across stages.
    class_weights = calculate_class_weights(combined_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model, best_val_acc = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion,
        device, epochs, patience, "Stage2"
    )
    
    training_time = time.time() - start_time
    mlflow.log_metric("stage2_best_val_acc", best_val_acc)
    mlflow.log_metric("stage2_training_time_sec", training_time)
    print(f"  [Stage 2 Complete] Best Val Acc: {best_val_acc*100:.2f}%, Time: {training_time:.1f}s")
    
    return model, training_time


def evaluate_model(model, test_data, device, batch_size=32, stage_name="test"):
    """Evaluate model on test data."""
    if len(test_data['images']) == 0:
        return {'accuracy': 0, 'f1': 0, 'n_samples': 0, 'confusion_matrix': []}
    
    dataset = EmojiHeroDataset(test_data['images'], test_data['labels'], transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    mlflow.log_metric(f"{stage_name}_test_accuracy", acc)
    mlflow.log_metric(f"{stage_name}_test_f1", f1)
    
    return {'accuracy': acc, 'f1': f1, 'n_samples': len(all_labels), 'confusion_matrix': cm.tolist()}


def evaluate_common_and_other(model, splits, device, stage_name):
    all_results = evaluate_model(model, splits['test_common'], device, stage_name=stage_name)
    other_results = evaluate_model(model, splits['test_other'], device, stage_name=f"{stage_name}_other")
    return {
        'accuracy': all_results['accuracy'],
        'f1': all_results['f1'],
        'n_samples': all_results['n_samples'],
        'confusion_matrix': all_results['confusion_matrix'],
        'other_accuracy': other_results['accuracy'],
        'other_f1': other_results['f1'],
        'other_n_samples': other_results['n_samples'],
        'other_confusion_matrix': other_results['confusion_matrix'],
    }


def write_csv_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def summarize_metric_rows(metric_rows):
    summary = {}
    grouped = defaultdict(list)
    for row in metric_rows:
        grouped[(row['model'], row['condition'])].append(row)
    for (model_name, condition), rows in grouped.items():
        key = f"{model_name}/{condition}"
        summary[key] = {
            'n_folds': len(rows),
            'mean_accuracy': float(np.mean([row['accuracy'] for row in rows])),
            'std_accuracy': float(np.std([row['accuracy'] for row in rows])),
            'mean_f1': float(np.mean([row['f1'] for row in rows])),
            'std_f1': float(np.std([row['f1'] for row in rows])),
            'mean_other_accuracy': float(np.mean([row['other_accuracy'] for row in rows])),
            'std_other_accuracy': float(np.std([row['other_accuracy'] for row in rows])),
            'mean_other_f1': float(np.mean([row['other_f1'] for row in rows])),
            'std_other_f1': float(np.std([row['other_f1'] for row in rows])),
        }
    return summary


def add_metric_row(metric_rows, model_name, leave_out_user, condition, result, splits,
                   stage1_time_sec=0, stage2_time_sec=0, personal_calibration_used=0):
    total_time_sec = stage1_time_sec + stage2_time_sec
    metric_rows.append({
        'model': model_name,
        'leave_out_user': str(leave_out_user),
        'condition': condition,
        'accuracy': result['accuracy'],
        'f1': result['f1'],
        'n_samples': result['n_samples'],
        'other_accuracy': result['other_accuracy'],
        'other_f1': result['other_f1'],
        'other_n_samples': result['other_n_samples'],
        'elapsed_sec': total_time_sec,
        'stage1_time_sec': stage1_time_sec,
        'stage2_time_sec': stage2_time_sec,
        'total_time_sec': total_time_sec,
        'train_images': len(splits['train']['images']),
        'val_images': len(splits['val']['images']),
        'personal_calibration_used': personal_calibration_used,
        'neutral_calibration': splits['neutral_calibration'],
        'balanced_calibration': splits['balanced_calibration'],
        'missing_balanced_emotions': ','.join(ID_TO_EMOTION[label] for label in splits['missing_balanced_labels']),
        'stage2_train_source': '',
    })


def run_abc14(args):
    """Run A/B/C LOO personalization experiments.

    A: base model, no held-out-user adaptation.
    B: neutral-auth-like adaptation with neutral held-out-user samples only.
    C: balanced-14 adaptation with 7 emotion x 2 camera-view samples where possible.
    """
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = resolve_si_root(args.data_dir)
    data_layout = infer_data_layout(data_dir) if args.data_layout == 'auto' else args.data_layout
    print(f"Collecting records from: {data_dir}")
    print(f"Data layout: {data_layout}")
    records = collect_records(data_dir, data_layout)
    user_records = records_by_user(records)
    all_users = sorted(user_records.keys(), key=numeric_sort_key)
    print(f"Found {len(records)} images from {len(all_users)} users")

    if args.user:
        fold_users = [str(args.user)]
    elif args.folds == 'all':
        fold_users = all_users
    else:
        fold_users = parse_csv_option(args.folds)
    if args.max_folds:
        fold_users = fold_users[:args.max_folds]

    model_names = parse_csv_option(args.models) or ['efficientnet_b0']
    neutral_shots_list = parse_int_csv_option(args.neutral_shots) or [12]
    max_neutral_shots = max(neutral_shots_list)
    finetune_modes = parse_csv_option(args.abc_finetune_modes) or ['full', 'half']
    for mode in finetune_modes:
        if mode not in {'full', 'half', 'third', 'classifier_only'}:
            raise ValueError(f"Unsupported abc fine-tune mode: {mode}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or 'abc14'
    results_dir = Path(args.results_dir) / f"{run_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / 'fold_metrics.csv'
    manifest_path = results_dir / 'calibration_manifest.csv'
    results_path = results_dir / 'results.json'

    metric_rows = []
    manifest_rows = []
    results = {
        'timestamp': timestamp,
        'mode': 'abc14',
        'run_name': run_name,
        'data_dir': str(data_dir),
        'data_layout': data_layout,
        'models': model_names,
        'folds': fold_users,
        'neutral_shots': neutral_shots_list,
        'max_neutral_shots': max_neutral_shots,
        'balanced_replacement_policy': args.balanced_replacement_policy,
        'stage2_train_source': args.stage2_train_source,
        'users': {},
        'summary': {},
    }

    manifest_fields = [
        'model', 'leave_out_user', 'sampler', 'selection_note', 'replacement_for_emotion',
        'user_id', 'emotion', 'label', 'camera_index', 'capture_key', 'original_set',
        'filename', 'path',
    ]
    metric_fields = [
        'model', 'leave_out_user', 'condition', 'accuracy', 'f1', 'n_samples',
        'other_accuracy', 'other_f1', 'other_n_samples', 'elapsed_sec',
        'stage1_time_sec', 'stage2_time_sec', 'total_time_sec',
        'train_images', 'val_images', 'personal_calibration_used',
        'neutral_calibration', 'balanced_calibration',
        'missing_balanced_emotions', 'stage2_train_source',
    ]

    for leave_out_user in fold_users:
        splits = create_abc14_splits(
            user_records,
            leave_out_user,
            seed=seed + int(leave_out_user) if str(leave_out_user).isdigit() else seed,
            neutral_shots=max_neutral_shots,
            balanced_replacement_policy=args.balanced_replacement_policy,
        )
        expanded_manifest_rows = []
        for neutral_shots in neutral_shots_list:
            for row in splits['neutral_manifest'][:neutral_shots]:
                expanded_manifest_rows.append({
                    **row,
                    'sampler': f"neutral_auth_like_{neutral_shots}",
                })
        expanded_manifest_rows.extend(splits['balanced_manifest'])
        for row in expanded_manifest_rows:
            for model_name in model_names:
                manifest_rows.append({
                    **row,
                    'model': model_name,
                    'leave_out_user': str(leave_out_user),
                })

        if args.dry_run:
            continue

    if args.dry_run:
        write_csv_rows(manifest_path, manifest_fields, manifest_rows)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Dry Run] Wrote calibration manifest: {manifest_path}")
        print(f"[Dry Run] Results metadata: {results_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mlruns_dir = PROJECT_DIR / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
    mlflow.set_experiment("loo_cv_abc14")

    for model_name in model_names:
        results['users'].setdefault(model_name, {})
        for leave_out_user in fold_users:
            print(f"\n{'='*70}")
            print(f"ABC14: model={model_name}, leaving out user {leave_out_user}")
            print(f"{'='*70}")
            splits = create_abc14_splits(
                user_records,
                leave_out_user,
                seed=seed + int(leave_out_user) if str(leave_out_user).isdigit() else seed,
                neutral_shots=max_neutral_shots,
                balanced_replacement_policy=args.balanced_replacement_policy,
            )

            with mlflow.start_run(run_name=f"{run_name}_{model_name}_User_{leave_out_user}_abc14"):
                mlflow.log_params({
                    'model': model_name,
                    'leave_out_user': leave_out_user,
                    'mode': 'abc14',
                    'train_users': splits['train_users'],
                    'val_users': splits['val_users'],
                    'train_images': len(splits['train']['images']),
                    'val_images': len(splits['val']['images']),
                    'test_images': len(splits['test_common']['images']),
                    'neutral_shots': ','.join(str(item) for item in neutral_shots_list),
                    'max_neutral_shots': max_neutral_shots,
                    'stage2_train_source': args.stage2_train_source,
                    'epochs': args.epochs,
                    'patience': args.patience,
                })

                base_model, stage1_time = stage1_base_training(
                    splits['train'], splits['val'], device,
                    epochs=args.epochs, patience=args.patience,
                    batch_size=args.batch_size, model_name=model_name,
                    num_workers=args.num_workers,
                    persistent_workers=not args.no_persistent_workers
                )
                base_result = evaluate_common_and_other(base_model, splits, device, 'A_base')
                add_metric_row(metric_rows, model_name, leave_out_user, 'A_base', base_result,
                               splits, stage1_time, 0, 0)
                metric_rows[-1]['stage2_train_source'] = 'none'
                base_model_state = copy.deepcopy(base_model.state_dict())

                user_result = {
                    'A_base': base_result,
                    'stage1_time_sec': stage1_time,
                    'conditions': {},
                    'missing_balanced_emotions': [ID_TO_EMOTION[label] for label in splits['missing_balanced_labels']],
                }

                sampler_conditions = []
                for neutral_shots in neutral_shots_list:
                    neutral_records = splits['neutral_personal']['records'][:neutral_shots]
                    sampler_conditions.append((
                        f"B_neutral{neutral_shots}",
                        records_to_data(neutral_records),
                    ))
                sampler_conditions.append(('C_balanced14', splits['balanced_personal']))

                for sampler_name, personal_data in sampler_conditions:
                    for mode in finetune_modes:
                        condition = f"{sampler_name}_{mode}"
                        print(f"\n--- {condition} ---")
                        model_ft = create_model(model_name=model_name).to(device)
                        model_ft.load_state_dict(base_model_state)
                        classifier_only = mode == 'classifier_only'
                        unfreeze_ratio = 'full' if classifier_only else mode
                        personalized_model, stage2_time = stage2_personalize(
                            model_ft, splits['train'], splits['val'], personal_data, device,
                            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                            classifier_only=classifier_only, unfreeze_ratio=unfreeze_ratio,
                            train_source=args.stage2_train_source,
                            num_workers=args.num_workers,
                            persistent_workers=not args.no_persistent_workers,
                        )
                        condition_result = evaluate_common_and_other(
                            personalized_model, splits, device, condition
                        )
                        add_metric_row(metric_rows, model_name, leave_out_user, condition,
                                       condition_result, splits, stage1_time, stage2_time,
                                       len(personal_data['images']))
                        metric_rows[-1]['stage2_train_source'] = args.stage2_train_source
                        user_result['conditions'][condition] = {
                            **condition_result,
                            'stage2_time_sec': stage2_time,
                            'total_time_sec': stage1_time + stage2_time,
                        }

                results['users'][model_name][str(leave_out_user)] = user_result
                write_csv_rows(metrics_path, metric_fields, metric_rows)
                write_csv_rows(manifest_path, manifest_fields, manifest_rows)
                results['summary'] = summarize_metric_rows(metric_rows)
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

    results['summary'] = summarize_metric_rows(metric_rows)
    write_csv_rows(metrics_path, metric_fields, metric_rows)
    write_csv_rows(manifest_path, manifest_fields, manifest_rows)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults directory: {results_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Calibration manifest: {manifest_path}")


def run_loo_cv(args):
    """Run Leave-One-Out Cross-Validation with 2-stage pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if args.retrain:
        mode_str = "Retrain from Scratch"
    elif args.classifier_only:
        mode_str = "Classifier Only"
    else:
        mode_str = f"Unfreeze: {args.unfreeze_ratio}"
    print(f"Mode: {mode_str}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.retrain:
        mode_suffix = "retrain"
    elif args.classifier_only:
        mode_suffix = "classifier"
    else:
        mode_suffix = args.unfreeze_ratio  # 'full', 'half', or 'third'
    results_file = RESULTS_DIR / f"loo_2stage_{mode_suffix}_{timestamp}.json"
    
    mlruns_dir = PROJECT_DIR / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
    mlflow.set_experiment("loo_cv_2stage")
    
    print("Collecting data...")
    user_data = collect_all_data()
    all_users = sorted(user_data.keys(), key=int)
    print(f"Found {len(all_users)} users")
    
    results = {'timestamp': timestamp, 'mode': mode_suffix, 'users': {}, 'summary': {}}
    
    for leave_out_user in all_users:
        if args.user and str(leave_out_user) != str(args.user):
            continue
        
        print(f"\n{'='*60}")
        print(f"LOO: Leaving out User {leave_out_user}")
        print(f"{'='*60}")
        
        splits = create_loo_splits(user_data, leave_out_user)
        personal_data = splits['personal']
        val_loader_for_signature = DataLoader(
            EmojiHeroDataset(splits['val']['images'], splits['val']['labels'], transform=val_transform),
            batch_size=64, shuffle=False, num_workers=0
        )
        input_example = get_input_example(val_loader_for_signature, device)
        pip_requirements = get_pip_requirements()
        
        with mlflow.start_run(run_name=f"LOO_User_{leave_out_user}_{mode_suffix}"):
            mlflow.log_params({
                "leave_out_user": leave_out_user,
                "mode": mode_suffix,
                "train_users": splits['train_users'],
                "val_users": splits['val_users'],
                "train_images": len(splits['train']['images']),
                "personal_images": len(personal_data['images']),
                "epochs": args.epochs,
                "patience": args.patience,
                "classifier_only": args.classifier_only,
                "unfreeze_ratio": args.unfreeze_ratio,
                "retrain": args.retrain,
            })
            
            # RETRAIN MODE: Train from scratch with combined data
            if args.retrain:
                print(f"\n[RETRAIN MODE] Training from ImageNet weights with combined data...")
                
                # Combine train data with personal calibration data (same as Stage 2)
                combined_images = splits['train']['images'] + personal_data['images']
                combined_labels = splits['train']['labels'] + personal_data['labels']
                
                print(f"  Train data: {len(splits['train']['images'])} (base) + {len(personal_data['images'])} (personal) = {len(combined_images)} total")
                
                train_dataset = EmojiHeroDataset(combined_images, combined_labels, transform=train_transform)
                val_dataset = EmojiHeroDataset(splits['val']['images'], splits['val']['labels'], transform=val_transform)
                
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
                val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
                
                # Create fresh model from ImageNet weights
                start_time = time.time()
                model = create_model(pretrained=True).to(device)
                
                class_weights = calculate_class_weights(combined_labels, device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                
                model, best_val_acc = train_with_early_stopping(
                    model, train_loader, val_loader, optimizer, criterion,
                    device, epochs=args.epochs, patience=args.patience, stage_name="Retrain"
                )
                
                retrain_time = time.time() - start_time
                retrain_results = evaluate_model(model, splits['test'], device, stage_name="retrain")
                
                print(f"[RETRAIN] Test Acc: {retrain_results['accuracy']*100:.2f}%, Training Time: {retrain_time:.1f}s")
                
                mlflow.log_metrics({
                    "retrain_training_time_sec": retrain_time,
                    "retrain_test_accuracy": retrain_results['accuracy'],
                    "retrain_best_val_acc": best_val_acc,
                })
                
                # Log model
                retrain_input_example, retrain_signature = build_signature(model, input_example)
                mlflow_pytorch.log_model(
                    model,
                    name="model_retrain",
                    input_example=retrain_input_example,
                    signature=retrain_signature,
                    pip_requirements=pip_requirements,
                )
                
                # Store results (use retrain as both stage1 and stage2 for consistency)
                stage1_results = retrain_results
                stage2_results = retrain_results
                stage1_time = retrain_time
                stage2_time = 0
                improvement = 0  # No improvement since single training
                
            else:
                # STANDARD 2-STAGE PIPELINE
                # Stage 1: Train base model
                base_model, stage1_time = stage1_base_training(
                    splits['train'], splits['val'], device,
                    epochs=args.epochs, patience=args.patience
                )
                stage1_results = evaluate_model(base_model, splits['test'], device, stage_name="stage1")
                print(f"[Stage 1] Test Acc: {stage1_results['accuracy']*100:.2f}%, Training Time: {stage1_time:.1f}s")
                
                # Log Stage 1 model to MLflow
                stage1_input_example, stage1_signature = build_signature(base_model, input_example)
                mlflow_pytorch.log_model(
                    base_model,
                    name="model_stage1",
                    input_example=stage1_input_example,
                    signature=stage1_signature,
                    pip_requirements=pip_requirements,
                )
                
                # Skip Stage 2 if --stage1-only flag is set
                if args.stage1_only:
                    print(f"\n[Stage 1 Only Mode] Skipping Stage 2 (Personalization)")
                    stage2_results = {'accuracy': 0, 'f1': 0, 'n_samples': 0}
                    stage2_time = 0
                    improvement = 0
                    mlflow.log_metrics({
                        "final_stage1_acc": stage1_results['accuracy'],
                        "stage1_only": 1,
                        "total_training_time_sec": stage1_time,
                    })
                else:
                    # Stage 2: Personalize
                    personalized_model, stage2_time = stage2_personalize(
                        base_model, splits['train'], splits['val'], personal_data, device,
                        epochs=args.epochs, patience=args.patience, 
                        classifier_only=args.classifier_only, unfreeze_ratio=args.unfreeze_ratio
                    )
                    stage2_results = evaluate_model(personalized_model, splits['test'], device, stage_name="stage2")
                    print(f"[Stage 2] Test Acc: {stage2_results['accuracy']*100:.2f}%, Training Time: {stage2_time:.1f}s")
                    stage2_input_example, stage2_signature = build_signature(personalized_model, input_example)
                    mlflow_pytorch.log_model(
                        personalized_model,
                        name="model_stage2",
                        input_example=stage2_input_example,
                        signature=stage2_signature,
                        pip_requirements=pip_requirements,
                    )
                    
                    # Summary
                    total_time = stage1_time + stage2_time
                    improvement = stage2_results['accuracy'] - stage1_results['accuracy']
                    print(f"\nUser {leave_out_user} Results:")
                    print(f"   Stage 1 (Base):        {stage1_results['accuracy']*100:.2f}% (Time: {stage1_time:.1f}s)")
                    print(f"   Stage 2 (Personalize): {stage2_results['accuracy']*100:.2f}% ({improvement*100:+.2f}%p, Time: {stage2_time:.1f}s)")
                    print(f"   Total Training Time:   {total_time:.1f}s")
                    
                    mlflow.log_metrics({
                        "final_stage1_acc": stage1_results['accuracy'],
                        "final_stage2_acc": stage2_results['accuracy'],
                        "improvement": improvement,
                        "total_training_time_sec": total_time,
                    })
        
        results['users'][leave_out_user] = {
            'stage1_accuracy': stage1_results['accuracy'],
            'stage1_f1': stage1_results['f1'],
            'stage2_accuracy': stage2_results['accuracy'],
            'stage2_f1': stage2_results['f1'],
            'improvement': improvement,
            'n_samples': stage1_results['n_samples'],
            'stage1_time_sec': stage1_time,
            'stage2_time_sec': stage2_time,
            'total_time_sec': stage1_time + stage2_time,
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary statistics
    if results['users']:
        s1_accs = [r['stage1_accuracy'] for r in results['users'].values()]
        s2_accs = [r['stage2_accuracy'] for r in results['users'].values()]
        improvements = [r['improvement'] for r in results['users'].values()]
        s1_times = [r['stage1_time_sec'] for r in results['users'].values()]
        s2_times = [r['stage2_time_sec'] for r in results['users'].values()]
        total_times = [r['total_time_sec'] for r in results['users'].values()]
        
        import numpy as np
        results['summary'] = {
            'n_users': len(results['users']),
            'mode': mode_suffix,
            'mean_stage1_acc': float(np.mean(s1_accs)),
            'std_stage1_acc': float(np.std(s1_accs)),
            'mean_stage2_acc': float(np.mean(s2_accs)),
            'std_stage2_acc': float(np.std(s2_accs)),
            'mean_improvement': float(np.mean(improvements)),
            'std_improvement': float(np.std(improvements)),
            'mean_stage1_time_sec': float(np.mean(s1_times)),
            'mean_stage2_time_sec': float(np.mean(s2_times)),
            'mean_total_time_sec': float(np.mean(total_times)),
        }
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY ({mode_suffix.upper()} MODE)")
        print(f"{'='*60}")
        print(f"Mean Stage 1 (Base):        {results['summary']['mean_stage1_acc']*100:.2f}% ± {results['summary']['std_stage1_acc']*100:.2f}%")
        print(f"Mean Stage 2 (Personalize): {results['summary']['mean_stage2_acc']*100:.2f}% ± {results['summary']['std_stage2_acc']*100:.2f}%")
        print(f"Mean Improvement:           {results['summary']['mean_improvement']*100:+.2f}%p ± {results['summary']['std_improvement']*100:.2f}%p")
        print(f"Mean Training Time:         {results['summary']['mean_total_time_sec']:.1f}s (S1: {results['summary']['mean_stage1_time_sec']:.1f}s, S2: {results['summary']['mean_stage2_time_sec']:.1f}s)")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


def run_compare_modes(args):
    """
    Compare ALL 4 methods using the SAME data splits for fair comparison:
    1. Base (Stage 1 only)
    2. Retrain (from ImageNet with combined data)
    3. Full Fine-tuning (all layers)
    4. Half Fine-tuning (blocks.3-6 only)
    
    Full and Half Fine-tuning share the SAME Stage 1 base model.
    """
    import copy
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    print("COMPARE ALL MODES: Base vs Retrain vs Full vs Half")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"loo_compare_all_{timestamp}.json"
    
    mlruns_dir = PROJECT_DIR / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
    mlflow.set_experiment("loo_cv_compare_all")
    
    print("Collecting data...")
    user_data = collect_all_data()
    all_users = sorted(user_data.keys(), key=int)
    print(f"Found {len(all_users)} users")
    
    results = {
        'timestamp': timestamp, 
        'mode': 'compare_all', 
        'users': {}, 
        'summary': {}
    }
    
    for leave_out_user in all_users:
        if args.user and str(leave_out_user) != str(args.user):
            continue
        
        print(f"\n{'='*60}")
        print(f"LOO: Leaving out User {leave_out_user}")
        print(f"{'='*60}")
        
        splits = create_loo_splits(user_data, leave_out_user)
        personal_data = splits['personal']
        
        with mlflow.start_run(run_name=f"LOO_User_{leave_out_user}_compare_all"):
            mlflow.log_params({
                "leave_out_user": leave_out_user,
                "mode": "compare_all",
                "train_users": splits['train_users'],
                "val_users": splits['val_users'],
                "train_images": len(splits['train']['images']),
                "personal_images": len(personal_data['images']),
                "epochs": args.epochs,
                "patience": args.patience,
            })
            
            # =============================================
            # 1. Stage 1: Train BASE model (shared for Full/Half)
            # =============================================
            print("\n--- [1/4] Training BASE Model (Stage 1) ---")
            base_model, stage1_time = stage1_base_training(
                splits['train'], splits['val'], device,
                epochs=args.epochs, patience=args.patience
            )
            base_results = evaluate_model(base_model, splits['test'], device, stage_name="base")
            print(f"[BASE] Test Acc: {base_results['accuracy']*100:.2f}%, Training Time: {stage1_time:.1f}s")
            
            # Save base model state for Full/Half comparison
            base_model_state = copy.deepcopy(base_model.state_dict())
            
            # =============================================
            # 2. RETRAIN: Train from ImageNet with combined data
            # =============================================
            print("\n--- [2/4] Training RETRAIN Model (from ImageNet) ---")
            combined_images = splits['train']['images'] + personal_data['images']
            combined_labels = splits['train']['labels'] + personal_data['labels']
            
            retrain_dataset = EmojiHeroDataset(combined_images, combined_labels, transform=train_transform)
            val_dataset = EmojiHeroDataset(splits['val']['images'], splits['val']['labels'], transform=val_transform)
            
            retrain_loader = DataLoader(retrain_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
            
            retrain_start = time.time()
            retrain_model = create_model(pretrained=True).to(device)
            
            class_weights = calculate_class_weights(combined_labels, device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            for param in retrain_model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(retrain_model.parameters(), lr=1e-4)
            
            retrain_model, _ = train_with_early_stopping(
                retrain_model, retrain_loader, val_loader, optimizer, criterion,
                device, epochs=args.epochs, patience=args.patience, stage_name="Retrain"
            )
            retrain_time = time.time() - retrain_start
            retrain_results = evaluate_model(retrain_model, splits['test'], device, stage_name="retrain")
            print(f"[RETRAIN] Test Acc: {retrain_results['accuracy']*100:.2f}%, Training Time: {retrain_time:.1f}s")
            
            # =============================================
            # 3. FULL Fine-tuning (from Base model)
            # =============================================
            print("\n--- [3/4] Training FULL Fine-tuning ---")
            model_full = create_model().to(device)
            model_full.load_state_dict(base_model_state)
            
            personalized_full, full_time = stage2_personalize(
                model_full, splits['train'], splits['val'], personal_data, device,
                epochs=args.epochs, patience=args.patience, 
                classifier_only=False, unfreeze_ratio='full'
            )
            full_results = evaluate_model(personalized_full, splits['test'], device, stage_name="full")
            print(f"[FULL] Test Acc: {full_results['accuracy']*100:.2f}%, Training Time: {full_time:.1f}s")
            
            # =============================================
            # 4. HALF Fine-tuning (from SAME Base model)
            # =============================================
            print("\n--- [4/4] Training HALF Fine-tuning ---")
            model_half = create_model().to(device)
            model_half.load_state_dict(base_model_state)
            
            personalized_half, half_time = stage2_personalize(
                model_half, splits['train'], splits['val'], personal_data, device,
                epochs=args.epochs, patience=args.patience, 
                classifier_only=False, unfreeze_ratio='half'
            )
            half_results = evaluate_model(personalized_half, splits['test'], device, stage_name="half")
            print(f"[HALF] Test Acc: {half_results['accuracy']*100:.2f}%, Training Time: {half_time:.1f}s")
            
            # =============================================
            # Summary for this user
            # =============================================
            print(f"\n{'='*50}")
            print(f"User {leave_out_user} Comparison Results:")
            print(f"{'='*50}")
            print(f"  [BASE]    {base_results['accuracy']*100:.2f}% (Time: {stage1_time:.1f}s)")
            print(f"  [RETRAIN] {retrain_results['accuracy']*100:.2f}% (Time: {retrain_time:.1f}s)")
            print(f"  [FULL]    {full_results['accuracy']*100:.2f}% (Time: {stage1_time + full_time:.1f}s total)")
            print(f"  [HALF]    {half_results['accuracy']*100:.2f}% (Time: {stage1_time + half_time:.1f}s total)")
            print(f"{'='*50}")
            
            mlflow.log_metrics({
                "base_acc": base_results['accuracy'],
                "retrain_acc": retrain_results['accuracy'],
                "full_acc": full_results['accuracy'],
                "half_acc": half_results['accuracy'],
                "base_time": stage1_time,
                "retrain_time": retrain_time,
                "full_total_time": stage1_time + full_time,
                "half_total_time": stage1_time + half_time,
            })
        
        results['users'][leave_out_user] = {
            'base_accuracy': base_results['accuracy'],
            'retrain_accuracy': retrain_results['accuracy'],
            'full_accuracy': full_results['accuracy'],
            'half_accuracy': half_results['accuracy'],
            'n_samples': base_results['n_samples'],
            'base_time_sec': stage1_time,
            'retrain_time_sec': retrain_time,
            'full_time_sec': full_time,
            'half_time_sec': half_time,
            'full_total_time_sec': stage1_time + full_time,
            'half_total_time_sec': stage1_time + half_time,
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary statistics
    if results['users']:
        base_accs = [r['base_accuracy'] for r in results['users'].values()]
        retrain_accs = [r['retrain_accuracy'] for r in results['users'].values()]
        full_accs = [r['full_accuracy'] for r in results['users'].values()]
        half_accs = [r['half_accuracy'] for r in results['users'].values()]
        retrain_times = [r['retrain_time_sec'] for r in results['users'].values()]
        full_total_times = [r['full_total_time_sec'] for r in results['users'].values()]
        half_total_times = [r['half_total_time_sec'] for r in results['users'].values()]
        
        results['summary'] = {
            'n_users': len(results['users']),
            'mean_base_acc': float(np.mean(base_accs)),
            'std_base_acc': float(np.std(base_accs)),
            'mean_retrain_acc': float(np.mean(retrain_accs)),
            'std_retrain_acc': float(np.std(retrain_accs)),
            'mean_full_acc': float(np.mean(full_accs)),
            'std_full_acc': float(np.std(full_accs)),
            'mean_half_acc': float(np.mean(half_accs)),
            'std_half_acc': float(np.std(half_accs)),
            'mean_retrain_time': float(np.mean(retrain_times)),
            'mean_full_total_time': float(np.mean(full_total_times)),
            'mean_half_total_time': float(np.mean(half_total_times)),
        }
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY: All Methods Comparison")
        print(f"{'='*60}")
        print(f"  [BASE]    {results['summary']['mean_base_acc']*100:.2f}% ± {results['summary']['std_base_acc']*100:.2f}%")
        print(f"  [RETRAIN] {results['summary']['mean_retrain_acc']*100:.2f}% ± {results['summary']['std_retrain_acc']*100:.2f}% (Time: {results['summary']['mean_retrain_time']:.1f}s)")
        print(f"  [FULL]    {results['summary']['mean_full_acc']*100:.2f}% ± {results['summary']['std_full_acc']*100:.2f}% (Time: {results['summary']['mean_full_total_time']:.1f}s)")
        print(f"  [HALF]    {results['summary']['mean_half_acc']*100:.2f}% ± {results['summary']['std_half_acc']*100:.2f}% (Time: {results['summary']['mean_half_total_time']:.1f}s)")
        print(f"{'='*60}")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="LOO CV with 2-Stage Pipeline")
    parser.add_argument('--user', type=str, help="Run LOO for specific user only")
    parser.add_argument('--folds', type=str, default='all', help="Comma-separated LOO users or 'all'")
    parser.add_argument('--max-folds', type=int, default=None, help="Optional cap for quick pilots")
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR,
                        help="Dataset root. Can be data/emoji-hero-vr-db-si, data_ori/emoji-hero-vr-db-si, or data_ori.")
    parser.add_argument('--data-layout', type=str, default='auto', choices=['auto', 'flat', 'original'],
                        help="flat: emotion folders directly under root; original: set/emotion folders")
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--models', type=str, default='efficientnet_b0',
                        help="Comma-separated timm model names, e.g. efficientnet_b0,resnet18,convnext_tiny")
    parser.add_argument('--run-name', type=str, default='loo')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--epochs', type=int, default=50, help="Epochs per stage")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8,
                        help="DataLoader workers for training/validation loaders")
    parser.add_argument('--no-persistent-workers', action='store_true',
                        help="Disable persistent DataLoader workers; useful on Windows after worker hangs")
    parser.add_argument('--classifier-only', action='store_true', 
                        help="Stage 2: Train classifier only (default: all layers)")
    parser.add_argument('--unfreeze-ratio', type=str, default='full', choices=['full', 'half', 'third'],
                        help="Stage 2 unfreeze ratio: 'full' (all layers), 'half' (2/3 = blocks.3-6), 'third' (1/3 = blocks.5-6)")
    parser.add_argument('--stage1-only', action='store_true',
                        help="Run only Stage 1 (Base model) for timing measurement")
    parser.add_argument('--retrain', action='store_true',
                        help="Retrain from scratch with combined data (28 users + LOO calibration) for comparison")
    parser.add_argument('--compare-modes', action='store_true',
                        help="Compare Full vs Half fine-tuning using the SAME Stage 1 base model for fair comparison")
    parser.add_argument('--abc14', action='store_true',
                        help="Run A/base, B/neutral-auth-like, C/balanced14 personalization experiments")
    parser.add_argument('--abc-finetune-modes', type=str, default='full,half,classifier_only',
                        help="Comma-separated Stage 2 modes for ABC: full,half,third,classifier_only")
    parser.add_argument('--neutral-shots', type=str, default='2,6,12',
                        help="Comma-separated Neutral image counts for B/neutral-auth-like; max 12 supports all 37 users")
    parser.add_argument('--balanced-replacement-policy', type=str, default='keep_14',
                        choices=['keep_14', 'strict'],
                        help="How C/balanced14 handles missing pairs; keep_14 uses same-user similar-emotion fallbacks")
    parser.add_argument('--stage2-train-source', type=str, default='base_plus_personal',
                        choices=['base_plus_personal', 'personal_only'],
                        help="Stage 2 data: rehearse base train data plus personal samples, or use personal samples only")
    parser.add_argument('--dry-run', action='store_true',
                        help="For --abc14: verify data loading and write calibration manifest without training")
    
    args = parser.parse_args()
    
    if args.abc14:
        run_abc14(args)
    elif args.compare_modes:
        run_compare_modes(args)
    else:
        run_loo_cv(args)


if __name__ == "__main__":
    main()
