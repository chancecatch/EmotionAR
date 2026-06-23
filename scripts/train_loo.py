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
import math
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
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


VERIFICATION_BACKBONE_MODELS = ['resnet18', 'convnext_tiny']


def resolve_model_names(args):
    model_names = parse_csv_option(args.models) or ['efficientnet_b0']
    if getattr(args, 'include_verification_backbones', False):
        for model_name in VERIFICATION_BACKBONE_MODELS:
            if model_name not in model_names:
                model_names.append(model_name)
    return model_names


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


def is_final_projection_parameter(name):
    name_parts = name.split('.')
    return name_parts[0] in {'conv_head', 'bn2'} or name_parts[-1] in {'conv_head', 'bn2'}


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
    mode_aliases = {
        'classifier_head_only': 'classifier_head_only',
        'last_two_blocks_plus_head': 'last_two_blocks_plus_head',
        'third': 'last_two_blocks_plus_head',
        'upper_half_backbone_plus_head': 'upper_half_backbone_plus_head',
        'half': 'upper_half_backbone_plus_head',
        'full_network': 'full_network',
        'full': 'full_network',
    }
    unfreeze_mode = mode_aliases.get(unfreeze_ratio)
    if unfreeze_mode is None:
        raise ValueError(f"Unknown unfreeze_ratio: {unfreeze_ratio}")

    if unfreeze_mode == 'full_network':
        for param in model.parameters():
            param.requires_grad = True
        print(f"  [Unfreeze: {unfreeze_ratio}] All layers trainable")
        return model

    for param in model.parameters():
        param.requires_grad = False

    named_params = list(model.named_parameters())
    has_efficientnet_blocks = any(name.startswith('blocks.') for name, _ in named_params)
    if has_efficientnet_blocks:
        if unfreeze_mode == 'classifier_head_only':
            unfreeze_blocks = []
        elif unfreeze_mode == 'last_two_blocks_plus_head':
            unfreeze_blocks = [5, 6]
        elif unfreeze_mode == 'upper_half_backbone_plus_head':
            unfreeze_blocks = [3, 4, 5, 6]
        else:
            raise ValueError(f"Unsupported EfficientNet unfreeze mode: {unfreeze_mode}")
        for name, param in named_params:
            if is_classifier_parameter(name) or is_final_projection_parameter(name):
                param.requires_grad = True
            for block_num in unfreeze_blocks:
                if f'blocks.{block_num}' in name:
                    param.requires_grad = True
        print(f"  [Unfreeze: {unfreeze_ratio}] EfficientNet blocks {unfreeze_blocks} + final projection + classifier/head")
    else:
        if unfreeze_mode == 'classifier_head_only':
            fraction = 0
        elif unfreeze_mode == 'last_two_blocks_plus_head':
            fraction = 1 / 3
        elif unfreeze_mode == 'upper_half_backbone_plus_head':
            fraction = 1 / 2
        else:
            raise ValueError(f"Unsupported generic unfreeze mode: {unfreeze_mode}")
        start_idx = max(0, int(len(named_params) * (1 - fraction)))
        for idx, (name, param) in enumerate(named_params):
            if idx >= start_idx or is_classifier_parameter(name) or is_final_projection_parameter(name):
                param.requires_grad = True
        print(f"  [Unfreeze: {unfreeze_ratio}] Last {fraction:.0%} of parameters + final projection + classifier/head")
    
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
    
    if classifier_only or unfreeze_ratio == 'classifier_head_only':
        mode = "Classifier Only"
    elif unfreeze_ratio in {'third', 'last_two_blocks_plus_head'}:
        mode = "Unfreeze 1/3 (blocks.5, 6)"
    elif unfreeze_ratio in {'half', 'upper_half_backbone_plus_head'}:
        mode = "Unfreeze 2/3 (blocks.3-6)"
    elif unfreeze_ratio in {'full', 'full_network'}:
        mode = "Full Layers"
    else:
        mode = unfreeze_ratio
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
    
    if classifier_only or unfreeze_ratio == 'classifier_head_only':
        # For EfficientNet, this intentionally includes conv_head and bn2
        # with the classifier/head to match the ABCD partial-unfreeze protocol.
        model = setup_partial_unfreeze(model, unfreeze_ratio='classifier_head_only')
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)
    elif unfreeze_ratio in ['third', 'half', 'last_two_blocks_plus_head', 'upper_half_backbone_plus_head']:
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


ABCD_LAYER_MODES = [
    ('classifier_head_only', 'Classifier Head Only'),
    ('last_two_blocks_plus_head', 'Last Two EfficientNet Blocks Plus Head'),
    ('upper_half_backbone_plus_head', 'Upper Half Backbone Plus Head'),
    ('full_network', 'Full Network'),
]

ABCD_C_SHOTS = [7, 14]
ABCD_DEFAULT_C_MODE_KEYS = ['upper_half_backbone_plus_head', 'full_network']
ABCD_D_CONDITION_MODES = [
    ('D_manyshot_upper_half_backbone_plus_head', 'upper_half_backbone_plus_head'),
    ('D_manyshot_full_network', 'full_network'),
    ('D_scratch_upper_reference', 'scratch'),
]
ABCD_DEFAULT_D_CONDITION_KEYS = ['D_manyshot_full_network']

ABCD_CONDITION_FULL_NAMES = {
    'A_base': 'Subject-Independent Base Facial Emotion Recognition Model Without Personalization',
    'B_film7': 'Joint Identity-Emotion Embedding With FiLM-Based User-Profile Conditioning Using 7 Enrollment Images',
    'B_film14': 'Joint Identity-Emotion Embedding With FiLM-Based User-Profile Conditioning Using 14 Enrollment Images',
    'B_proto7': 'Joint Identity-Emotion Embedding With Emotion Prototype User Profile Without Held-Out-User Fine-Tuning Using 7 Enrollment Images',
    'B_proto14': 'Joint Identity-Emotion Embedding With Emotion Prototype User Profile Without Held-Out-User Fine-Tuning Using 14 Enrollment Images',
    'D_manyshot_upper_half_backbone_plus_head': 'Many-Shot User-Specific Fine-Tuning Upper Reference With Upper Half Backbone Plus Head',
    'D_manyshot_full_network': 'Many-Shot User-Specific Fine-Tuning Upper Reference With Full Network',
    'D_scratch_upper_reference': 'Many-Shot Retraining From Scratch Upper Reference',
}

ABCD_FOLD_FIELDS = [
    'Model Name',
    'Experiment Family Full Name',
    'Condition Full Name',
    'User Profile Method Full Name',
    'Enrollment Image Count',
    'Enrollment Image Selection Full Name',
    'Fine-Tuning Strategy Full Name',
    'Layer-Freezing Strategy Full Name',
    'Held-Out User ID',
    'Common Test Image Count',
    'Enrollment Image Count Actually Used',
    'Missing Enrollment Emotion Names',
    'Replacement Policy Full Name',
    'Accuracy',
    'Macro F1 Score',
    'Non-Neutral Accuracy',
    'Non-Neutral Macro F1 Score',
    'Win Tie Loss Compared With Subject-Independent Base',
    'Accuracy Difference From Base Percentage Points',
    'Macro F1 Difference From Base Percentage Points',
    'Base Model Training Time Seconds',
    'Joint Identity Emotion Embedding Training Time Seconds',
    'User Profile Construction Time Seconds',
    'FiLM Conditioning Training Time Seconds',
    'Fine-Tuning Time Seconds',
    'Total Training Or Adaptation Time Seconds',
    'Mean Inference Time Per Image Milliseconds',
    'Trainable Parameter Count',
    'Total Parameter Count',
    'Run Status Full Name',
    'Completed Timestamp',
    'Result Checkpoint Or Resume Source',
]

ABCD_SUMMARY_FIELDS = [
    'Model Name',
    'Experiment Family Full Name',
    'Condition Full Name',
    'Mean Accuracy',
    'Accuracy Standard Deviation',
    'Mean Macro F1 Score',
    'Macro F1 Score Standard Deviation',
    'Mean Non-Neutral Accuracy',
    'Mean Non-Neutral Macro F1 Score',
    'Mean Accuracy Difference From Base Percentage Points',
    'Mean Macro F1 Difference From Base Percentage Points',
    'Users Improved Compared With Base',
    'Users Tied Compared With Base',
    'Users Worse Compared With Base',
    'Mean Total Training Or Adaptation Time Seconds',
    'Mean Joint Identity Emotion Embedding Training Time Seconds',
    'Mean User Profile Construction Time Seconds',
    'Mean Fine-Tuning Time Seconds',
    'Mean Inference Time Per Image Milliseconds',
    'Strict Complete-Emotion User Count',
    'All User Count',
]

ABCD_MANIFEST_FIELDS = [
    'Held-Out User ID',
    'Split Role Full Name',
    'Experiment Family Full Name',
    'Condition Full Name',
    'Enrollment Image Count',
    'Target Emotion Name',
    'Actual Emotion Name',
    'Camera Index',
    'Capture Key',
    'Original Dataset Split Name',
    'Selection Note Full Name',
    'Replacement For Missing Emotion Name',
    'Filename',
    'Path',
]

ABCD_PER_EMOTION_FIELDS = [
    'Model Name',
    'Held-Out User ID',
    'Experiment Family Full Name',
    'Condition Full Name',
    'Emotion Name',
    'Precision',
    'Recall',
    'F1 Score',
    'Support Count',
]


def abcd_c_condition_key(shots, mode_key):
    return f'C_finetune{shots}_{mode_key}'


def abcd_c_condition_full_name(shots, mode_full_name):
    return f'Few-Shot User-Specific Fine-Tuning Using {shots} Enrollment Images With {mode_full_name}'


def abcd_condition_full_name(condition_key):
    if condition_key in ABCD_CONDITION_FULL_NAMES:
        return ABCD_CONDITION_FULL_NAMES[condition_key]
    for shots in ABCD_C_SHOTS:
        prefix = f'C_finetune{shots}_'
        if condition_key.startswith(prefix):
            mode_key = condition_key[len(prefix):]
            mode_name = dict(ABCD_LAYER_MODES).get(mode_key, mode_key)
            return abcd_c_condition_full_name(shots, mode_name)
    return condition_key


def abcd_family_full_name(condition_key):
    if condition_key.startswith('A_'):
        return 'A: Subject-Independent Base Facial Emotion Recognition'
    if condition_key.startswith('B_film'):
        return 'B: User-Profile Conditioning With Joint Identity-Emotion Embedding And FiLM'
    if condition_key.startswith('B_proto'):
        return 'B: Joint Identity-Emotion Embedding With Emotion Prototype User Profile'
    if condition_key.startswith('C_'):
        return 'C: Few-Shot User-Specific Fine-Tuning'
    if condition_key.startswith('D_'):
        return 'D: Many-Shot Upper Reference'
    return 'Unknown Experiment Family'


def abcd_profile_method_full_name(condition_key):
    if condition_key.startswith('B_film'):
        return 'Joint Identity-Emotion Embedding With FiLM-Based Conditioning'
    if condition_key.startswith('B_proto'):
        return 'Joint Identity-Emotion Embedding With Emotion Prototype Matching'
    return 'Not Applicable'


def abcd_finetuning_strategy_full_name(condition_key):
    if condition_key.startswith('C_'):
        return 'Few-Shot User-Specific Fine-Tuning With Population Data Rehearsal'
    if condition_key.startswith('D_manyshot'):
        return 'Many-Shot User-Specific Fine-Tuning With Population Data Rehearsal'
    if condition_key == 'D_scratch_upper_reference':
        return 'Retraining From Scratch With Population And Held-Out User Support Data'
    return 'Not Applicable'


def abcd_layer_strategy_full_name(condition_key):
    if condition_key == 'D_scratch_upper_reference':
        return 'All Layers Trained From Randomly Initialized Classification Head'
    mode_map = dict(ABCD_LAYER_MODES)
    if condition_key.startswith('C_'):
        mode_key = condition_key.split('_', 2)[2]
        return mode_map.get(mode_key, mode_key)
    if condition_key == 'D_manyshot_upper_half_backbone_plus_head':
        return 'Upper Half Backbone Plus Head'
    if condition_key == 'D_manyshot_full_network':
        return 'Full Network'
    return 'Not Applicable'


def abcd_enrollment_count(condition_key):
    for shots in (7, 14):
        if str(shots) in condition_key and (condition_key.startswith('B_') or condition_key.startswith('C_')):
            return shots
    if condition_key.startswith('D_'):
        return 'Many-Shot'
    return 0


def abcd_enrollment_selection_full_name(condition_key):
    if '7' in condition_key and (condition_key.startswith('B_') or condition_key.startswith('C_')):
        return 'One Labeled Enrollment Image Per Emotion From Held-Out User Support Pool'
    if '14' in condition_key and (condition_key.startswith('B_') or condition_key.startswith('C_')):
        return 'One Central And Side Camera Enrollment Pair Per Emotion From Held-Out User Support Pool'
    if condition_key.startswith('D_'):
        return 'All Non-Test Held-Out User Images From The Common 80 Percent Support Pool'
    return 'No Held-Out User Enrollment Images'


def trainable_parameter_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def current_timestamp():
    return datetime.now().isoformat(timespec='seconds')


def read_csv_rows(path):
    if not path.exists():
        return []
    with open(path, newline='') as csvfile:
        return list(csv.DictReader(csvfile))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def read_json(path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def split_common_test_records(personal_records, test_ratio=0.2, seed=42):
    """Select an approximately 20% user-level common test holdout by capture group."""
    rng = random.Random(seed)
    groups_by_label = defaultdict(list)
    grouped = defaultdict(list)
    for record in personal_records:
        grouped[record['capture_key']].append(record)
    for group in grouped.values():
        label_counts = Counter(record['label'] for record in group)
        label = label_counts.most_common(1)[0][0]
        groups_by_label[label].append(group)

    test_paths = set()
    for label in range(len(EMOTIONS)):
        groups = list(groups_by_label.get(label, []))
        rng.shuffle(groups)
        if len(groups) <= 1:
            continue
        n_test_groups = max(1, int(round(len(groups) * test_ratio)))
        for group in groups[:n_test_groups]:
            for record in group:
                test_paths.add(record['path'])

    if not test_paths and personal_records:
        fallback = list(grouped.values())
        rng.shuffle(fallback)
        for record in fallback[0]:
            test_paths.add(record['path'])

    test_records = [record for record in personal_records if record['path'] in test_paths]
    support_records = [record for record in personal_records if record['path'] not in test_paths]
    return support_records, test_records


def select_abcd_enrollment_records(support_records, shots, seed=42, replacement_policy='keep_14',
                                   allow_partial=False):
    """Select 7 or 14 emotion-aware enrollment images and retain target labels for profiles."""
    if shots not in {7, 14}:
        raise ValueError(f"ABCD enrollment supports 7 or 14 shots, got {shots}")
    rng = random.Random(seed)
    selected_items = []
    manifest = []
    missing_labels = []
    selected_paths = set()

    def add_item(record, target_label, selection_note, replacement_for=''):
        selected_paths.add(record['path'])
        selected_items.append({
            'record': record,
            'target_label': target_label,
            'target_emotion': ID_TO_EMOTION[target_label],
            'selection_note': selection_note,
            'replacement_for_emotion': replacement_for,
        })
        manifest.append({
            **manifest_row(record, f'abcd_balanced{shots}'),
            'target_label': target_label,
            'target_emotion': ID_TO_EMOTION[target_label],
            'selection_note': selection_note,
            'replacement_for_emotion': replacement_for,
        })

    for label in range(len(EMOTIONS)):
        if shots == 14:
            pairs = paired_captures(support_records, label=label, excluded_paths=selected_paths)
            rng.shuffle(pairs)
            if not pairs:
                missing_labels.append(label)
                continue
            _, pair_records = pairs[0]
            for record in sorted(pair_records, key=lambda r: r['camera_index']):
                add_item(record, label, 'emotion_camera_pair')
        else:
            candidates = [
                record for record in support_records
                if record['label'] == label and record['path'] not in selected_paths
            ]
            candidates = sorted(candidates, key=lambda r: (r['camera_index'] != '0', r['capture_key']))
            if not candidates:
                missing_labels.append(label)
                continue
            add_item(candidates[0], label, 'one_image_per_emotion')

    if missing_labels and replacement_policy == 'strict':
        missing = ','.join(ID_TO_EMOTION[label] for label in missing_labels)
        raise RuntimeError(f"Missing required emotions for balanced{shots}: {missing}")

    if missing_labels and replacement_policy == 'keep_14':
        for target_label in missing_labels:
            replacement_records = []
            fallback_priority = SIMILAR_EMOTION_FALLBACKS.get(target_label, []) + [
                label for label in range(len(EMOTIONS))
                if label != target_label and label not in SIMILAR_EMOTION_FALLBACKS.get(target_label, [])
            ]
            for fallback_label in fallback_priority:
                if shots == 14:
                    pairs = paired_captures(support_records, label=fallback_label, excluded_paths=selected_paths)
                    rng.shuffle(pairs)
                    if pairs:
                        replacement_records = sorted(pairs[0][1], key=lambda r: r['camera_index'])
                        break
                else:
                    candidates = [
                        record for record in support_records
                        if record['label'] == fallback_label and record['path'] not in selected_paths
                    ]
                    candidates = sorted(candidates, key=lambda r: (r['camera_index'] != '0', r['capture_key']))
                    if candidates:
                        replacement_records = [candidates[0]]
                        break
            for record in replacement_records:
                if len(selected_items) >= shots:
                    break
                add_item(record, target_label, 'similar_emotion_replacement', ID_TO_EMOTION[target_label])

    if len(selected_items) < shots and not allow_partial:
        raise RuntimeError(f"Only found {len(selected_items)} records for balanced{shots}")
    return selected_items[:shots], manifest[:shots], missing_labels


def create_abcd_splits(user_records, leave_out_user, seed=42, test_ratio=0.2,
                       balanced_replacement_policy='keep_14'):
    """Create fresh ABCD splits with a common 20% held-out-user test set."""
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

    support_records, common_test_records = split_common_test_records(personal_records, test_ratio, seed)
    val_records_balanced = balance_records(val_records, seed)
    enrollment_items = {}
    enrollment_manifests = {}
    missing_by_shots = {}
    for shots in ABCD_C_SHOTS:
        items, manifest, missing = select_abcd_enrollment_records(
            support_records, shots=shots, seed=seed, replacement_policy=balanced_replacement_policy
        )
        enrollment_items[shots] = items
        enrollment_manifests[shots] = manifest
        missing_by_shots[shots] = missing

    other_emotion_test_records = [record for record in common_test_records if record['label'] != NEUTRAL_LABEL]
    return {
        'train': records_to_data(train_records),
        'val': records_to_data(val_records_balanced),
        'test_common': records_to_data(common_test_records),
        'test_other': records_to_data(other_emotion_test_records),
        'support': records_to_data(support_records),
        'enrollment_items': enrollment_items,
        'enrollment_manifests': enrollment_manifests,
        'missing_by_shots': missing_by_shots,
        'train_records': train_records,
        'val_records': val_records_balanced,
        'support_records': support_records,
        'test_records': common_test_records,
        'train_user_ids': sorted(train_users, key=numeric_sort_key),
        'val_user_ids': sorted(val_users, key=numeric_sort_key),
        'train_users': len(train_users),
        'val_users': len(val_users),
        'personal_total': len(personal_records),
        'personal_support': len(support_records),
        'personal_test': len(common_test_records),
    }


class RecordIndexDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = list(records)
        self.transform = transform or val_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, idx


def create_feature_extractor(model_name, state_dict, device):
    model = create_model(model_name=model_name).to(device)
    model.load_state_dict(state_dict)
    if not hasattr(model, 'reset_classifier'):
        raise RuntimeError(f"{model_name} does not support reset_classifier for feature extraction")
    model.reset_classifier(0)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def flatten_model_features(features):
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim > 2:
        features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)
    return features


def extract_feature_map(feature_model, records, device, batch_size=64, num_workers=0):
    unique_records = []
    seen = set()
    for record in records:
        if record['path'] not in seen:
            unique_records.append(record)
            seen.add(record['path'])
    loader = make_data_loader(
        RecordIndexDataset(unique_records, transform=val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    feature_map = {}
    feature_dim = None
    feature_model.eval()
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            features = flatten_model_features(feature_model(images))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            if feature_dim is None:
                feature_dim = int(features.shape[1])
            for idx, feature in zip(indices.numpy().tolist(), features.cpu().numpy()):
                feature_map[unique_records[idx]['path']] = feature.astype(np.float32)
    return feature_map, feature_dim or 0


def build_profile_vector(enrollment_items, feature_map, feature_dim):
    profile = np.zeros((len(EMOTIONS), feature_dim), dtype=np.float32)
    counts = np.zeros(len(EMOTIONS), dtype=np.float32)
    for item in enrollment_items:
        feature = feature_map[item['record']['path']]
        label = int(item['target_label'])
        profile[label] += feature
        counts[label] += 1
    for label in range(len(EMOTIONS)):
        if counts[label] > 0:
            profile[label] /= counts[label]
            norm = np.linalg.norm(profile[label])
            if norm > 0:
                profile[label] /= norm
    return profile.reshape(-1)


def build_user_profiles(records, shots, feature_map, feature_dim, seed=42,
                        replacement_policy='keep_14', allow_partial=False):
    profiles = {}
    selected_paths = set()
    manifests = []
    grouped = records_by_user(records)
    for user_id, user_records in grouped.items():
        items, manifest, _ = select_abcd_enrollment_records(
            user_records, shots=shots, seed=seed + int(user_id),
            replacement_policy=replacement_policy, allow_partial=allow_partial
        )
        profiles[user_id] = build_profile_vector(items, feature_map, feature_dim)
        selected_paths.update(item['record']['path'] for item in items)
        manifests.extend(manifest)
    return profiles, selected_paths, manifests


class FeatureEmotionDataset(Dataset):
    def __init__(self, records, feature_map):
        self.records = list(records)
        self.feature_map = feature_map

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return (
            torch.tensor(self.feature_map[record['path']], dtype=torch.float32),
            int(record['label']),
        )


class FeatureIdentityEmotionDataset(Dataset):
    def __init__(self, records, feature_map, user_to_identity_label):
        self.records = [
            record for record in records
            if record['user_id'] in user_to_identity_label and record['path'] in feature_map
        ]
        self.feature_map = feature_map
        self.user_to_identity_label = user_to_identity_label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return (
            torch.tensor(self.feature_map[record['path']], dtype=torch.float32),
            int(record['label']),
            int(self.user_to_identity_label[record['user_id']]),
        )


class JointIdentityEmotionEmbeddingHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=512, num_identity_classes=1, num_emotion_classes=7):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        self.emotion_head = nn.Linear(embedding_dim, num_emotion_classes)
        self.identity_head = nn.Linear(embedding_dim, num_identity_classes)

    def embed(self, features):
        embedding = self.projector(features)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def forward(self, features):
        embedding = self.embed(features)
        return embedding, self.emotion_head(embedding), self.identity_head(embedding)


def train_joint_identity_emotion_embedder(train_records, val_records, base_feature_map, input_dim, device,
                                          embedding_dim=512, identity_loss_weight=1.0,
                                          epochs=50, patience=10, batch_size=64, lr=1e-4):
    """Train a lightweight joint identity-emotion projection on frozen FER features."""
    start_time = time.time()
    train_user_ids = sorted({record['user_id'] for record in train_records}, key=numeric_sort_key)
    user_to_identity_label = {user_id: idx for idx, user_id in enumerate(train_user_ids)}
    if not train_user_ids:
        raise RuntimeError("Cannot train joint identity-emotion embedding without training users")

    train_dataset = FeatureIdentityEmotionDataset(train_records, base_feature_map, user_to_identity_label)
    val_dataset = FeatureEmotionDataset(val_records, base_feature_map)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = JointIdentityEmotionEmbeddingHead(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_identity_classes=len(train_user_ids),
        num_emotion_classes=len(EMOTIONS),
    ).to(device)
    emotion_labels = [record['label'] for record in train_dataset.records]
    emotion_criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(emotion_labels, device))
    identity_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_labels, train_preds = [], []
        identity_labels, identity_preds = [], []
        for features, emotions, identities in train_loader:
            features = features.to(device)
            emotions = emotions.to(device)
            identities = identities.to(device)
            optimizer.zero_grad()
            _, emotion_logits, identity_logits = model(features)
            loss = (
                emotion_criterion(emotion_logits, emotions)
                + identity_loss_weight * identity_criterion(identity_logits, identities)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            train_preds.extend(emotion_logits.argmax(dim=1).detach().cpu().numpy())
            train_labels.extend(emotions.detach().cpu().numpy())
            identity_preds.extend(identity_logits.argmax(dim=1).detach().cpu().numpy())
            identity_labels.extend(identities.detach().cpu().numpy())

        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for features, emotions in val_loader:
                features = features.to(device)
                _, emotion_logits, _ = model(features)
                val_preds.extend(emotion_logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(emotions.numpy())
        val_acc = accuracy_score(val_labels, val_preds) if val_labels else 0
        train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0
        identity_acc = accuracy_score(identity_labels, identity_preds) if identity_labels else 0
        avg_loss = total_loss / max(1, len(train_dataset))
        print(
            f"  Joint Embedding Epoch {epoch+1}: "
            f"Loss={avg_loss:.4f}, Emotion Train Acc={train_acc:.4f}, "
            f"Identity Train Acc={identity_acc:.4f}, Emotion Val Acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Joint Embedding Early Stopping] at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, time.time() - start_time, {
        'input_dim': input_dim,
        'embedding_dim': embedding_dim,
        'num_identity_classes': len(train_user_ids),
        'identity_loss_weight': identity_loss_weight,
        'best_val_emotion_acc': best_val_acc,
    }


def extract_joint_embedding_map(joint_model, base_feature_map, records, device, batch_size=256):
    unique_records = []
    seen = set()
    for record in records:
        if record['path'] not in seen:
            unique_records.append(record)
            seen.add(record['path'])
    loader = DataLoader(FeatureEmotionDataset(unique_records, base_feature_map), batch_size=batch_size, shuffle=False)
    joint_map = {}
    embedding_dim = None
    joint_model.eval()
    with torch.no_grad():
        offset = 0
        for features, _ in loader:
            features = features.to(device)
            embeddings = joint_model.embed(features)
            if embedding_dim is None:
                embedding_dim = int(embeddings.shape[1])
            for record, embedding in zip(unique_records[offset:offset + len(features)], embeddings.cpu().numpy()):
                joint_map[record['path']] = embedding.astype(np.float32)
            offset += len(features)
    return joint_map, embedding_dim or 0


class FeatureProfileDataset(Dataset):
    def __init__(self, records, feature_map, profiles):
        self.records = list(records)
        self.feature_map = feature_map
        self.profiles = profiles

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return (
            torch.tensor(self.feature_map[record['path']], dtype=torch.float32),
            torch.tensor(self.profiles[record['user_id']], dtype=torch.float32),
            int(record['label']),
        )


class FiLMFeatureClassifier(nn.Module):
    def __init__(self, feature_dim, profile_dim, num_classes=7):
        super().__init__()
        hidden_dim = max(128, min(512, profile_dim // 4))
        self.modulator = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features, profiles):
        gamma_beta = self.modulator(profiles)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        conditioned = features * (1.0 + gamma) + beta
        return self.classifier(conditioned)


def train_film_conditioner(train_records, val_records, feature_map, train_profiles, val_profiles,
                           feature_dim, profile_dim, device, epochs=50, patience=10,
                           batch_size=64, lr=1e-4):
    start_time = time.time()
    train_dataset = FeatureProfileDataset(train_records, feature_map, train_profiles)
    val_dataset = FeatureProfileDataset(val_records, feature_map, val_profiles)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = FiLMFeatureClassifier(feature_dim, profile_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for features, profiles, labels in train_loader:
            features = features.to(device)
            profiles = profiles.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features, profiles), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for features, profiles, labels in val_loader:
                features = features.to(device)
                profiles = profiles.to(device)
                logits = model(features, profiles)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())
        val_acc = accuracy_score(val_labels, val_preds) if val_labels else 0
        print(f"  FiLM Epoch {epoch+1}: Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [FiLM Early Stopping] at epoch {epoch+1}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, time.time() - start_time


def prediction_metrics(labels, preds, elapsed_sec=0):
    labels = list(labels)
    preds = list(preds)
    if not labels:
        return {
            'accuracy': 0,
            'f1': 0,
            'n_samples': 0,
            'other_accuracy': 0,
            'other_f1': 0,
            'other_n_samples': 0,
            'confusion_matrix': [],
            'per_emotion': [],
            'mean_inference_time_ms': 0,
        }
    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))
    precision, recall, f1_values, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(EMOTIONS))), zero_division=0
    )
    other = [(label, pred) for label, pred in zip(labels, preds) if label != NEUTRAL_LABEL]
    if other:
        other_labels, other_preds = zip(*other)
        other_accuracy = accuracy_score(other_labels, other_preds)
        other_f1 = f1_score(
            other_labels, other_preds, average='macro',
            labels=[label for label in range(len(EMOTIONS)) if label != NEUTRAL_LABEL],
            zero_division=0,
        )
    else:
        other_accuracy, other_f1 = 0, 0
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro', labels=list(range(len(EMOTIONS))), zero_division=0),
        'n_samples': len(labels),
        'other_accuracy': other_accuracy,
        'other_f1': other_f1,
        'other_n_samples': len(other),
        'confusion_matrix': cm.tolist(),
        'per_emotion': [
            {
                'emotion': ID_TO_EMOTION[label],
                'precision': float(precision[label]),
                'recall': float(recall[label]),
                'f1': float(f1_values[label]),
                'support': int(support[label]),
            }
            for label in range(len(EMOTIONS))
        ],
        'mean_inference_time_ms': (elapsed_sec / len(labels) * 1000.0) if labels else 0,
    }


def evaluate_model_detailed(model, test_data, device, batch_size=32):
    if len(test_data['images']) == 0:
        return prediction_metrics([], [])
    dataset = EmojiHeroDataset(test_data['images'], test_data['labels'], transform=val_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    labels_all, preds_all = [], []
    start = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
    return prediction_metrics(labels_all, preds_all, time.time() - start)


def evaluate_film_model(film_model, feature_map, profile_vector, test_records, device):
    labels, preds = [], []
    profile = torch.tensor(profile_vector, dtype=torch.float32, device=device).unsqueeze(0)
    start = time.time()
    film_model.eval()
    with torch.no_grad():
        for record in test_records:
            feature = torch.tensor(feature_map[record['path']], dtype=torch.float32, device=device).unsqueeze(0)
            logits = film_model(feature, profile)
            preds.append(int(logits.argmax(dim=1).item()))
            labels.append(int(record['label']))
    return prediction_metrics(labels, preds, time.time() - start)


def evaluate_prototype_model(feature_map, profile_vector, feature_dim, test_records):
    prototypes = profile_vector.reshape(len(EMOTIONS), feature_dim)
    labels, preds = [], []
    start = time.time()
    for record in test_records:
        feature = feature_map[record['path']]
        scores = prototypes @ feature
        preds.append(int(np.argmax(scores)))
        labels.append(int(record['label']))
    return prediction_metrics(labels, preds, time.time() - start)


def abcd_result_row(condition_key, heldout_user, result, splits, base_result=None,
                    model_name='',
                    enrollment_count=0, enrollment_used=0, missing_labels=None,
                    replacement_policy='keep_14', base_time=0, profile_time=0,
                    joint_embedding_time=0, film_time=0, finetune_time=0, trainable_params=0,
                    total_params=0, checkpoint_source=''):
    missing_labels = missing_labels or []
    if base_result is None or condition_key == 'A_base':
        win_tie_loss = 'Base Reference'
        delta_acc = 0.0
        delta_f1 = 0.0
    else:
        delta_acc = (result['accuracy'] - base_result['accuracy']) * 100.0
        delta_f1 = (result['f1'] - base_result['f1']) * 100.0
        if math.isclose(result['accuracy'], base_result['accuracy'], rel_tol=0, abs_tol=1e-12):
            win_tie_loss = 'Tie Compared With Subject-Independent Base'
        elif result['accuracy'] > base_result['accuracy']:
            win_tie_loss = 'Win Compared With Subject-Independent Base'
        else:
            win_tie_loss = 'Loss Compared With Subject-Independent Base'
    total_time = base_time + joint_embedding_time + profile_time + film_time + finetune_time
    return {
        'Model Name': model_name,
        'Experiment Family Full Name': abcd_family_full_name(condition_key),
        'Condition Full Name': abcd_condition_full_name(condition_key),
        'User Profile Method Full Name': abcd_profile_method_full_name(condition_key),
        'Enrollment Image Count': enrollment_count,
        'Enrollment Image Selection Full Name': abcd_enrollment_selection_full_name(condition_key),
        'Fine-Tuning Strategy Full Name': abcd_finetuning_strategy_full_name(condition_key),
        'Layer-Freezing Strategy Full Name': abcd_layer_strategy_full_name(condition_key),
        'Held-Out User ID': str(heldout_user),
        'Common Test Image Count': result['n_samples'],
        'Enrollment Image Count Actually Used': enrollment_used,
        'Missing Enrollment Emotion Names': ', '.join(ID_TO_EMOTION[label] for label in missing_labels),
        'Replacement Policy Full Name': (
            'Keep Enrollment Count By Same-User Similar-Emotion Replacement'
            if replacement_policy == 'keep_14' else 'Strict Complete Emotion Enrollment Only'
        ),
        'Accuracy': result['accuracy'],
        'Macro F1 Score': result['f1'],
        'Non-Neutral Accuracy': result['other_accuracy'],
        'Non-Neutral Macro F1 Score': result['other_f1'],
        'Win Tie Loss Compared With Subject-Independent Base': win_tie_loss,
        'Accuracy Difference From Base Percentage Points': delta_acc,
        'Macro F1 Difference From Base Percentage Points': delta_f1,
        'Base Model Training Time Seconds': base_time,
        'Joint Identity Emotion Embedding Training Time Seconds': joint_embedding_time,
        'User Profile Construction Time Seconds': profile_time,
        'FiLM Conditioning Training Time Seconds': film_time,
        'Fine-Tuning Time Seconds': finetune_time,
        'Total Training Or Adaptation Time Seconds': total_time,
        'Mean Inference Time Per Image Milliseconds': result['mean_inference_time_ms'],
        'Trainable Parameter Count': trainable_params,
        'Total Parameter Count': total_params,
        'Run Status Full Name': 'Completed',
        'Completed Timestamp': current_timestamp(),
        'Result Checkpoint Or Resume Source': checkpoint_source,
    }


def abcd_per_emotion_rows(condition_key, heldout_user, result, model_name=''):
    rows = []
    for item in result['per_emotion']:
        rows.append({
            'Model Name': model_name,
            'Held-Out User ID': str(heldout_user),
            'Experiment Family Full Name': abcd_family_full_name(condition_key),
            'Condition Full Name': abcd_condition_full_name(condition_key),
            'Emotion Name': item['emotion'],
            'Precision': item['precision'],
            'Recall': item['recall'],
            'F1 Score': item['f1'],
            'Support Count': item['support'],
        })
    return rows


def abcd_manifest_rows_for_split(heldout_user, splits, model_condition='All Conditions'):
    rows = []
    for record in splits['test_records']:
        rows.append({
            'Held-Out User ID': str(heldout_user),
            'Split Role Full Name': 'Common Test Holdout Image',
            'Experiment Family Full Name': 'All ABCD Experiment Families',
            'Condition Full Name': model_condition,
            'Enrollment Image Count': 0,
            'Target Emotion Name': record['emotion'],
            'Actual Emotion Name': record['emotion'],
            'Camera Index': record['camera_index'],
            'Capture Key': record['capture_key'],
            'Original Dataset Split Name': record.get('original_set', ''),
            'Selection Note Full Name': 'Selected For Common 20 Percent Held-Out User Test Set',
            'Replacement For Missing Emotion Name': '',
            'Filename': record['filename'],
            'Path': record['path'],
        })
    for shots, manifest in splits['enrollment_manifests'].items():
        for row in manifest:
            rows.append({
                'Held-Out User ID': str(heldout_user),
                'Split Role Full Name': f'{shots} Enrollment Image User Profile Or Fine-Tuning Support',
                'Experiment Family Full Name': 'B And C Enrollment-Based Conditions',
                'Condition Full Name': f'All {shots} Enrollment Image Conditions',
                'Enrollment Image Count': shots,
                'Target Emotion Name': row.get('target_emotion', row['emotion']),
                'Actual Emotion Name': row['emotion'],
                'Camera Index': row['camera_index'],
                'Capture Key': row['capture_key'],
                'Original Dataset Split Name': row.get('original_set', ''),
                'Selection Note Full Name': row.get('selection_note', ''),
                'Replacement For Missing Emotion Name': row.get('replacement_for_emotion', ''),
                'Filename': row['filename'],
                'Path': row['path'],
            })
    return rows


def summarize_abcd_rows(fold_rows):
    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[(row.get('Model Name', ''), row['Condition Full Name'])].append(row)
    summary_rows = []
    for (model_name, condition_full_name), rows in grouped.items():
        accuracies = [float(row['Accuracy']) for row in rows]
        f1_values = [float(row['Macro F1 Score']) for row in rows]
        other_acc = [float(row['Non-Neutral Accuracy']) for row in rows]
        other_f1 = [float(row['Non-Neutral Macro F1 Score']) for row in rows]
        delta_acc = [float(row['Accuracy Difference From Base Percentage Points']) for row in rows]
        delta_f1 = [float(row['Macro F1 Difference From Base Percentage Points']) for row in rows]
        total_times = [float(row['Total Training Or Adaptation Time Seconds']) for row in rows]
        joint_times = [float(row.get('Joint Identity Emotion Embedding Training Time Seconds', 0) or 0) for row in rows]
        profile_times = [float(row['User Profile Construction Time Seconds']) for row in rows]
        finetune_times = [float(row['Fine-Tuning Time Seconds']) for row in rows]
        inference_times = [float(row['Mean Inference Time Per Image Milliseconds']) for row in rows]
        wins = sum(row['Win Tie Loss Compared With Subject-Independent Base'].startswith('Win') for row in rows)
        ties = sum(row['Win Tie Loss Compared With Subject-Independent Base'].startswith('Tie') for row in rows)
        losses = sum(row['Win Tie Loss Compared With Subject-Independent Base'].startswith('Loss') for row in rows)
        strict_count = sum(not row['Missing Enrollment Emotion Names'] for row in rows)
        summary_rows.append({
            'Model Name': model_name,
            'Experiment Family Full Name': rows[0]['Experiment Family Full Name'],
            'Condition Full Name': condition_full_name,
            'Mean Accuracy': float(np.mean(accuracies)),
            'Accuracy Standard Deviation': float(np.std(accuracies)),
            'Mean Macro F1 Score': float(np.mean(f1_values)),
            'Macro F1 Score Standard Deviation': float(np.std(f1_values)),
            'Mean Non-Neutral Accuracy': float(np.mean(other_acc)),
            'Mean Non-Neutral Macro F1 Score': float(np.mean(other_f1)),
            'Mean Accuracy Difference From Base Percentage Points': float(np.mean(delta_acc)),
            'Mean Macro F1 Difference From Base Percentage Points': float(np.mean(delta_f1)),
            'Users Improved Compared With Base': wins,
            'Users Tied Compared With Base': ties,
            'Users Worse Compared With Base': losses,
            'Mean Total Training Or Adaptation Time Seconds': float(np.mean(total_times)),
            'Mean Joint Identity Emotion Embedding Training Time Seconds': float(np.mean(joint_times)),
            'Mean User Profile Construction Time Seconds': float(np.mean(profile_times)),
            'Mean Fine-Tuning Time Seconds': float(np.mean(finetune_times)),
            'Mean Inference Time Per Image Milliseconds': float(np.mean(inference_times)),
            'Strict Complete-Emotion User Count': strict_count,
            'All User Count': len(rows),
        })
    return sorted(summary_rows, key=lambda row: (row.get('Model Name', ''), row['Condition Full Name']))


def flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status):
    write_csv_rows(paths['fold'], ABCD_FOLD_FIELDS, fold_rows)
    write_csv_rows(paths['per_emotion'], ABCD_PER_EMOTION_FIELDS, per_emotion_rows)
    write_csv_rows(paths['manifest'], ABCD_MANIFEST_FIELDS, manifest_rows)
    write_csv_rows(paths['summary'], ABCD_SUMMARY_FIELDS, summarize_abcd_rows(fold_rows))
    write_json(paths['confusion'], confusion_payload)
    write_json(paths['status'], status)


def abcd_is_complete(status, model_name, heldout_user, condition_key):
    return condition_key in status.get('completed_conditions', {}).get(model_name, {}).get(str(heldout_user), [])


def abcd_mark_complete(status, model_name, heldout_user, condition_key):
    status.setdefault('completed_conditions', {}).setdefault(model_name, {}).setdefault(str(heldout_user), [])
    completed = status['completed_conditions'][model_name][str(heldout_user)]
    if condition_key not in completed:
        completed.append(condition_key)


def run_abcd(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = resolve_si_root(args.data_dir)
    data_layout = infer_data_layout(data_dir) if args.data_layout == 'auto' else args.data_layout
    records = collect_records(data_dir, data_layout)
    user_records = records_by_user(records)
    all_users = sorted(user_records.keys(), key=numeric_sort_key)
    if args.user:
        fold_users = [str(args.user)]
    elif args.folds == 'all':
        fold_users = all_users
    else:
        fold_users = parse_csv_option(args.folds)
    if args.max_folds:
        fold_users = fold_users[:args.max_folds]
    model_names = resolve_model_names(args)
    abcd_c_modes = parse_csv_option(args.abcd_c_modes)
    if not abcd_c_modes:
        selected_c_layer_modes = ABCD_LAYER_MODES
    else:
        layer_mode_names = dict(ABCD_LAYER_MODES)
        unsupported = [mode for mode in abcd_c_modes if mode not in layer_mode_names]
        if unsupported:
            raise ValueError(
                f"Unsupported ABCD C fine-tuning modes: {unsupported}. "
                f"Supported: {list(layer_mode_names)}"
            )
        selected_c_layer_modes = [(mode, layer_mode_names[mode]) for mode in abcd_c_modes]
    abcd_d_condition_keys = parse_csv_option(args.abcd_d_conditions)
    if not abcd_d_condition_keys:
        selected_d_conditions = ABCD_D_CONDITION_MODES
    else:
        d_condition_modes = dict(ABCD_D_CONDITION_MODES)
        unsupported = [key for key in abcd_d_condition_keys if key not in d_condition_modes]
        if unsupported:
            raise ValueError(
                f"Unsupported ABCD D upper/reference conditions: {unsupported}. "
                f"Supported: {list(d_condition_modes)}"
            )
        selected_d_conditions = [(key, d_condition_modes[key]) for key in abcd_d_condition_keys]

    if args.abcd_resume_dir:
        results_dir = Path(args.abcd_resume_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or 'abcd'
        results_dir = Path(args.results_dir) / f"{run_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    paths = {
        'fold': results_dir / 'abcd_fold_level_results.csv',
        'summary': results_dir / 'abcd_condition_summary.csv',
        'manifest': results_dir / 'abcd_support_and_test_manifest.csv',
        'per_emotion': results_dir / 'abcd_per_emotion_results.csv',
        'confusion': results_dir / 'abcd_confusion_matrices.json',
        'status': results_dir / 'abcd_run_status.json',
        'config': results_dir / 'abcd_run_config.json',
    }
    config = {
        'mode': 'abcd',
        'data_dir': str(data_dir),
        'data_layout': data_layout,
        'models': model_names,
        'requested_models': parse_csv_option(args.models) or ['efficientnet_b0'],
        'include_verification_backbones': bool(args.include_verification_backbones),
        'verification_backbone_models_added_by_option': (
            VERIFICATION_BACKBONE_MODELS if args.include_verification_backbones else []
        ),
        'fold_users': fold_users,
        'common_test_holdout_ratio': args.abcd_test_ratio,
        'replacement_policy': args.balanced_replacement_policy,
        'c_finetuning_layer_modes': [mode_name for _, mode_name in selected_c_layer_modes],
        'd_upper_reference_conditions': [
            ABCD_CONDITION_FULL_NAMES[condition_key]
            for condition_key, _ in selected_d_conditions
        ],
        'planned_condition_count_per_user_per_model': (
            1 + 4 + (len(ABCD_C_SHOTS) * len(selected_c_layer_modes)) + len(selected_d_conditions)
        ),
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'b_joint_embedding_dim': args.abcd_joint_embedding_dim,
        'b_joint_embedding_learning_rate': args.abcd_joint_lr,
        'b_joint_identity_loss_weight': args.abcd_joint_identity_loss_weight,
        'seed': seed,
    }
    write_json(paths['config'], config)

    fold_rows = read_csv_rows(paths['fold'])
    per_emotion_rows = read_csv_rows(paths['per_emotion'])
    manifest_rows = read_csv_rows(paths['manifest'])
    confusion_payload = read_json(paths['confusion'], {})
    status = read_json(paths['status'], {'completed_conditions': {}, 'run_dir': str(results_dir)})
    status['run_dir'] = str(results_dir)

    existing_manifest_users = {
        (row['Held-Out User ID'], row['Condition Full Name']) for row in manifest_rows
    }
    for heldout_user in fold_users:
        splits = create_abcd_splits(
            user_records, heldout_user,
            seed=seed + int(heldout_user) if str(heldout_user).isdigit() else seed,
            test_ratio=args.abcd_test_ratio,
            balanced_replacement_policy=args.balanced_replacement_policy,
        )
        if (str(heldout_user), 'All Conditions') not in existing_manifest_users:
            manifest_rows.extend(abcd_manifest_rows_for_split(heldout_user, splits))
            existing_manifest_users.add((str(heldout_user), 'All Conditions'))

    if args.dry_run:
        flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)
        print(f"[Dry Run] ABCD manifests and config written to: {results_dir}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mlruns_dir = PROJECT_DIR / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
    mlflow.set_experiment("loo_cv_abcd")

    for model_name in model_names:
        for heldout_user in fold_users:
            print(f"\n{'='*80}")
            print(f"ABCD: model={model_name}, held-out user={heldout_user}")
            print(f"{'='*80}")
            splits = create_abcd_splits(
                user_records, heldout_user,
                seed=seed + int(heldout_user) if str(heldout_user).isdigit() else seed,
                test_ratio=args.abcd_test_ratio,
                balanced_replacement_policy=args.balanced_replacement_policy,
            )
            base_ckpt = checkpoint_dir / f"{model_name}_user{heldout_user}_base.pt"
            base_meta = checkpoint_dir / f"{model_name}_user{heldout_user}_base.json"

            with mlflow.start_run(run_name=f"abcd_{model_name}_User_{heldout_user}"):
                if base_ckpt.exists() and base_meta.exists():
                    base_model = create_model(model_name=model_name).to(device)
                    base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
                    base_time = read_json(base_meta, {}).get('stage1_time_sec', 0)
                    print(f"Loaded base checkpoint: {base_ckpt}")
                else:
                    base_model, base_time = stage1_base_training(
                        splits['train'], splits['val'], device,
                        epochs=args.epochs, patience=args.patience,
                        batch_size=args.batch_size, model_name=model_name,
                        num_workers=args.num_workers,
                        persistent_workers=not args.no_persistent_workers,
                    )
                    torch.save(base_model.state_dict(), base_ckpt)
                    write_json(base_meta, {'stage1_time_sec': base_time, 'completed_at': current_timestamp()})

                base_state = copy.deepcopy(base_model.state_dict())
                base_result = None
                if not abcd_is_complete(status, model_name, heldout_user, 'A_base'):
                    result = evaluate_model_detailed(base_model, splits['test_common'], device, args.batch_size)
                    trainable, total = trainable_parameter_counts(base_model)
                    row = abcd_result_row(
                        'A_base', heldout_user, result, splits, base_result=None,
                        model_name=model_name,
                        base_time=base_time, trainable_params=trainable, total_params=total,
                        checkpoint_source=str(base_ckpt),
                    )
                    fold_rows.append(row)
                    per_emotion_rows.extend(abcd_per_emotion_rows('A_base', heldout_user, result, model_name))
                    confusion_payload.setdefault(model_name, {}).setdefault(str(heldout_user), {})[
                        abcd_condition_full_name('A_base')
                    ] = result['confusion_matrix']
                    abcd_mark_complete(status, model_name, heldout_user, 'A_base')
                    flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)
                    base_result = result
                else:
                    base_matches = [
                        row for row in fold_rows
                        if row['Held-Out User ID'] == str(heldout_user)
                        and row['Condition Full Name'] == abcd_condition_full_name('A_base')
                    ]
                    if base_matches:
                        base_result = {
                            'accuracy': float(base_matches[-1]['Accuracy']),
                            'f1': float(base_matches[-1]['Macro F1 Score']),
                        }
                    else:
                        base_result = evaluate_model_detailed(base_model, splits['test_common'], device, args.batch_size)

                needs_b = any(
                    not abcd_is_complete(status, model_name, heldout_user, key)
                    for key in ['B_film7', 'B_film14', 'B_proto7', 'B_proto14']
                )
                if needs_b:
                    base_feature_model = create_feature_extractor(model_name, base_state, device)
                    feature_records = (
                        splits['train_records'] + splits['val_records'] +
                        splits['support_records'] + splits['test_records']
                    )
                    base_feature_map, base_feature_dim = extract_feature_map(
                        base_feature_model, feature_records, device,
                        batch_size=args.batch_size, num_workers=0,
                    )
                    joint_ckpt = checkpoint_dir / f"{model_name}_user{heldout_user}_joint_identity_emotion_embedding.pt"
                    joint_meta = checkpoint_dir / f"{model_name}_user{heldout_user}_joint_identity_emotion_embedding.json"
                    if joint_ckpt.exists() and joint_meta.exists():
                        joint_info = read_json(joint_meta, {})
                        joint_model = JointIdentityEmotionEmbeddingHead(
                            input_dim=joint_info.get('input_dim', base_feature_dim),
                            embedding_dim=joint_info.get('embedding_dim', args.abcd_joint_embedding_dim),
                            num_identity_classes=joint_info.get('num_identity_classes', len(splits['train_user_ids'])),
                            num_emotion_classes=len(EMOTIONS),
                        ).to(device)
                        joint_model.load_state_dict(torch.load(joint_ckpt, map_location=device))
                        joint_time = joint_info.get('joint_embedding_time_sec', 0)
                        print(f"Loaded joint identity-emotion embedding: {joint_ckpt}")
                    else:
                        joint_model, joint_time, joint_info = train_joint_identity_emotion_embedder(
                            splits['train_records'], splits['val_records'],
                            base_feature_map, base_feature_dim, device,
                            embedding_dim=args.abcd_joint_embedding_dim,
                            identity_loss_weight=args.abcd_joint_identity_loss_weight,
                            epochs=args.epochs, patience=args.patience,
                            batch_size=args.batch_size, lr=args.abcd_joint_lr,
                        )
                        joint_info.update({
                            'joint_embedding_time_sec': joint_time,
                            'completed_at': current_timestamp(),
                        })
                        torch.save(joint_model.state_dict(), joint_ckpt)
                        write_json(joint_meta, joint_info)
                    feature_map, feature_dim = extract_joint_embedding_map(
                        joint_model, base_feature_map, feature_records, device,
                        batch_size=args.batch_size,
                    )
                    joint_trainable, joint_total = trainable_parameter_counts(joint_model)
                    for shots in ABCD_C_SHOTS:
                        profile_start = time.time()
                        heldout_profile = build_profile_vector(
                            splits['enrollment_items'][shots], feature_map, feature_dim
                        )
                        profile_time = time.time() - profile_start
                        missing = splits['missing_by_shots'][shots]
                        profile_dim = len(heldout_profile)

                        proto_key = f'B_proto{shots}'
                        if not abcd_is_complete(status, model_name, heldout_user, proto_key):
                            result = evaluate_prototype_model(
                                feature_map, heldout_profile, feature_dim, splits['test_records']
                            )
                            row = abcd_result_row(
                                proto_key, heldout_user, result, splits, base_result=base_result,
                                model_name=model_name,
                                enrollment_count=shots, enrollment_used=len(splits['enrollment_items'][shots]),
                                missing_labels=missing,
                                replacement_policy=args.balanced_replacement_policy,
                                base_time=base_time, profile_time=profile_time,
                                joint_embedding_time=joint_time,
                                trainable_params=joint_trainable, total_params=joint_total,
                                checkpoint_source=str(joint_ckpt),
                            )
                            fold_rows.append(row)
                            per_emotion_rows.extend(abcd_per_emotion_rows(proto_key, heldout_user, result, model_name))
                            confusion_payload.setdefault(model_name, {}).setdefault(str(heldout_user), {})[
                                abcd_condition_full_name(proto_key)
                            ] = result['confusion_matrix']
                            abcd_mark_complete(status, model_name, heldout_user, proto_key)
                            flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)

                        film_key = f'B_film{shots}'
                        if not abcd_is_complete(status, model_name, heldout_user, film_key):
                            train_profiles, train_profile_paths, _ = build_user_profiles(
                                splits['train_records'], shots, feature_map, feature_dim,
                                seed=seed, replacement_policy=args.balanced_replacement_policy,
                                allow_partial=True,
                            )
                            val_profiles, val_profile_paths, _ = build_user_profiles(
                                splits['val_records'], shots, feature_map, feature_dim,
                                seed=seed + 17, replacement_policy=args.balanced_replacement_policy,
                                allow_partial=True,
                            )
                            film_train_records = [
                                record for record in splits['train_records']
                                if record['path'] not in train_profile_paths
                            ] or splits['train_records']
                            film_val_records = [
                                record for record in splits['val_records']
                                if record['path'] not in val_profile_paths
                            ] or splits['val_records']
                            film_model, film_time = train_film_conditioner(
                                film_train_records, film_val_records, feature_map,
                                train_profiles, val_profiles, feature_dim, profile_dim, device,
                                epochs=args.epochs, patience=args.patience,
                                batch_size=args.batch_size, lr=args.abcd_film_lr,
                            )
                            result = evaluate_film_model(
                                film_model, feature_map, heldout_profile, splits['test_records'], device
                            )
                            trainable, total = trainable_parameter_counts(film_model)
                            row = abcd_result_row(
                                film_key, heldout_user, result, splits, base_result=base_result,
                                model_name=model_name,
                                enrollment_count=shots, enrollment_used=len(splits['enrollment_items'][shots]),
                                missing_labels=missing,
                                replacement_policy=args.balanced_replacement_policy,
                                base_time=base_time, profile_time=profile_time,
                                joint_embedding_time=joint_time, film_time=film_time,
                                trainable_params=joint_trainable + trainable,
                                total_params=joint_total + total,
                                checkpoint_source=str(joint_ckpt),
                            )
                            fold_rows.append(row)
                            per_emotion_rows.extend(abcd_per_emotion_rows(film_key, heldout_user, result, model_name))
                            confusion_payload.setdefault(model_name, {}).setdefault(str(heldout_user), {})[
                                abcd_condition_full_name(film_key)
                            ] = result['confusion_matrix']
                            abcd_mark_complete(status, model_name, heldout_user, film_key)
                            flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)

                for shots in ABCD_C_SHOTS:
                    personal_records = [item['record'] for item in splits['enrollment_items'][shots]]
                    personal_data = records_to_data(personal_records)
                    for mode_key, _ in selected_c_layer_modes:
                        condition_key = abcd_c_condition_key(shots, mode_key)
                        if abcd_is_complete(status, model_name, heldout_user, condition_key):
                            continue
                        model_ft = create_model(model_name=model_name).to(device)
                        model_ft.load_state_dict(base_state)
                        personalized_model, finetune_time = stage2_personalize(
                            model_ft, splits['train'], splits['val'], personal_data, device,
                            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                            classifier_only=False, unfreeze_ratio=mode_key,
                            train_source=args.stage2_train_source,
                            num_workers=args.num_workers,
                            persistent_workers=not args.no_persistent_workers,
                        )
                        result = evaluate_model_detailed(personalized_model, splits['test_common'], device, args.batch_size)
                        trainable, total = trainable_parameter_counts(personalized_model)
                        row = abcd_result_row(
                            condition_key, heldout_user, result, splits, base_result=base_result,
                            model_name=model_name,
                            enrollment_count=shots, enrollment_used=len(personal_records),
                            missing_labels=splits['missing_by_shots'][shots],
                            replacement_policy=args.balanced_replacement_policy,
                            base_time=base_time, finetune_time=finetune_time,
                            trainable_params=trainable, total_params=total,
                            checkpoint_source=str(base_ckpt),
                        )
                        fold_rows.append(row)
                        per_emotion_rows.extend(abcd_per_emotion_rows(condition_key, heldout_user, result, model_name))
                        confusion_payload.setdefault(model_name, {}).setdefault(str(heldout_user), {})[
                            abcd_condition_full_name(condition_key)
                        ] = result['confusion_matrix']
                        abcd_mark_complete(status, model_name, heldout_user, condition_key)
                        flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)

                for condition_key, mode_key in selected_d_conditions:
                    if condition_key == 'D_scratch_upper_reference':
                        continue
                    if abcd_is_complete(status, model_name, heldout_user, condition_key):
                        continue
                    model_ft = create_model(model_name=model_name).to(device)
                    model_ft.load_state_dict(base_state)
                    personalized_model, finetune_time = stage2_personalize(
                        model_ft, splits['train'], splits['val'], splits['support'], device,
                        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                        classifier_only=False, unfreeze_ratio=mode_key,
                        train_source=args.stage2_train_source,
                        num_workers=args.num_workers,
                        persistent_workers=not args.no_persistent_workers,
                    )
                    result = evaluate_model_detailed(personalized_model, splits['test_common'], device, args.batch_size)
                    trainable, total = trainable_parameter_counts(personalized_model)
                    row = abcd_result_row(
                        condition_key, heldout_user, result, splits, base_result=base_result,
                        model_name=model_name,
                        enrollment_count='Many-Shot', enrollment_used=len(splits['support']['images']),
                        replacement_policy=args.balanced_replacement_policy,
                        base_time=base_time, finetune_time=finetune_time,
                        trainable_params=trainable, total_params=total,
                        checkpoint_source=str(base_ckpt),
                    )
                    fold_rows.append(row)
                    per_emotion_rows.extend(abcd_per_emotion_rows(condition_key, heldout_user, result, model_name))
                    confusion_payload.setdefault(model_name, {}).setdefault(str(heldout_user), {})[
                        abcd_condition_full_name(condition_key)
                    ] = result['confusion_matrix']
                    abcd_mark_complete(status, model_name, heldout_user, condition_key)
                    flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)

                scratch_key = 'D_scratch_upper_reference'
                if (
                    any(condition_key == scratch_key for condition_key, _ in selected_d_conditions)
                    and not abcd_is_complete(status, model_name, heldout_user, scratch_key)
                ):
                    combined_images = splits['train']['images'] + splits['support']['images']
                    combined_labels = splits['train']['labels'] + splits['support']['labels']
                    train_dataset = EmojiHeroDataset(combined_images, combined_labels, transform=train_transform)
                    val_dataset = EmojiHeroDataset(splits['val']['images'], splits['val']['labels'], transform=val_transform)
                    train_loader = make_data_loader(
                        train_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, persistent_workers=not args.no_persistent_workers
                    )
                    val_loader = make_data_loader(
                        val_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, persistent_workers=not args.no_persistent_workers
                    )
                    scratch_model = create_model(model_name=model_name).to(device)
                    for param in scratch_model.parameters():
                        param.requires_grad = True
                    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(combined_labels, device))
                    optimizer = optim.Adam(scratch_model.parameters(), lr=1e-4)
                    start = time.time()
                    scratch_model, _ = train_with_early_stopping(
                        scratch_model, train_loader, val_loader, optimizer, criterion,
                        device, args.epochs, args.patience, "ABCD Scratch Upper Reference"
                    )
                    scratch_time = time.time() - start
                    result = evaluate_model_detailed(scratch_model, splits['test_common'], device, args.batch_size)
                    trainable, total = trainable_parameter_counts(scratch_model)
                    row = abcd_result_row(
                        scratch_key, heldout_user, result, splits, base_result=base_result,
                        model_name=model_name,
                        enrollment_count='Many-Shot', enrollment_used=len(splits['support']['images']),
                        replacement_policy=args.balanced_replacement_policy,
                        base_time=0, finetune_time=scratch_time,
                        trainable_params=trainable, total_params=total,
                        checkpoint_source='Scratch retraining condition; no base checkpoint used',
                    )
                    fold_rows.append(row)
                    per_emotion_rows.extend(abcd_per_emotion_rows(scratch_key, heldout_user, result, model_name))
                    confusion_payload.setdefault(model_name, {}).setdefault(str(heldout_user), {})[
                        abcd_condition_full_name(scratch_key)
                    ] = result['confusion_matrix']
                    abcd_mark_complete(status, model_name, heldout_user, scratch_key)
                    flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)

    flush_abcd_outputs(paths, fold_rows, per_emotion_rows, manifest_rows, confusion_payload, status)
    print(f"\nABCD results saved to: {results_dir}")


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
        'requested_models': parse_csv_option(args.models) or ['efficientnet_b0'],
        'include_verification_backbones': bool(args.include_verification_backbones),
        'verification_backbone_models_added_by_option': (
            VERIFICATION_BACKBONE_MODELS if args.include_verification_backbones else []
        ),
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
    parser.add_argument('--include-verification-backbones', action='store_true',
                        help="Append the two verification robustness backbones, resnet18 and convnext_tiny, to --models")
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
    parser.add_argument(
        '--unfreeze-ratio', type=str, default='full',
        choices=[
            'full', 'half', 'third',
            'classifier_head_only', 'last_two_blocks_plus_head',
            'upper_half_backbone_plus_head', 'full_network',
        ],
        help=(
            "Stage 2 unfreeze mode. Legacy aliases: full, half, third. "
            "ABCD names: classifier_head_only, last_two_blocks_plus_head, "
            "upper_half_backbone_plus_head, full_network."
        ),
    )
    parser.add_argument('--stage1-only', action='store_true',
                        help="Run only Stage 1 (Base model) for timing measurement")
    parser.add_argument('--retrain', action='store_true',
                        help="Retrain from scratch with combined data (28 users + LOO calibration) for comparison")
    parser.add_argument('--compare-modes', action='store_true',
                        help="Compare Full vs Half fine-tuning using the SAME Stage 1 base model for fair comparison")
    parser.add_argument('--abc14', action='store_true',
                        help="Run A/base, B/neutral-auth-like, C/balanced14 personalization experiments")
    parser.add_argument('--abcd', action='store_true',
                        help="Run fresh reduced ABCD experiment: A base, B FiLM/prototype profiles, selected C 7/14-shot fine-tuning, selected D many-shot references")
    parser.add_argument('--abcd-resume-dir', type=Path, default=None,
                        help="Existing ABCD result directory to resume from; completed user-condition rows are skipped")
    parser.add_argument('--abcd-test-ratio', type=float, default=0.2,
                        help="Held-out user common test holdout ratio for ABCD")
    parser.add_argument('--abcd-film-lr', type=float, default=1e-4,
                        help="Learning rate for the B/FiLM feature-profile conditioner")
    parser.add_argument('--abcd-joint-embedding-dim', type=int, default=512,
                        help="Embedding dimension for B joint identity-emotion projection")
    parser.add_argument('--abcd-joint-lr', type=float, default=1e-4,
                        help="Learning rate for the B joint identity-emotion projection")
    parser.add_argument('--abcd-joint-identity-loss-weight', type=float, default=1.0,
                        help="Identity-loss weight for the B joint identity-emotion projection")
    parser.add_argument('--abcd-c-modes', type=str, default=','.join(ABCD_DEFAULT_C_MODE_KEYS),
                        help="Comma-separated C fine-tuning modes for ABCD, or all. Default runs upper_half_backbone_plus_head and full_network for a 10-condition run")
    parser.add_argument('--abcd-d-conditions', type=str, default=','.join(ABCD_DEFAULT_D_CONDITION_KEYS),
                        help="Comma-separated D condition keys for ABCD, or all. Default runs D_manyshot_full_network only for a 10-condition run")
    parser.add_argument('--abc-finetune-modes', type=str, default='full,half,classifier_only',
                        help="Comma-separated Stage 2 modes for legacy ABC14: full,half,third,classifier_only")
    parser.add_argument('--neutral-shots', type=str, default='2,6,12',
                        help="Comma-separated Neutral image counts for B/neutral-auth-like; max 12 supports all 37 users")
    parser.add_argument('--balanced-replacement-policy', type=str, default='keep_14',
                        choices=['keep_14', 'strict'],
                        help="How C/balanced14 handles missing pairs; keep_14 uses same-user similar-emotion fallbacks")
    parser.add_argument('--stage2-train-source', type=str, default='base_plus_personal',
                        choices=['base_plus_personal', 'personal_only'],
                        help="Stage 2 data: rehearse base train data plus personal samples, or use personal samples only")
    parser.add_argument('--dry-run', action='store_true',
                        help="For --abc14 or --abcd: verify data loading and write manifests without training")
    
    args = parser.parse_args()
    
    if args.abcd:
        run_abcd(args)
    elif args.abc14:
        run_abc14(args)
    elif args.compare_modes:
        run_compare_modes(args)
    else:
        run_loo_cv(args)


if __name__ == "__main__":
    main()
