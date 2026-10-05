"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
Normal sampling only (shuffle for train, no class oversampling).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from collections import Counter

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from sklearn.utils.class_weight import compute_class_weight


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(root: Union[str, Path]) -> List[str]:
    root = str(root)
    files = []
    for f in os.listdir(root):
        p = os.path.join(root, f)
        if os.path.isfile(p) and Path(p).suffix.lower() in IMG_EXTS:
            files.append(f)
    files.sort()
    return files


def _seeded_subset(items: Sequence[str], subset_size: Optional[int], seed: Optional[int]) -> List[str]:
    if subset_size is None or subset_size <= 0 or subset_size >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), subset_size)


def compute_balanced_class_weights(targets: Sequence[int]) -> np.ndarray:
    """
    Returns sklearn-style balanced weights per class index, aligned with sorted(unique(targets)).
    Useful for nn.CrossEntropyLoss(weight=...).

    NOTE: You must map these weights to your class indices correctly.
    """
    y = np.asarray(targets)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    # weights[i] corresponds to classes[i]
    # if classes are [0..C-1] this is already aligned.
    return weights.astype(np.float32)


class CustomImageDataset(Dataset):
    """
    Unlabeled image dataset for a directory.
    - Filters by image extensions
    - Converts to RGB for transform consistency
    - Optional deterministic subset
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        subset_size: Optional[int] = None,
        subset_seed: Optional[int] = None,
    ):
        self.image_dir = str(image_dir)
        self.transform = transform

        all_files = _list_images(self.image_dir)
        self.image_files = _seeded_subset(all_files, subset_size, subset_seed)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class CustomImageDatasetV1(Dataset):
    """
    Labeled dataset created from (path, class_name) pairs.
    IMPORTANT: class_to_idx should come from TRAIN and be shared with val/test.
    """

    def __init__(
        self,
        image_label_pairs: Sequence[Tuple[str, str]],
        transform: Optional[Callable],
        class_to_idx: Optional[Dict[str, int]] = None,
        strict_labels: bool = True,
    ):
        self.samples = list(image_label_pairs)
        self.transform = transform
        self.loader = default_loader

        if class_to_idx is None:
            class_names = sorted({label for _, label in self.samples})
            self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        else:
            self.class_to_idx = dict(class_to_idx)

        self.targets: List[int] = []
        filtered_samples = []
        for path, label_name in self.samples:
            if label_name not in self.class_to_idx:
                if strict_labels:
                    raise ValueError(
                        f"Label '{label_name}' not in class_to_idx. "
                        "Build mapping from train and pass it to val/test."
                    )
                else:
                    continue
            self.targets.append(self.class_to_idx[label_name])
            filtered_samples.append((path, label_name))

        self.samples = filtered_samples

        inv = sorted(self.class_to_idx.items(), key=lambda x: x[1])
        self.class_names = [k for k, _ in inv]

    def __getitem__(self, index: int):
        path, label_name = self.samples[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[label_name]
        return img, label

    def __len__(self) -> int:
        return len(self.samples)


def create_unlabeled_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    train_batch_size: int,
    test_batch_size: int,
    val_batch_size: int,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    subset_seed: Optional[int] = None,
    num_workers: int = 1,
    pin_memory: bool = True,
):
    """
    Normal sampling for unlabeled:
      - train: shuffle=True
      - val/test: shuffle=False
    Uses batch_size consistently.
    """
    train_data = CustomImageDataset(train_dir, transform=train_transform, subset_size=train_subset, subset_seed=subset_seed)
    val_data = CustomImageDataset(val_dir, transform=test_transform, subset_size=val_subset, subset_seed=subset_seed)
    test_data = CustomImageDataset(test_dir, transform=test_transform, subset_size=test_subset, subset_seed=subset_seed)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def create_labeled_dataloaders_from_paths(
    train_paths: Sequence[Tuple[str, str]],
    test_paths: Optional[Sequence[Tuple[str, str]]],
    train_transform,
    test_transform,
    batch_size: int,
    num_workers: int = 1,
    pin_memory: bool = True,
    return_class_weights: bool = True,
    verbose: bool = True,
):
    """
    Normal sampling from explicit path lists.
    - Shares class mapping from train -> test (important!)
    - Returns class weights computed from TRAIN targets (optional)
    """
    train_dataset = CustomImageDatasetV1(train_paths, train_transform, class_to_idx=None)
    class_to_idx = train_dataset.class_to_idx

    test_loader = None
    if test_paths is not None and len(test_paths) > 0:
        test_dataset = CustomImageDatasetV1(test_paths, test_transform, class_to_idx=class_to_idx, strict_labels=True)
    else:
        test_dataset = None

    if verbose:
        train_counts = Counter(train_dataset.targets)
        print("\n[INFO] Training samples per class:")
        for cls, idx in class_to_idx.items():
            print(f"  {cls}: {train_counts.get(idx, 0)}")

        if test_dataset is not None:
            test_counts = Counter(test_dataset.targets)
            print("\n[INFO] Validation/Test samples per class:")
            for cls, idx in class_to_idx.items():
                print(f"  {cls}: {test_counts.get(idx, 0)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    class_weights = None
    if return_class_weights:
        class_weights = compute_balanced_class_weights(train_dataset.targets)

    return train_loader, test_loader, train_dataset.class_names, class_weights
