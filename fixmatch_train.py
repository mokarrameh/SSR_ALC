"""FixMatch baseline training script.

Trains an image classifier using:
  - supervised CE on labeled images
  - FixMatch pseudo-label loss on unlabeled images (weak->pseudo, strong->train)

This script follows the repo's conventions:
  - K-fold splits via utils.fold_handle
  - Output dirs via engine.utils.create_output_dir
  - Training loop via engine.fixmatch_engine.fit_fixmatch

Restored from the original `fixmatch.py`; training logic is unchanged.

Run (example):
  python fixmatch_train.py --config configs/fixmatch.yaml --data_path /path/to/data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import os
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from colorama import Fore, Style
from collections import Counter

# Options (CycleGAN style)
from options.train_options import TrainOptions

# Custom modules
from engine.utils import create_output_dir
from engine.fixmatch_engine import fit_fixmatch
from utils.data_setup import CustomImageDatasetV1, compute_balanced_class_weights
from utils.fold_handle import get_fold_train_val_test_data, add_external_imgs_train_only, resolve_split_csv
from utils.data_setup import create_unlabeled_dataloaders, create_labeled_dataloaders_from_paths


def seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_labeled_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(280),
        transforms.RandomCrop(256),
        transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    return train_transform, test_transform


def build_fixmatch_unlabeled_transforms():
    """Return (weak_transform, strong_transform) applied to the same underlying PIL image."""

    # Weak aug ~= crop/flip.
    weak = transforms.Compose(
        [
            transforms.Resize(280),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # Strong aug: add heavier photometric/geometric noise.
    strong = transforms.Compose(
        [
            transforms.Resize(280),
            transforms.RandomCrop(256),
            transforms.RandomPerspective(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    return weak, strong


class FixMatchUnlabeledDataset(Dataset):
    """Unlabeled dataset returning (weak_aug_tensor, strong_aug_tensor)."""

    def __init__(
        self,
        image_dir: Union[str, Path],
        weak_transform: Callable,
        strong_transform: Callable,
        subset_size: Optional[int] = None,
        subset_seed: Optional[int] = None,
    ):
        self.Image = Image
        self.image_dir = str(image_dir)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        files = []
        for f in os.listdir(self.image_dir):
            p = os.path.join(self.image_dir, f)
            if os.path.isfile(p) and Path(p).suffix.lower() in exts:
                files.append(f)
        files.sort()

        if subset_size is not None and subset_size > 0 and subset_size < len(files):
            rng = random.Random(subset_seed)
            files = rng.sample(files, subset_size)

        self.image_files = files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        import os

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = self.Image.open(img_path).convert("RGB")

        x_w = self.weak_transform(img)
        x_s = self.strong_transform(img)
        return x_w, x_s


def make_fixmatch_model(num_classes: int) -> nn.Module:
    # Avoid internet downloads (weights=None). Handle older torchvision (pretrained=False).
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _make_labeled_loaders(
    train_paths: Sequence[Tuple[str, str]],
    val_paths: Sequence[Tuple[str, str]],
    test_paths: Sequence[Tuple[str, str]],
    train_transform,
    test_transform,
    batch_size: int,
    num_workers: int,
):
    train_dataset = CustomImageDatasetV1(train_paths, transform=train_transform, class_to_idx=None)
    class_to_idx = train_dataset.class_to_idx

    val_dataset = CustomImageDatasetV1(val_paths, transform=test_transform, class_to_idx=class_to_idx, strict_labels=True)
    test_dataset = CustomImageDatasetV1(test_paths, transform=test_transform, class_to_idx=class_to_idx, strict_labels=True)


    train_counts = Counter(train_dataset.targets)
    print("\n[INFO] Training samples per class:")
    for cls, idx in class_to_idx.items():
        print(f"  {cls}: {train_counts.get(idx, 0)}")

    val_counts = Counter(val_dataset.targets)
    print("\n[INFO] Validation samples per class:")
    for cls, idx in class_to_idx.items():
        print(f"  {cls}: {val_counts.get(idx, 0)}")

    test_counts = Counter(test_dataset.targets)
    print("\n[INFO] Test samples per class:")
    for cls, idx in class_to_idx.items():
        print(f"  {cls}: {test_counts.get(idx, 0)}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_weights = compute_balanced_class_weights(train_dataset.targets)
    return train_loader, val_loader, test_loader, train_dataset.class_names, class_weights


def main():
    opt = TrainOptions().parse()
    if opt.name is None:
        opt.name = "FIXMATCH"

    # ---- colors ----
    c_ = Fore.GREEN
    i_ = Fore.BLUE
    b_ = Fore.MAGENTA
    sr_ = Style.RESET_ALL

    seed_everything(opt.seed)
    device = torch.device(opt.device)

    # ---- paths ----
    data_path = Path(opt.data_path)
    l_image_path = data_path / opt.labeled_dir
    u_image_path = data_path / opt.unlabeled_dir
    split_csv_path = resolve_split_csv(opt.split_csv, l_image_path)
    extra_img_dir = (l_image_path / opt.extra_img_dir) if opt.extra_img_dir else None

    u_train_path = u_image_path / opt.u_train_subdir
    u_val_path = u_image_path / opt.u_val_subdir
    u_test_path = u_image_path / opt.u_test_subdir

    # ---- transforms ----
    labeled_train_tf, labeled_test_tf = build_labeled_transforms()
    weak_u_tf, strong_u_tf = build_fixmatch_unlabeled_transforms()

    # ---- unlabeled loaders (FixMatch needs weak+strong views) ----
    u_train_dataset = FixMatchUnlabeledDataset(
        image_dir=u_train_path,
        weak_transform=weak_u_tf,
        strong_transform=strong_u_tf,
    )
    u_train_loader = DataLoader(
        u_train_dataset,
        batch_size=opt.u_train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # NOTE: FixMatch baseline does not compute AE/test loss here; keep only metrics
    # that are actually recorded inside engine.fixmatch_engine.fit_fixmatch.
    fold_results = {
        "test_acc": [],
        "test_acc_mca": [],
        "test_f1": [],
        "test_f1_mca": [],
        "test_auroc": [],
        "test_auroc_mca": [],
        "test_precision": [],
        "test_recall": [],
    }

    for fold in opt.folds:
        print(f"\n{Fore.CYAN}[INFO] Fold {fold}{Style.RESET_ALL}")

        train_paths, val_paths, test_paths = get_fold_train_val_test_data(
            split_csv_path,
            l_image_path,
            fold_idx=fold,
            val_ratio=0.2,
            seed=opt.seed,
        )

        # ---- optional external images to TRAIN only ----
        added_subset = []
        if extra_img_dir is not None and opt.extra_img_n > 0:
            train_paths, added_subset = add_external_imgs_train_only(
                train_paths,
                val_paths,
                test_paths,
                extra_dir=extra_img_dir,
                n_extra=opt.extra_img_n,
                seed=opt.extra_seed,
                label="ileum",
            )
        print(f"[INFO] Added {len(added_subset)} external Ileum samples to TRAIN only (fold={fold}).")

        l_train_loader, l_val_loader, l_test_loader, class_names, class_weights = _make_labeled_loaders(
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths,
            train_transform=labeled_train_tf,
            test_transform=labeled_test_tf,
            batch_size=opt.l_batch_size,
            num_workers=opt.num_workers,
        )

        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        print(f"{b_}[INFO] Experiment: {opt.name} / fold={fold}{sr_}")
        print(f"{i_}[INFO] Labeled train dl length: {len(l_train_loader)}{sr_}")
        print(f"[INFO] Labeled val dl length: {len(l_val_loader)}")
        print(f"[INFO] Labeled test dl length: {len(l_test_loader)}")
        print(f"[INFO] Unlabeled train dl length: {len(u_train_loader)}")
        print(f"[INFO] Class weights: {weights_tensor}")
        print(f"{i_}[INFO] Class names: {class_names}{sr_}")
        print("-" * 50 + "\n")

        # ---- model/optim ----
        model = make_fixmatch_model(num_classes=len(class_names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.cl_lr, weight_decay=opt.weight_decay)

        # ---- outputs ----
        base_out = (
            Path(opt.checkpoints_dir)
            / opt.name
            / "fixmatch_resnet18"
            / f"{opt.extra_img_n}"
            / f"fold_{fold}"
        )
        training_progress_dir, model_weights_dir = create_output_dir(base_out)

        # ---- fit ----
        fit_fixmatch(
            model=model,
            train_labeled_loader=l_train_loader,
            train_unlabeled_loader=u_train_loader,
            val_loader=l_val_loader,
            test_loader=l_test_loader,
            optimizer=optimizer,
            device=device,
            weak_transform=None,
            strong_transform=None,
            training_progress_dir=training_progress_dir,
            model_weights_dir=model_weights_dir,
            num_epochs=opt.epochs,
            fold=fold,
            fold_results=fold_results,
            lambda_u=1.0,
            tau=0.95,
            class_weights=weights_tensor,
            scheduler=None,
            monitor="f1_macro",
        )

        print(f"{c_}Experiment completed successfully.{sr_}")
        print("-" * 50 + "\n")

    # ---- fold averages ----
    print(f"Average performance across folds: ")
    for metric, values in fold_results.items():
        mean = np.mean(values) if len(values) else float("nan")
        std = np.std(values) if len(values) else float("nan")
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
