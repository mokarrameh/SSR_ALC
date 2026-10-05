"""generate_kvasirv2_10fold_csv.py

Generate a stratified 10-fold split CSV for the Kvasir v2 dataset.

This file used to contain a hard-coded local path and executed on import.
It is a small command-line utility.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Kvasir v2 stratified K-fold split CSV.")
    p.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset root (class folders).")
    p.add_argument("--output_csv", type=str, default="kvasirv2_10fold_stratified.csv")
    p.add_argument("--num_folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    all_entries = []
    class_names = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])

    for class_name in class_names:
        class_dir = dataset_root / class_name
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                all_entries.append({"file-name": fname, "class-name": class_name})

    df = pd.DataFrame(all_entries)
    if df.empty:
        raise RuntimeError(f"No images found under: {dataset_root}")

    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    df["label-idx"] = df["class-name"].map(label_to_idx)

    df["split-index"] = -1
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["label-idx"])):
        df.loc[val_idx, "split-index"] = fold_idx

    out = Path(args.output_csv)
    df[["file-name", "class-name", "split-index"]].to_csv(out, sep=";", index=False)
    print(f"[INFO] CSV saved to: {out}")


if __name__ == "__main__":
    main()
