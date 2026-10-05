#!/usr/bin/env python3
"""only_ae_cluster_metrics.py

Compute ONLY the cluster metrics for an ONLY_AE checkpoint.

This script:
- Loads an AE checkpoint from: checkpoints/ONLY_AE/<AE_NAME>/model_weights/AE_100_epochs.pth
  (or a user-provided --ckpt)
- Reads a split CSV containing at least: 'class-name', 'file-name'
- Builds full image paths as: <image_root>/<class-name>/<file-name>
- Encodes images to latent vectors via the AE encoder
- Prints cluster metrics (no plots / no t-SNE)

Cluster metrics implemented:
- Intra-class distance (mean pairwise distance within each class)
- Inter-class distance (mean pairwise distance between class means)
- Cluster score = inter / (intra + 1e-8)
- Also prints sklearn label-based cluster quality scores when possible:
  silhouette, Calinski-Harabasz, Davies-Bouldin

Example:
  python analysis/only_ae_cluster_metrics.py \
    --ae_name deepConvAEV1 \
    --split_csv splits/hyperkvasir_5fold_no_ileum.csv \
    --image_root dataset/labeled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import random

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute cluster metrics for ONLY_AE AE latents (no plots).")

    p.add_argument(
        "--base_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="SSR_ALC repo root (default: inferred from this file location)",
    )

    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--ae_name",
        type=str,
        choices=["shallowConvAE", "convAE", "deepConvAE", "deepConvAEV1"],
        help="Which ONLY_AE model to use.",
    )
    model_group.add_argument(
        "--all",
        action="store_true",
        help="Compute metrics for all ONLY_AE variants found under checkpoints/ONLY_AE/.",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. Default: <base_dir>/checkpoints/ONLY_AE/<ae_name>/model_weights/AE_100_epochs.pth",
    )

    p.add_argument("--split_csv", type=str, required=True, help="CSV with columns 'class-name' and 'file-name'")
    p.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory containing class folders (e.g. dataset/labeled)",
    )
    p.add_argument("--sep", type=str, default=";", help="CSV separator (default: ';')")

    p.add_argument("--device", type=str, default=None, help="Torch device (cpu/cuda). Default: auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)

    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on number of samples to encode (0 = no cap).",
    )

    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the CSV rows before applying --max_samples (deterministic with --seed).",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed used when --shuffle is set.")

    p.add_argument(
        "--split_col",
        type=str,
        default=None,
        help="Optional column name to filter rows (e.g. 'split' or 'Fold').",
    )
    p.add_argument(
        "--split_value",
        type=str,
        default=None,
        help="Optional value for --split_col filter (string compare).",
    )

    return p.parse_args()


def load_pairs_from_split_csv(split_csv_path: Path, image_root: Path, sep: str, split_col: str | None, split_value: str | None):
    df = pd.read_csv(split_csv_path, sep=sep)

    required = {"class-name", "file-name"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"split CSV missing columns {missing}. Found: {list(df.columns)}")

    if split_col is not None:
        if split_col not in df.columns:
            raise ValueError(f"--split_col={split_col} not in CSV columns: {list(df.columns)}")
        if split_value is None:
            raise ValueError("--split_col was provided but --split_value is None")
        df = df[df[split_col].astype(str) == str(split_value)]

    pairs = []
    for _, row in df.iterrows():
        cls = str(row["class-name"])
        fn = str(row["file-name"])
        pairs.append((str(image_root / cls / fn), cls))

    return pairs


def compute_inter_intra_metrics(latents: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    unique_classes = np.unique(labels)

    intra_dists: list[float] = []
    class_means: list[np.ndarray] = []

    for c in unique_classes:
        class_points = latents[labels == c]
        if len(class_points) < 2:
            continue
        dists = pairwise_distances(class_points)
        intra_dists.append(float(np.mean(dists)))
        class_means.append(np.mean(class_points, axis=0))

    if not intra_dists or len(class_means) < 2:
        return float("nan"), float("nan"), float("nan")

    d_intra = float(np.mean(intra_dists))

    class_means_arr = np.array(class_means)
    inter_dists = pairwise_distances(class_means_arr)
    d_inter = float(np.mean(inter_dists[np.triu_indices(len(class_means_arr), k=1)]))

    score = float(d_inter / (d_intra + 1e-8))
    return d_intra, d_inter, score


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base_dir}")

    sys.path.insert(0, str(base_dir))

    split_csv = Path(args.split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(f"split_csv not found: {split_csv}")

    image_root = Path(args.image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms

        from engine.models.autoencoder import shallowConvAE, ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1
        from utils.data_setup import CustomImageDatasetV1
    except Exception as e:
        raise SystemExit(f"Missing dependencies/imports: {type(e).__name__}: {e}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run_one(ae_name: str) -> None:
        ckpt = Path(args.ckpt) if args.ckpt else (base_dir / "checkpoints" / "ONLY_AE" / ae_name / "model_weights" / "AE_100_epochs.pth")
        if not ckpt.exists():
            print(f"\n[SKIP] {ae_name}: checkpoint not found: {ckpt}")
            return

        if ae_name == "shallowConvAE":
            ae = shallowConvAE().to(device)
        elif ae_name == "convAE":
            ae = ConvAutoencoder().to(device)
        elif ae_name == "deepConvAE":
            ae = DeepConvAutoencoder().to(device)
        elif ae_name == "deepConvAEV1":
            ae = DeepConvAutoencoderV1().to(device)
        else:
            print(f"\n[SKIP] unknown ae_name: {ae_name}")
            return

        ae.eval()
        ae.load_state_dict(torch.load(str(ckpt), map_location=device))

        pairs = load_pairs_from_split_csv(
            split_csv_path=split_csv,
            image_root=image_root,
            sep=args.sep,
            split_col=args.split_col,
            split_value=args.split_value,
        )

        if args.shuffle:
            rng = random.Random(int(args.seed))
            rng.shuffle(pairs)

        if args.max_samples and args.max_samples > 0:
            pairs = pairs[: int(args.max_samples)]

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        dataset = CustomImageDatasetV1(pairs, transform=transform, class_to_idx=None, strict_labels=True)
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
        )

        all_latents = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                encoded = ae(batch_x)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                all_latents.append(encoded.detach().cpu())
                all_labels.append(batch_y.detach().cpu())

        latents = torch.cat(all_latents, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        d_intra, d_inter, score = compute_inter_intra_metrics(latents, labels)

        print("\n" + "=" * 88)
        print("[ONLY_AE CLUSTER METRICS]")
        print(f"ae_name: {ae_name}")
        print(f"ckpt:    {ckpt}")
        print(f"csv:     {split_csv}")
        if args.split_col is not None:
            print(f"filter:  {args.split_col} == {args.split_value}")
        print(f"n:       {latents.shape[0]} samples")
        print(f"dim:     {latents.shape[1]} latent dim")
        print(f"classes: {len(np.unique(labels))} ({dataset.class_names})")

        print("\n[INTER/INTRA]")
        print(f"intra_class_distance: {d_intra:.6f}")
        print(f"inter_class_distance: {d_inter:.6f}")
        print(f"inter_over_intra:     {score:.6f}")

        # Optional sklearn metrics (label-based cluster quality)
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

            if len(np.unique(labels)) >= 2 and latents.shape[0] > len(np.unique(labels)):
                try:
                    sil = float(silhouette_score(latents, labels))
                except Exception:
                    sil = float("nan")
                try:
                    ch = float(calinski_harabasz_score(latents, labels))
                except Exception:
                    ch = float("nan")
                try:
                    db = float(davies_bouldin_score(latents, labels))
                except Exception:
                    db = float("nan")

                print("\n[SKLEARN]")
                print(f"silhouette:          {sil:.6f}")
                print(f"calinski_harabasz:   {ch:.6f}")
                print(f"davies_bouldin:      {db:.6f}")
            else:
                print("\n[SKLEARN]")
                print("Not enough classes/samples for silhouette/CH/DB.")
        except Exception as e:
            print("\n[SKLEARN]")
            print(f"skipped (missing sklearn metrics?): {type(e).__name__}: {e}")

    if args.all:
        root = base_dir / "checkpoints" / "ONLY_AE"
        if root.exists():
            candidates = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
            # Keep a stable preferred order when folders exist.
            preferred = ["shallowConvAE", "convAE", "deepConvAE", "deepConvAEV1"]
            ordered = [n for n in preferred if n in candidates] + [n for n in candidates if n not in preferred]
        else:
            ordered = ["shallowConvAE", "convAE", "deepConvAE", "deepConvAEV1"]
        for name in ordered:
            run_one(name)
    else:
        run_one(str(args.ae_name))


if __name__ == "__main__":
    main()
