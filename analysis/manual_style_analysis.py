#!/usr/bin/env python3
"""manual_style_analysis.py

A simple terminal script that reproduces the manual EXP_4-style analysis
(print metrics + save the same SVG plots) without Jupyter.

It reads checkpoints in these layouts:
  checkpoints/<EXP_NAME>/<RUN_ID>/<MODEL>/
    fold_0/model_weights/0.csv
    fold_0/model_weights/0_test.csv
    fold_1/model_weights/1.csv
    ...

And ONLY_AE layout:
    checkpoints/ONLY_AE/<AE_NAME>/
        model_weights/AE_100_epochs.pth
        model_weights/training_results.csv

Outputs:
  results_analysis/<EXP_NAME>_<RUN_ID>/<MODEL>/*.svg

This script intentionally uses the existing plotting helpers in
`visualization/vis.py` to match the manual notebooks.
"""

from __future__ import annotations

import argparse
import ast
import sys
import re
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances

_DEFAULT_CLASS_NAMES = ["cecum", "ileum", "pylorus", "rectum", "stomach", "z-line"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual-style SSR_ALC checkpoint analysis (no notebooks).")

    p.add_argument(
        "--base_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="SSR_ALC repo root (default: inferred from this file location)",
    )
    p.add_argument("--exp", type=str, required=True, help="EXP name, e.g. EXP_1, EXP_2, EXP_3, EXP_4, FIXMATCH")
    p.add_argument(
        "--run_id",
        type=str,
        default="40",
        help="Subfolder under checkpoints/<exp>/ (default: 40). Can include nested paths, e.g. fixmatch_resnet18/40",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional explicit run dir. If set, overrides --exp/--run_id resolution.",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output root directory (default: results_analysis/<exp>_<run_id>)",
    )

    p.add_argument("--models", nargs="*", default=None, help="Optional subset of model dir names")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--class_names", nargs="+", default=_DEFAULT_CLASS_NAMES)

    p.add_argument(
        "--best_fold_extra",
        action="store_true",
        help="Also generate extra plots for the best-performing fold.",
    )
    p.add_argument(
        "--best_fold_by",
        choices=["val_f1", "test_f1"],
        default="test_f1",
        help="Metric for selecting best fold (default: test_f1).",
    )

    # Optional: t-SNE plot from AE latents (replicates the notebook t-SNE, but via CLI)
    p.add_argument("--tsne", action="store_true", help="Also create a t-SNE plot from AE latents (saves SVG)")
    p.add_argument(
        "--cluster_metrics",
        action="store_true",
        help="Compute and print cluster metrics from AE latents (no plots).",
    )
    p.add_argument(
        "--tsne_split_csv",
        type=str,
        default=None,
        help="Split CSV path (must contain columns 'class-name' and 'file-name')",
    )
    p.add_argument(
        "--tsne_image_root",
        type=str,
        default=None,
        help="Root dir containing class folders (e.g., .../HK_ANA)",
    )
    p.add_argument("--tsne_sep", type=str, default=";", help="CSV separator (default: ';')")
    p.add_argument("--tsne_device", type=str, default=None, help="Torch device, e.g. cpu or cuda")
    p.add_argument("--tsne_batch_size", type=int, default=32)
    p.add_argument("--tsne_num_workers", type=int, default=1)
    p.add_argument("--tsne_samples", type=int, default=2000)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--tsne_seed", type=int, default=42)
    p.add_argument(
        "--shuffle_samples",
        action="store_true",
        help="Shuffle split CSV rows before taking --tsne_samples (uses --tsne_seed).",
    )
    p.add_argument(
        "--tsne_ae_fold",
        type=int,
        default=None,
        help="Which fold's AE checkpoint to use (default: best fold by --best_fold_by)",
    )

    return p.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fold_mw(model_dir: Path, fold_idx: int) -> Path:
    return model_dir / f"fold_{fold_idx}" / "model_weights"


def _safe_tag(s: str) -> str:
    return str(s).strip().replace("/", "_").replace(" ", "_")


def _is_model_dir(p: Path) -> bool:
    return (p / "fold_0" / "model_weights").exists()


def _is_only_ae_dir(p: Path) -> bool:
    mw = p / "model_weights"
    if not mw.exists():
        return False
    has_ckpt = (mw / "AE_100_epochs.pth").exists() or bool(list(mw.glob("AE*.pth")))
    has_csv = (mw / "training_results.csv").exists()
    return has_ckpt and has_csv


def _is_fold_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("fold_") and (p / "model_weights").exists()


def discover_model_dirs(run_dir: Path) -> list[Path]:
    """Return model directories under run_dir.

    Two supported layouts:
    1) checkpoints/<EXP>/<RUN_ID>/<MODEL>/fold_k/...
    2) checkpoints/<EXP>/<MODEL>/<RUN_ID>/fold_k/... (FIXMATCH)
       In this case run_dir itself is the model directory.
    """
    if _is_model_dir(run_dir) or _is_only_ae_dir(run_dir):
        return [run_dir]
    subdirs = [p for p in sorted(run_dir.iterdir()) if p.is_dir()]
    model_dirs = [p for p in subdirs if _is_model_dir(p) or _is_only_ae_dir(p)]
    return model_dirs


def model_output_name(model_dir: Path, run_dir: Path) -> str:
    # If run_dir itself is a model dir (layout #2), name it after its parent (e.g. fixmatch_resnet18)
    if model_dir.resolve() == run_dir.resolve() and _is_model_dir(run_dir):
        return run_dir.parent.name
    return model_dir.name


def compute_cluster_metrics(latents: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
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


def infer_ae_name_from_model_dir(model_dir_name: str) -> str | None:
    n = str(model_dir_name).strip().lower()
    # Common names in this repo: simpleNN_convAE, simpleNN_deepConvAE, simpleNN_deepConvAEV1
    if "deepconvaev1" in n:
        return "deepConvAEV1"
    if "deepconvae" in n:
        return "deepConvAE"
    if "convae" in n:
        return "convAE"
    if "shallow" in n:
        return "shallowConvAE"
    return None


def find_experiment_ae_checkpoint(model_dir: Path, fold_preference: int | None) -> Path | None:
    """Prefer the AE checkpoint trained in the experiment folder.

    Expected to live under: fold_k/model_weights/AE_*.pth
    """

    def candidates_in_fold(mw: Path, fold_idx: int) -> list[Path]:
        if not mw.exists():
            return []
        cands = sorted(mw.glob("AE*.pth"))
        if not cands:
            return []
        # Prefer files that include the fold index in the filename (e.g. AE_100_epochs_3fold.pth)
        patt = re.compile(rf"(^|_)({fold_idx})fold(\D|$)", re.IGNORECASE)
        matched = [p for p in cands if patt.search(p.name)]
        return matched or cands

    if fold_preference is not None:
        mw = _fold_mw(model_dir, int(fold_preference))
        cands = candidates_in_fold(mw, int(fold_preference))
        if cands:
            return cands[0]

    # Otherwise scan all folds in order.
    for fold_dir in sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")]):
        m = re.search(r"(\d+)$", fold_dir.name)
        fold_idx = int(m.group(1)) if m else 0
        mw = fold_dir / "model_weights"
        cands = candidates_in_fold(mw, fold_idx)
        if cands:
            return cands[0]
    return None


def load_paths_from_split_csv(split_csv_path: Path, image_root: Path, sep: str) -> list[tuple[str, str]]:
    df = pd.read_csv(split_csv_path, sep=sep)
    required = {"class-name", "file-name"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"split CSV missing columns {missing}. Found: {list(df.columns)}")
    out: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        cls = str(row["class-name"])
        fn = str(row["file-name"])
        out.append((str(image_root / cls / fn), cls))
    return out


def load_fold_data_from_checkpoints(model_dir: Path, num_folds: int, fix_list_string) -> list[pd.DataFrame]:
    fold_data: list[pd.DataFrame] = []
    for i in range(num_folds):
        df = pd.read_csv(_fold_mw(model_dir, i) / f"{i}.csv")
        for col in ["val_acc_mca", "val_f1_mca", "val_auroc_mca"]:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(fix_list_string)
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        fold_data.append(df)
    return fold_data


def load_test_data_from_checkpoints(model_dir: Path, num_folds: int, fix_list_string) -> list[pd.DataFrame]:
    fold_data: list[pd.DataFrame] = []
    for i in range(num_folds):
        df = pd.read_csv(_fold_mw(model_dir, i) / f"{i}_test.csv")
        for col in ["test_acc_mca", "test_f1_mca", "test_auroc_mca"]:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(fix_list_string)
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        fold_data.append(df)
    return fold_data


def best_fold_by_val_f1(all_fold_data: list[pd.DataFrame]) -> tuple[int, int | None, float]:
    best_idx = int(np.argmax([float(df["val_f1"].max()) for df in all_fold_data]))
    try:
        best_epoch = int(all_fold_data[best_idx].loc[all_fold_data[best_idx]["val_f1"].idxmax(), "Epoch"])
    except Exception:
        best_epoch = None
    best_val = float(all_fold_data[best_idx]["val_f1"].max())
    return best_idx, best_epoch, best_val


def best_fold_by_test_f1(test_folds: list[pd.DataFrame]) -> tuple[int, float]:
    # Each test df is a single-row csv (Fold column present)
    vals = []
    for df in test_folds:
        row = df.iloc[-1]
        vals.append(float(row.get("test_f1")))
    best_idx = int(np.argmax(vals))
    return best_idx, float(vals[best_idx])


def plot_metric_single_fold(df: pd.DataFrame, metric_name: str, ylabel: str, save_dir: Path, filename: str, plot_configs) -> None:
    plot_configs()
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 110)
    plt.plot(df["Epoch"], df[f"train_{metric_name}"] * 100, label=f"Train {metric_name}", color="blue")
    plt.plot(df["Epoch"], df[f"val_{metric_name}"] * 100, label=f"Val {metric_name}", color="red")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    _ensure_dir(save_dir)
    out = save_dir / f"{filename}.svg"
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_precision_recall_single_fold(df: pd.DataFrame, save_dir: Path, filename: str, plot_configs) -> None:
    plot_configs()
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1.1)

    plt.plot(df["Epoch"], df["train_precision"], label="Train Precision", color="blue")
    plt.plot(df["Epoch"], df["val_precision"], label="Val Precision", color="red")
    plt.plot(df["Epoch"], df["train_recall"], label="Train Recall", color="green")
    plt.plot(df["Epoch"], df["val_recall"], label="Val Recall", color="orange")

    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Score", fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    _ensure_dir(save_dir)
    out = save_dir / f"{filename}.svg"
    plt.savefig(out, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_mca_metric_single_fold_for_classes(
    df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    class_names: list[str],
    save_dir: Path,
    plot_configs,
) -> None:
    num_classes = len(class_names)
    _ensure_dir(save_dir)

    for cls_idx in range(num_classes):
        plot_configs()
        plt.figure(figsize=(10, 6))
        plt.ylim(0, 1.1)

        epochs = df["Epoch"].values
        train_vals = np.array(
            df[f"train_{metric_name}"]
            .apply(lambda x: np.fromstring(str(x).strip("[]"), sep=" "))
            .apply(lambda arr: arr[cls_idx] if arr.size > cls_idx else np.nan)
        )
        val_vals = np.array(
            df[f"val_{metric_name}"]
            .apply(lambda arr: arr[cls_idx] if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) > cls_idx else np.nan)
        )

        plt.plot(epochs, train_vals, label="Train", color="blue", marker="o", markersize=4)
        plt.plot(epochs, val_vals, label="Val", color="red", linestyle="--", marker="x", markersize=4)

        plt.xlabel("Epoch", fontweight="bold")
        plt.ylabel(ylabel, fontweight="bold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out = save_dir / f"{metric_name}_best_fold_class_{class_names[cls_idx]}.svg"
        plt.savefig(out, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base_dir}")

    sys.path.insert(0, str(base_dir))
    from visualization.vis import (
        plot_configs,
        print_max_metrics,
        summarize_best_fold_results,
        summarize_test_results,
        plot_AE_Loss_CV_all_folds,
        plot_metric_CV_all_folds,
        plot_precision_recall_CV_all_folds,
        plot_mca_metric_CV_all_folds_for_classes,
        fix_list_string,
    )

    if args.tsne or args.cluster_metrics:
        if not args.tsne_split_csv or not args.tsne_image_root:
            raise SystemExit("--tsne/--cluster_metrics requires --tsne_split_csv and --tsne_image_root")

    run_dir = Path(args.run_dir) if args.run_dir else (base_dir / "checkpoints" / args.exp / args.run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (base_dir / "results_analysis" / f"{_safe_tag(args.exp)}_{_safe_tag(args.run_id)}")
    )
    _ensure_dir(out_dir)

    # Discover model dirs
    model_dirs = discover_model_dirs(run_dir)
    if args.models and len(model_dirs) > 1:
        wanted = set(args.models)
        model_dirs = [p for p in model_dirs if p.name in wanted]

    if not model_dirs:
        raise RuntimeError(f"No model dirs found under {run_dir}")

    class_names = list(args.class_names)

    print(f"RUN_DIR: {run_dir}")
    print(f"OUT_DIR: {out_dir}")
    print(f"MODELS: {[p.name for p in model_dirs]}")

    for model_dir in model_dirs:
        model_name = model_output_name(model_dir, run_dir)
        print("\n" + "=" * 120)
        print(f"MODEL: {model_name}")
        print("=" * 120)

        save_dir = out_dir / model_name
        _ensure_dir(save_dir)

        is_only_ae = _is_only_ae_dir(model_dir)

        if is_only_ae:
            mw = model_dir / "model_weights"
            tr_csv = mw / "training_results.csv"
            if tr_csv.exists():
                try:
                    df_tr = pd.read_csv(tr_csv)
                    if {"Epoch", "AE_train_loss", "AE_val_loss"}.issubset(set(df_tr.columns)):
                        best_idx = int(df_tr["AE_val_loss"].idxmin()) if len(df_tr) else -1
                        if best_idx >= 0:
                            best_epoch = int(df_tr.loc[best_idx, "Epoch"])
                            best_val = float(df_tr.loc[best_idx, "AE_val_loss"])
                            best_tr = float(df_tr.loc[best_idx, "AE_train_loss"])
                            print(f"[ONLY_AE] best val AE loss: {best_val:.6f} (epoch={best_epoch}, train={best_tr:.6f})")
                except Exception as e:
                    print(f"[WARN] failed reading ONLY_AE training_results.csv: {e}")

            all_fold_data = []
            test_data = []
        else:
            # Per-fold prints (like notebook)
            for fold in range(int(args.num_folds)):
                print(f"Data related to fold #{fold}.")
                df_fold = pd.read_csv(_fold_mw(model_dir, fold) / f"{fold}.csv")
                plot_configs()
                has_ae_loss = "AE_val_loss" in df_fold.columns
                print_max_metrics(df_fold, class_names, loss=has_ae_loss)
                print("*" * 100)

            # Summaries
            all_fold_data = load_fold_data_from_checkpoints(model_dir, int(args.num_folds), fix_list_string)
            _summary, _mca_array, _best_epochs = summarize_best_fold_results(all_fold_data)

            test_data = load_test_data_from_checkpoints(model_dir, int(args.num_folds), fix_list_string)
            _test_summary = summarize_test_results(test_data)

        # Cluster metrics only (no plots): supported for ANY model_dir where we can locate an AE checkpoint.
        if args.cluster_metrics:
            ae_name = None
            if args.exp.strip().upper() == "ONLY_AE":
                ae_name = str(args.run_id).strip()
            else:
                ae_name = infer_ae_name_from_model_dir(model_dir.name)

            if ae_name is None:
                print(f"[WARN] cannot infer AE type for cluster metrics from model dir: {model_dir.name}; skipping")
            else:
                ckpt = None
                ckpt_source = "experiment"
                if is_only_ae:
                    # Prefer the ONLY_AE checkpoint in this folder
                    cands = sorted((model_dir / "model_weights").glob("AE*.pth"))
                    ckpt = cands[0] if cands else None
                    ckpt_source = "ONLY_AE"
                else:
                    # Prefer experiment AE checkpoint; fall back to ONLY_AE
                    try:
                        ckpt = find_experiment_ae_checkpoint(model_dir, fold_preference=None)
                    except Exception:
                        ckpt = None
                    if ckpt is None:
                        ckpt = base_dir / "checkpoints" / "ONLY_AE" / ae_name / "model_weights" / "AE_100_epochs.pth"
                        ckpt_source = "ONLY_AE"

                if ckpt is None or not Path(ckpt).exists():
                    print(f"[WARN] AE checkpoint not found ({ckpt_source}): {ckpt}; skipping cluster metrics")
                else:
                    try:
                        import torch
                        from torchvision import transforms

                        from engine.models.autoencoder import (
                            shallowConvAE,
                            ConvAutoencoder,
                            DeepConvAutoencoder,
                            DeepConvAutoencoderV1,
                        )
                        from utils.data_setup import CustomImageDatasetV1
                        from torch.utils.data import DataLoader
                    except Exception as e:
                        print(f"[WARN] missing deps for cluster metrics ({type(e).__name__}: {e}); skipping")
                    else:
                        device = args.tsne_device or ("cuda" if torch.cuda.is_available() else "cpu")

                        if ae_name == "shallowConvAE":
                            ae = shallowConvAE().to(device)
                        elif ae_name == "convAE":
                            ae = ConvAutoencoder().to(device)
                        elif ae_name == "deepConvAE":
                            ae = DeepConvAutoencoder().to(device)
                        elif ae_name == "deepConvAEV1":
                            ae = DeepConvAutoencoderV1().to(device)
                        else:
                            print(f"[WARN] unknown AE type {ae_name}; skipping cluster metrics")
                            ae = None

                        if ae is not None:
                            try:
                                ae.load_state_dict(torch.load(str(ckpt), map_location=device))
                                ae.eval()
                            except Exception as e:
                                print(f"[WARN] failed loading AE checkpoint {ckpt}: {e}; skipping cluster metrics")
                            else:
                                split_csv = Path(args.tsne_split_csv)
                                image_root = Path(args.tsne_image_root)
                                pairs = load_paths_from_split_csv(split_csv, image_root, args.tsne_sep)

                                if args.shuffle_samples:
                                    rng = random.Random(int(args.tsne_seed))
                                    rng.shuffle(pairs)

                                # Sample first N (same semantics as tsne)
                                if int(args.tsne_samples) > 0:
                                    pairs = pairs[: int(args.tsne_samples)]

                                test_transform = transforms.Compose(
                                    [
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                    ]
                                )

                                ds = CustomImageDatasetV1(pairs, transform=test_transform, class_to_idx=None, strict_labels=True)
                                dl = DataLoader(
                                    ds,
                                    batch_size=int(args.tsne_batch_size),
                                    shuffle=False,
                                    num_workers=int(args.tsne_num_workers),
                                    pin_memory=True,
                                )

                                all_latents = []
                                all_labels = []
                                with torch.no_grad():
                                    for x, y in dl:
                                        x = x.to(device)
                                        encoded, _ = ae(x)
                                        all_latents.append(encoded.detach().cpu())
                                        all_labels.append(y.detach().cpu())

                                latents = torch.cat(all_latents, dim=0).numpy()
                                labels = torch.cat(all_labels, dim=0).numpy()

                                d_intra, d_inter, ratio = compute_cluster_metrics(latents, labels)
                                print("\n[CLUSTER METRICS]")
                                print(f"AE: {ae_name} | ckpt={ckpt} [{ckpt_source}]")
                                print(f"n={latents.shape[0]} dim={latents.shape[1]} classes={len(np.unique(labels))}")
                                print(f"Intra-class distance: {d_intra:.4f}")
                                print(f"Inter-class distance: {d_inter:.4f}")
                                print(f"Cluster score (inter/intra): {ratio:.4f}")

                                # Extra common metrics (optional)
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
                                        print("[SKLEARN]")
                                        print(f"silhouette: {sil:.4f}")
                                        print(f"calinski_harabasz: {ch:.4f}")
                                        print(f"davies_bouldin: {db:.4f}")
                                except Exception:
                                    pass

        # Optional: t-SNE plot (per model)
        if args.tsne:
            ae_name = infer_ae_name_from_model_dir(model_dir.name)
            if ae_name is None:
                print(f"[WARN] cannot infer AE type from model dir: {model_dir.name}; skipping t-SNE")
            else:
                # Pick which fold's AE to use
                if args.tsne_ae_fold is not None:
                    ae_fold = int(args.tsne_ae_fold)
                else:
                    # For t-SNE we default to the "best performing" fold by test_f1.
                    # If test results are unavailable for any reason, fall back to val_f1.
                    try:
                        ae_fold, _best_val = best_fold_by_test_f1(test_data)
                    except Exception:
                        try:
                            ae_fold, _best_epoch, _best_val = best_fold_by_val_f1(all_fold_data)
                        except Exception:
                            ae_fold = 0

                exp_ckpt = find_experiment_ae_checkpoint(model_dir, fold_preference=ae_fold)
                ckpt = exp_ckpt
                ckpt_source = "experiment"
                if ckpt is None:
                    ckpt = base_dir / "checkpoints" / "ONLY_AE" / ae_name / "model_weights" / "AE_100_epochs.pth"
                    ckpt_source = "ONLY_AE"

                if ckpt is None or not Path(ckpt).exists():
                    print(f"[WARN] AE checkpoint not found ({ckpt_source}): {ckpt}; skipping t-SNE")
                else:
                    try:
                        import torch
                        from torchvision import transforms

                        from engine.models.autoencoder import (
                            shallowConvAE,
                            ConvAutoencoder,
                            DeepConvAutoencoder,
                            DeepConvAutoencoderV1,
                        )
                        from utils.data_setup import create_labeled_dataloaders_from_paths
                        from visualization.extract_latent import plot_tsne
                    except Exception as e:
                        print(f"[WARN] missing dependencies for t-SNE ({type(e).__name__}: {e}); skipping")
                    else:
                        device = args.tsne_device or ("cuda" if torch.cuda.is_available() else "cpu")

                        if ae_name == "shallowConvAE":
                            ae = shallowConvAE().to(device)
                        elif ae_name == "convAE":
                            ae = ConvAutoencoder().to(device)
                        elif ae_name == "deepConvAE":
                            ae = DeepConvAutoencoder().to(device)
                        elif ae_name == "deepConvAEV1":
                            ae = DeepConvAutoencoderV1().to(device)
                        else:
                            print(f"[WARN] unknown AE type {ae_name}; skipping t-SNE")
                            ae = None

                        if ae is not None:
                            try:
                                ae.load_state_dict(torch.load(str(ckpt), map_location=device))
                            except Exception as e:
                                print(f"[WARN] failed loading AE checkpoint {ckpt}: {e}; skipping t-SNE")
                            else:
                                split_csv = Path(args.tsne_split_csv)
                                image_root = Path(args.tsne_image_root)
                                pairs = load_paths_from_split_csv(split_csv, image_root, args.tsne_sep)

                                if args.shuffle_samples:
                                    rng = random.Random(int(args.tsne_seed))
                                    rng.shuffle(pairs)

                                test_transform = transforms.Compose(
                                    [
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                    ]
                                )

                                dataloader, _dummy, _class_names, _weights = create_labeled_dataloaders_from_paths(
                                    train_paths=pairs,
                                    test_paths=[],
                                    train_transform=test_transform,
                                    test_transform=test_transform,
                                    batch_size=int(args.tsne_batch_size),
                                    num_workers=int(args.tsne_num_workers),
                                    return_class_weights=False,
                                    verbose=False,
                                )

                                out_tsne = save_dir / "tsne_plot.svg"
                                print(
                                    f"[INFO] t-SNE: model={model_dir.name} AE={ae_name} fold={ae_fold} ckpt={ckpt} [{ckpt_source}] -> {out_tsne}"
                                )
                                plot_tsne(
                                    ae,
                                    dataloader,
                                    device,
                                    num_samples=int(args.tsne_samples),
                                    perplexity=float(args.tsne_perplexity),
                                    n_iter=int(args.tsne_n_iter),
                                    seed=int(args.tsne_seed),
                                    save_path=str(out_tsne),
                                    show=False,
                                )
                                plt.close("all")

        # Plots across folds (saved by vis.py)
        # ONLY run these when we actually have fold data.
        if all_fold_data:
            # AE loss plot exists only for AE-based experiments (not FIXMATCH)
            if {"AE_train_loss", "AE_val_loss"}.issubset(set(all_fold_data[0].columns)):
                plot_AE_Loss_CV_all_folds(
                    all_fold_data,
                    metric_name="AE Loss",
                    ylabel="Reconstruction Loss",
                    save_path=str(save_dir),
                    filename="AE_Loss_CV",
                )
                plt.close("all")

            plot_metric_CV_all_folds(all_fold_data, metric_name="f1", ylabel="F1 Score (%)", save_path=str(save_dir), filename="f1_CV")
            plt.close("all")

            plot_metric_CV_all_folds(all_fold_data, metric_name="acc", ylabel="Accuracy (%)", save_path=str(save_dir), filename="acc_CV")
            plt.close("all")

            plot_metric_CV_all_folds(all_fold_data, metric_name="auroc", ylabel="AUROC (%)", save_path=str(save_dir), filename="auroc_CV")
            plt.close("all")

            plot_precision_recall_CV_all_folds(all_fold_data, save_path=str(save_dir), filename="precision_recall_CV")
            plt.close("all")

            plot_mca_metric_CV_all_folds_for_classes(all_fold_data, "f1_mca", "F1 MCA", "F1 MCA", class_names, save_dir=str(save_dir))
            plt.close("all")
            plot_mca_metric_CV_all_folds_for_classes(all_fold_data, "acc_mca", "ACCURACY MCA", "ACCURACY MCA", class_names, save_dir=str(save_dir))
            plt.close("all")
            plot_mca_metric_CV_all_folds_for_classes(all_fold_data, "auroc_mca", "AUROC MCA", "AUROC MCA", class_names, save_dir=str(save_dir))
            plt.close("all")

        # Optional: best fold extra plots
        if args.best_fold_extra:
            if not all_fold_data:
                print("[WARN] --best_fold_extra requested but no fold data is available; skipping")
                continue
            if args.best_fold_by == "val_f1":
                best_fold, best_epoch, best_val = best_fold_by_val_f1(all_fold_data)
                print(f"\n[INFO] Best fold by max val_f1: fold={best_fold}, epoch={best_epoch}, val_f1={best_val:.4f}")
            else:
                best_fold, best_val = best_fold_by_test_f1(test_data)
                best_epoch = None
                print(f"\n[INFO] Best fold by test_f1: fold={best_fold}, test_f1={best_val:.4f}")

            df_best = all_fold_data[best_fold]
            plot_metric_single_fold(df_best, "f1", "F1 Score (%)", save_dir, "f1_best_fold", plot_configs)
            plot_metric_single_fold(df_best, "acc", "Accuracy (%)", save_dir, "acc_best_fold", plot_configs)
            plot_metric_single_fold(df_best, "auroc", "AUROC (%)", save_dir, "auroc_best_fold", plot_configs)
            plot_precision_recall_single_fold(df_best, save_dir, "precision_recall_best_fold", plot_configs)
            plot_mca_metric_single_fold_for_classes(df_best, "f1_mca", "F1 MCA", class_names, save_dir, plot_configs)

    print(f"\nDone. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
