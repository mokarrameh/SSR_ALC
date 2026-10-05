#!/usr/bin/env python3
"""Summarize the Ileum sensitivity analysis (paper Sec. 5.9.1, Fig. 9).

For each N (number of extra train-only Ileum images) and each fold, picks the
epoch with the best macro validation F1 and reads the Ileum validation F1 at
that epoch. Reports mean ± std (ddof=0) across folds and plots both curves.

This is the logic of the original `ileum.ipynb` notebook as a script.

Expected layout (written by exp4_train.py with configs/ileum_sensitivity.yaml):
    <run_dir>/<N>/<model>/fold_<k>/model_weights/<k>.csv

Example:
    python analysis/ileum_sensitivity_summary.py --run_dir checkpoints/ileum_sensitivity
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, required=True, help="e.g. checkpoints/ileum_sensitivity")
    p.add_argument("--model", type=str, default="simpleNN_deepConvAE")
    p.add_argument("--n_values", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50, 56])
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ileum_index", type=int, default=1,
                   help="Class index of Ileum (classes are sorted alphabetically: cecum, ileum, ...)")
    p.add_argument("--out_dir", type=str, default=None, help="Defaults to --run_dir")
    return p.parse_args()


def parse_metric_array(value):
    """Parse a per-class metric cell such as '[0.91 0.83 0.77]' or 'tensor([...])'."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = re.sub(r"^tensor\(", "", s)
    if s.endswith(")"):
        s = s[:-1]
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [float(x) for x in parsed]
    except Exception:
        pass
    parts = [x for x in re.split(r"[,\s]+", s.strip("[]")) if x]
    try:
        return [float(x) for x in parts]
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for N in args.n_values:
        for fold in args.folds:
            csv_path = run_dir / str(N) / args.model / f"fold_{fold}" / "model_weights" / f"{fold}.csv"
            if not csv_path.exists():
                print(f"[WARNING] Missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            best = df.loc[df["val_f1"].astype(float).idxmax()]
            mca = parse_metric_array(best["val_f1_mca"])
            ileum = float(mca[args.ileum_index]) if mca and args.ileum_index < len(mca) else np.nan
            rows.append({"N": N, "fold": fold, "best_epoch": int(best["Epoch"]),
                         "best_val_f1": float(best["val_f1"]), "ileum_val_f1_at_best_epoch": ileum})

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        raise SystemExit("No results found.")
    summary = (detailed.groupby("N")
               .agg(macro_val_f1_mean=("best_val_f1", "mean"),
                    macro_val_f1_std=("best_val_f1", lambda x: x.std(ddof=0)),
                    ileum_val_f1_mean=("ileum_val_f1_at_best_epoch", "mean"),
                    ileum_val_f1_std=("ileum_val_f1_at_best_epoch", lambda x: x.std(ddof=0)),
                    n_folds=("fold", "count"))
               .reset_index())

    print("=== Per-fold best validation results ===")
    print(detailed.to_string(index=False))
    print("\n=== Summary across folds ===")
    print(summary.to_string(index=False))

    detailed.to_csv(out_dir / "ileum_sensitivity_best_val_per_fold.csv", index=False)
    summary.to_csv(out_dir / "ileum_sensitivity_summary_by_N.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    x = summary["N"].values
    plt.plot(x, summary["macro_val_f1_mean"].values, marker="o", linewidth=2, label="Macro validation F1")
    plt.plot(x, summary["ileum_val_f1_mean"].values, marker="s", linewidth=2, label="Ileum validation F1")
    plt.xlabel("Number of additional Ileum samples (N)")
    plt.ylabel("F1-score")
    plt.title("Sensitivity analysis with validation-based selection")
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "ileum_sensitivity_validation_plot.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
