#!/usr/bin/env python3
"""Paired t-test between two cross-validated runs (paper Sec. 5.9.3).

Reads the per-fold test CSVs (<fold>_test.csv) written by the training
scripts and compares macro F1 and one class's F1 (Ileum by default) fold by
fold with scipy.stats.ttest_rel.

Example (SSR-ALC Exp. 4 vs FixMatch):
    python analysis/paired_ttest.py \
        --run_a checkpoints/EXP_4/0/simpleNN_convAE \
        --run_b checkpoints/FIXMATCH/fixmatch_resnet18/0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_a", required=True, help="Run directory of method A (searched recursively for *_test.csv)")
    p.add_argument("--run_b", required=True, help="Run directory of method B")
    p.add_argument("--class_index", type=int, default=1, help="Per-class F1 to test (1 = ileum)")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return p.parse_args()


def _parse_array(cell) -> np.ndarray:
    return np.array([float(x) for x in re.split(r"[,\s]+", str(cell).strip("[]() tensor")) if x])


def load_fold_scores(run_dir: Path, folds, class_index: int):
    macro, per_class = {}, {}
    for p in sorted(run_dir.rglob("*_test.csv")):
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            fold = int(row["Fold"])
            if fold in folds:
                macro[fold] = float(row["test_f1"])
                per_class[fold] = float(_parse_array(row["test_f1_mca"])[class_index])
    missing = [f for f in folds if f not in macro]
    if missing:
        raise SystemExit(f"{run_dir}: missing test results for folds {missing}")
    return np.array([macro[f] for f in folds]), np.array([per_class[f] for f in folds])


def report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    t, p = stats.ttest_rel(a, b)
    print(f"{name}: A = {100 * a.mean():.2f} ± {100 * a.std(ddof=1):.2f} | "
          f"B = {100 * b.mean():.2f} ± {100 * b.std(ddof=1):.2f} | t = {t:.2f}, p = {p:.4f}")


def main() -> None:
    args = parse_args()
    a_macro, a_cls = load_fold_scores(Path(args.run_a), args.folds, args.class_index)
    b_macro, b_cls = load_fold_scores(Path(args.run_b), args.folds, args.class_index)
    report("Macro F1", a_macro, b_macro)
    report(f"Class {args.class_index} F1", a_cls, b_cls)


if __name__ == "__main__":
    main()
