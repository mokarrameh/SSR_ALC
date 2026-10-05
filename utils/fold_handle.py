from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


PathPair = Tuple[str, str]  # (filepath, class_name)


def resolve_split_csv(split_csv: Union[str, Path], labeled_root: Union[str, Path]) -> Path:
    """Locate a split CSV.

    Accepts an absolute path, a path relative to the current working directory
    (e.g. ``splits/hyperkvasir_5fold_expanded.csv``), or a bare file name inside
    ``<data_path>/<labeled_dir>`` (the layout used during the original experiments).
    """
    p = Path(split_csv)
    if p.is_absolute() or p.exists():
        return p
    return Path(labeled_root) / p


def _file_digest(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _assert_no_content_overlap(extra: Sequence[PathPair], split_name: str, split_paths: Sequence[PathPair]) -> None:
    """Fail if any external image is byte-identical to an image already in a split.

    Path comparison alone cannot catch the same image stored under two names
    (e.g. ``ileum/0.png`` and ``extra_ileum/0.png``), so compare file contents.
    Only files whose size matches an external image are hashed.
    """
    extra_sizes = {os.path.getsize(p) for p, _ in extra}
    candidates = [p for p, _ in split_paths if os.path.exists(p) and os.path.getsize(p) in extra_sizes]
    if not candidates:
        return
    extra_digests = {_file_digest(p) for p, _ in extra}
    clashes = [p for p in candidates if _file_digest(p) in extra_digests]
    if clashes:
        raise AssertionError(
            f"Leakage: {len(clashes)} external image(s) are byte-identical to images already in the "
            f"{split_name} split (e.g. {clashes[0]}). Use the original split with extra_img_n > 0, "
            "or the expanded split with extra_img_n = 0, not both."
        )


def _read_split_csv(split_csv_path: Union[str, Path], sep: str = ";") -> pd.DataFrame:
    p = Path(split_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Split CSV not found: {p}")
    df = pd.read_csv(p, sep=sep)
    return df


def _require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        where = f" ({ctx})" if ctx else ""
        raise ValueError(f"Missing required columns{where}: {missing}. Found: {list(df.columns)}")


def _stratified_train_val_split(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    # X can be any array-like; use index to keep simple
    train_idx, val_idx = next(sss.split(df.index, df[label_col]))
    return df.iloc[train_idx], df.iloc[val_idx]


def _to_path_list(df_subset: pd.DataFrame, path_fn: Callable[[pd.Series], Path], label_col: str) -> List[PathPair]:
    out: List[PathPair] = []
    for _, row in df_subset.iterrows():
        p = path_fn(row)
        out.append((str(p), str(row[label_col])))
    return out


def get_fold_train_val_test_data(
    split_csv_path: Union[str, Path],
    image_root: Union[str, Path],
    fold_idx: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    sep: str = ";",
    exclude_labels: Optional[Sequence[str]] = None,
) -> Tuple[List[PathPair], List[PathPair], List[PathPair]]:
    """
      - Hold out fold_idx as test
      - Stratified split remaining into train/val
      - Build paths as image_root/class-name/file-name
    """

    split_csv_path = Path(split_csv_path)
    image_root = Path(image_root)

    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path, sep=sep)

    if exclude_labels:
        exclude_set = {str(x) for x in exclude_labels}
        df = df[~df["class-name"].astype(str).isin(exclude_set)].reset_index(drop=True)

    required_cols = ["split-index", "class-name", "file-name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in split CSV.")

    # ---- Step 1: Hold out test fold ----
    test_df = df[df["split-index"] == fold_idx]
    non_test_df = df[df["split-index"] != fold_idx]

    if len(test_df) == 0:
        raise ValueError(f"No samples found for fold_idx={fold_idx}.")

    if len(non_test_df) == 0:
        raise ValueError("All samples belong to test fold. Invalid split configuration.")

    # ---- Step 2: Stratified train/val split ----
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio must be between 0 and 1. Got {val_ratio}")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )

    train_idx, val_idx = next(
        sss.split(non_test_df["file-name"], non_test_df["class-name"])
    )

    train_df = non_test_df.iloc[train_idx]
    val_df = non_test_df.iloc[val_idx]

    # ---- Build path list ----
    def to_path_list(df_subset):
        return [
            (str(image_root / row["class-name"] / row["file-name"]), row["class-name"])
            for _, row in df_subset.iterrows()
        ]

    return (
        to_path_list(train_df),
        to_path_list(val_df),
        to_path_list(test_df),
    )


def get_fold_train_val_test_data_anomaly(
    split_csv_path: Union[str, Path],
    image_root: Union[str, Path],
    fold_idx: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    sep: str = ";",
) -> Tuple[List[PathPair], List[PathPair], List[PathPair]]:
    """
    Anomaly case: image_root/Organ/classification/finding/file-name
    Label uses class-name column (same as your original).
    """
    image_root = Path(image_root)
    df = _read_split_csv(split_csv_path, sep=sep)

    _require_columns(
        df,
        ["split-index", "class-name", "file-name", "Organ", "classification", "finding"],
        ctx="get_fold_train_val_test_data_anomaly",
    )

    test_df = df[df["split-index"] == fold_idx]
    non_test_df = df[df["split-index"] != fold_idx]
    if len(test_df) == 0:
        raise ValueError(f"No samples found for fold_idx={fold_idx} in split-index column.")

    train_df, val_df = _stratified_train_val_split(non_test_df, label_col="class-name", val_ratio=val_ratio, seed=seed)

    def path_fn(row: pd.Series) -> Path:
        return image_root / row["Organ"] / row["classification"] / row["finding"] / row["file-name"]

    return (
        _to_path_list(train_df, path_fn, label_col="class-name"),
        _to_path_list(val_df, path_fn, label_col="class-name"),
        _to_path_list(test_df, path_fn, label_col="class-name"),
    )


def get_fold_train_val_test_data_abnormal(
    split_csv_path: Union[str, Path],
    image_root: Union[str, Path],
    fold_idx: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    sep: str = ";",
) -> Tuple[List[PathPair], List[PathPair], List[PathPair]]:
    """
    Abnormal case: image_root/Organ/classification/class-name/file-name
    """
    image_root = Path(image_root)
    df = _read_split_csv(split_csv_path, sep=sep)

    _require_columns(
        df,
        ["split-index", "class-name", "file-name", "Organ", "classification"],
        ctx="get_fold_train_val_test_data_abnormal",
    )

    test_df = df[df["split-index"] == fold_idx]
    non_test_df = df[df["split-index"] != fold_idx]
    if len(test_df) == 0:
        raise ValueError(f"No samples found for fold_idx={fold_idx} in split-index column.")

    train_df, val_df = _stratified_train_val_split(non_test_df, label_col="class-name", val_ratio=val_ratio, seed=seed)

    def path_fn(row: pd.Series) -> Path:
        return image_root / row["Organ"] / row["classification"] / row["class-name"] / row["file-name"]

    return (
        _to_path_list(train_df, path_fn, label_col="class-name"),
        _to_path_list(val_df, path_fn, label_col="class-name"),
        _to_path_list(test_df, path_fn, label_col="class-name"),
    )

def load_external_img_paths(extra_dir: Union[str, Path], label: str) -> List[PathPair]:
    extra_dir = Path(extra_dir)
    if not extra_dir.exists():
        raise FileNotFoundError(f"extra_dir does not exist: {extra_dir}")

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    files: List[str] = []
    for e in exts:
        files.extend(glob(str(extra_dir / e)))
    files = sorted(set(files))
    return [(str(Path(f).resolve()), label) for f in files]

def add_external_imgs_train_only(
    train_paths: Sequence[PathPair],
    val_paths: Sequence[PathPair],
    test_paths: Sequence[PathPair],
    extra_dir: Union[str, Path],
    n_extra: int,
    seed: int,
    label: str = "ileum",
) -> Tuple[List[PathPair], List[PathPair]]:
    """
    Adds external images ONLY to train_paths.
    Returns (train_paths_new, added_subset).

    Does NOT oversample — it simply appends n_extra samples.
    """
    if n_extra <= 0:
        return list(train_paths), []

    extra_all = load_external_img_paths(extra_dir, label)
    if len(extra_all) == 0:
        raise RuntimeError(f"No images found in extra dir: {extra_dir}")

    if n_extra > len(extra_all):
        raise RuntimeError(f"Requested n_extra={n_extra}, but only {len(extra_all)} external images exist in {extra_dir}.")

    rng = random.Random(seed)
    extra_subset = rng.sample(extra_all, n_extra)

    # Normalize to absolute resolved paths for robust leakage checks
    def norm(paths: Sequence[PathPair]) -> set:
        return {str(Path(p).resolve()) for p, _ in paths}

    train_set = norm(train_paths)
    val_set = norm(val_paths)
    test_set = norm(test_paths)
    extra_set = norm(extra_subset)

    if extra_set & val_set:
        raise AssertionError("Leakage: external samples overlap with VAL set!")
    if extra_set & test_set:
        raise AssertionError("Leakage: external samples overlap with TEST set!")
    if extra_set & train_set:
        raise AssertionError("Duplicate: some external samples already exist in TRAIN set!")

    _assert_no_content_overlap(extra_subset, "TRAIN", train_paths)
    _assert_no_content_overlap(extra_subset, "VAL", val_paths)
    _assert_no_content_overlap(extra_subset, "TEST", test_paths)

    train_paths_new = list(train_paths) + extra_subset
    return train_paths_new, extra_subset