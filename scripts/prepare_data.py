#!/usr/bin/env python3
"""Build the dataset layout expected by the training scripts.

    <out>/labeled/<class>/<image>            6 anatomical-landmark classes (HyperKvasir)
    <out>/labeled/ileum/<n>.png              the 56 GastroLab Ileum images (expanded split)
    <out>/labeled/extra_ileum/<n>.png        same 56 images, used as train-only extras
                                             (Ileum sensitivity analysis and HPO only)
    <out>/unlabeled/{train,val,test}/<image> fixed unlabeled subset (22,500 / 1,400 / 1,800)
    <kvasir_out>/labeled/<class>/<image>     Kvasir v2 (optional, Table 13)

Images are looked up by file name, so the HyperKvasir folders can be in their
original download structure (labeled-images/.../anatomical-landmarks/<class>/,
unlabeled-images/images/). Only the images listed in splits/ are used.

Example:
    python scripts/prepare_data.py \
        --hyperkvasir_labeled /data/hyper-kvasir/labeled-images \
        --hyperkvasir_unlabeled /data/hyper-kvasir/unlabeled-images \
        --gastrolab_ileum /data/gastrolab_ileum \
        --out dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "splits"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def index_images(root: Path) -> dict[str, list[Path]]:
    """Map file name -> paths for every image below root."""
    index: dict[str, list[Path]] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                index.setdefault(f, []).append(Path(dirpath) / f)
    return index


def lookup(index: dict[str, list[Path]], fname: str, sub: str | None = None) -> Path | None:
    """Find an image by name; if the name occurs in several folders, pick the one named like `sub`."""
    paths = index.get(fname, [])
    if len(paths) == 1:
        return paths[0]
    matches = [q for q in paths if sub and q.parent.name.endswith(str(sub))]
    return matches[0] if len(matches) == 1 else None


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def build(split_csv: Path, index: dict[str, Path], out_dir: Path, mode: str,
          subdir_col: str = "class-name", skip: set[str] | None = None) -> int:
    df = pd.read_csv(split_csv, sep=";")
    missing, n = [], 0
    for fname, sub in zip(df["file-name"], df[subdir_col]):
        if skip and fname in skip:
            continue
        src = lookup(index, fname, sub)
        if src is None:
            missing.append(fname)
            continue
        place(src, out_dir / str(sub) / fname, mode)
        n += 1
    if missing:
        raise SystemExit(f"{len(missing)} images from {split_csv.name} not found (e.g. {missing[:3]}).")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hyperkvasir_labeled", type=Path, help="HyperKvasir labeled-images folder")
    p.add_argument("--hyperkvasir_unlabeled", type=Path, help="HyperKvasir unlabeled-images folder")
    p.add_argument("--gastrolab_ileum", type=Path, help="Folder with the 56 GastroLab Ileum images (0.png ... 55.png)")
    p.add_argument("--kvasir", type=Path, help="Kvasir v2 folder (class sub-folders)")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--kvasir_out", type=Path, default=Path("dataset/kvasir"))
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    args = p.parse_args()

    gastro = set(pd.read_csv(SPLITS / "gastrolab_ileum_extra.csv", sep=";")["file-name"])

    if args.hyperkvasir_labeled:
        idx = index_images(args.hyperkvasir_labeled)
        n = build(SPLITS / "hyperkvasir_5fold_expanded.csv", idx, args.out / "labeled", args.mode, skip=gastro)
        print(f"[labeled] {n} HyperKvasir images -> {args.out / 'labeled'}")

    if args.gastrolab_ileum:
        idx = index_images(args.gastrolab_ileum)
        missing = sorted(gastro - set(idx))
        if missing:
            raise SystemExit(f"{len(missing)} GastroLab images missing (e.g. {missing[:3]}).")
        for f in sorted(gastro):
            place(idx[f][0], args.out / "labeled" / "ileum" / f, args.mode)
            place(idx[f][0], args.out / "labeled" / "extra_ileum" / f, args.mode)
        print(f"[gastrolab] {len(gastro)} Ileum images -> labeled/ileum and labeled/extra_ileum")
    else:
        print("[gastrolab] skipped: without --gastrolab_ileum only the original 9-Ileum split can be used")

    if args.hyperkvasir_unlabeled:
        idx = index_images(args.hyperkvasir_unlabeled)
        n = build(SPLITS / "hyperkvasir_unlabeled_subset.csv", idx, args.out / "unlabeled", args.mode,
                  subdir_col="subset")
        print(f"[unlabeled] {n} images -> {args.out / 'unlabeled'}/{{train,val,test}}")

    if args.kvasir:
        idx = index_images(args.kvasir)
        n = build(SPLITS / "kvasir_v2_10fold.csv", idx, args.kvasir_out / "labeled", args.mode)
        print(f"[kvasir] {n} images -> {args.kvasir_out / 'labeled'}")


if __name__ == "__main__":
    main()
