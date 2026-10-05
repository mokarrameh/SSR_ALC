#!/usr/bin/env python3
"""SSIM between test images and their autoencoder reconstructions (paper Table 11).

For each fold, the fold's labeled test images are passed through the
autoencoder and the mean SSIM over all test images is computed. The script
then reports mean ± std of the per-fold means.

Uses `compute_ssim_from_model` from visualization/cal_ssim.py (original code).

Examples:
    # jointly trained models (one AE per fold)
    python analysis/ssim_eval.py --ae convAE \
        --ckpt "checkpoints/EXP_4/0/simpleNN_convAE/fold_{fold}/model_weights/AE_100_epochs_{fold}fold.pth" \
        --split_csv splits/hyperkvasir_5fold_expanded.csv --image_root dataset/labeled

    # Experiments 1 and 3 (one unsupervised AE evaluated on every fold's test set)
    python analysis/ssim_eval.py --ae convAE \
        --ckpt checkpoints/ONLY_AE/convAE/model_weights/AE_100_epochs.pth \
        --split_csv splits/hyperkvasir_5fold_expanded.csv --image_root dataset/labeled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

from engine.models.autoencoder import ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1  # noqa: E402
from utils.data_setup import CustomImageDatasetV1  # noqa: E402
from utils.fold_handle import get_fold_train_val_test_data  # noqa: E402
from visualization.cal_ssim import compute_ssim_from_model  # noqa: E402

AES = {"convAE": ConvAutoencoder, "deepConvAE": DeepConvAutoencoder, "deepConvAEV1": DeepConvAutoencoderV1}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ae", choices=list(AES), required=True)
    p.add_argument("--ckpt", required=True, help="AE checkpoint; may contain '{fold}'")
    p.add_argument("--split_csv", required=True)
    p.add_argument("--image_root", required=True, help="Folder with <class-name>/<file-name> images")
    p.add_argument("--exclude_labels", nargs="*", default=None, help="e.g. ileum (Experiment 5)")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    fold_means = []
    for fold in args.folds:
        _, _, test_paths = get_fold_train_val_test_data(
            args.split_csv, args.image_root, fold_idx=fold, exclude_labels=args.exclude_labels)
        loader = DataLoader(CustomImageDatasetV1(test_paths, tf), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        ae = AES[args.ae]().to(args.device)
        ae.load_state_dict(torch.load(args.ckpt.format(fold=fold), map_location=args.device))
        scores = compute_ssim_from_model(ae, loader, args.device)
        fold_means.append(float(np.mean(scores)))
        print(f"fold {fold}: mean SSIM = {fold_means[-1]:.4f} over {len(scores)} test images")

    m = np.array(fold_means)
    print(f"SSIM across folds: {m.mean():.4f} ± {m.std():.4f}")


if __name__ == "__main__":
    main()
