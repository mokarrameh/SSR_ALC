"""
Cross-dataset generalization on Kvasir v2 (paper Sec. 4.18 / Table 13).

The autoencoder is initialized from the best-fold Experiment 4 (SSR-ALC)
HyperKvasir weights and fine-tuned on labeled Kvasir images with the
objective of Eq. (4): labeled reconstruction + classification (no unlabeled
term, no LATE-SMOTE). 10-fold cross-validation.

Restored from the original `train_kvasir_AE_transfer.py` and ported to the
YAML/CLI options. Training logic (engine/engine_kvasir_no_un.py) is unchanged.

Example:
    python train_kvasir.py --config configs/kvasir.yaml --data_path /path/to/kvasir
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, Style
from torchvision import transforms

from options.train_options import TrainOptions
from utils.config_utils import load_yaml_config
from utils.data_setup import create_labeled_dataloaders_from_paths
from utils.fold_handle import get_fold_train_val_test_data, resolve_split_csv
from engine.utils import create_output_dir
from engine.models.autoencoder import ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1
from engine.models.classifier import Classifier512_kv, Classifier1024_kv, Classifier2048_kv
from engine.engine_kvasir_no_un import train


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(280),
        transforms.RandomCrop(256),
        transforms.RandomAffine(degrees=45, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return train_transform, test_transform


MODELS = {
    "convAE": (ConvAutoencoder, Classifier512_kv, 512),
    "deepConvAE": (DeepConvAutoencoder, Classifier1024_kv, 1024),
    "deepConvAEV1": (DeepConvAutoencoderV1, Classifier2048_kv, 2048),
}


def main():
    opt = TrainOptions().parse()
    seed_everything(opt.seed)
    device = torch.device(opt.device)

    # Pretrained HyperKvasir AE checkpoint per AE (train.pretrained_ae in the YAML).
    yaml_cfg = load_yaml_config(opt.config) if opt.config else {}
    pretrained = (yaml_cfg.get("train") or {}).get("pretrained_ae") or {}

    l_image_path = Path(opt.data_path) / opt.labeled_dir
    split_csv_path = resolve_split_csv(opt.split_csv, l_image_path)
    train_transform, test_transform = build_transforms()

    for AE_name in opt.aes:
        if AE_name not in MODELS:
            raise ValueError(f"Unknown AE name: {AE_name}. Choose from {list(MODELS)}")
        ae_cls, clf_cls, emb_dim = MODELS[AE_name]
        ckpt = pretrained.get(AE_name)
        if not ckpt:
            raise ValueError(f"No pretrained AE checkpoint configured for {AE_name} (train.pretrained_ae.{AE_name}).")

        fold_results = {k: [] for k in [
            "AE_test_loss", "test_loss", "test_acc", "test_acc_mca", "test_f1", "test_f1_mca",
            "test_auroc", "test_auroc_mca", "test_precision", "test_recall"]}

        for fold in opt.folds:
            print(f"\n{Fore.CYAN}[INFO] Fold {fold}{Style.RESET_ALL}")
            train_paths, val_paths, test_paths = get_fold_train_val_test_data(
                split_csv_path, l_image_path, fold_idx=fold, val_ratio=0.2, seed=opt.seed)

            l_train_dataloader, l_val_dataloader, class_names, _ = create_labeled_dataloaders_from_paths(
                train_paths=train_paths, test_paths=val_paths,
                train_transform=train_transform, test_transform=test_transform,
                batch_size=opt.l_batch_size, num_workers=opt.num_workers)
            l_test_dataloader, _, _, _ = create_labeled_dataloaders_from_paths(
                train_paths=test_paths, test_paths=None,
                train_transform=test_transform, test_transform=test_transform,
                batch_size=opt.l_batch_size, num_workers=opt.num_workers, verbose=False)

            AE = ae_cls().to(device)
            AE.load_state_dict(torch.load(ckpt, map_location=device))
            classifier = clf_cls(emb_dim, len(class_names), dropout_rate=0).to(device)

            print(f"[INFO] {opt.name}: simpleNN_{AE_name} | init from {ckpt}")
            training_progress_dir, model_weights_dir = create_output_dir(
                Path(opt.checkpoints_dir) / opt.name / f"simple_NN_{AE_name}")

            optimizer_A = torch.optim.Adam(AE.parameters(), lr=opt.ae_lr, weight_decay=opt.weight_decay)
            optimizer_B = torch.optim.Adam(classifier.parameters(), lr=opt.cl_lr)  # no weight decay (as original)
            scheduler_A = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_A, mode="min", factor=0.5, patience=10, threshold_mode="rel", cooldown=1, min_lr=1e-6)
            scheduler_B = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_B, mode="max", factor=0.5, patience=10, threshold_mode="rel", cooldown=1, min_lr=1e-6)

            train(
                AE=AE,
                classifier=classifier,
                train_dataloader=l_train_dataloader,
                val_dataloader=l_val_dataloader,
                test_dataloader=l_test_dataloader,
                optimizer_A=optimizer_A,
                optimizer_B=optimizer_B,
                loss_fn_A=nn.MSELoss(),
                loss_fn_B=nn.CrossEntropyLoss(),
                loss_fn_C=nn.CrossEntropyLoss(),
                epochs=opt.epochs,
                device=device,
                scheduler_A=scheduler_A,
                scheduler_B=scheduler_B,
                training_progress_dir=training_progress_dir,
                model_weights_dir=model_weights_dir,
                fold_results=fold_results,
                fold=fold,
                num_classes=len(class_names),
                alpha=opt.alpha,
                beta=opt.beta,
                gamma=opt.gamma,
                AE_lr=opt.ae_lr,
                CL_lr=opt.cl_lr,
                patience=opt.patience,
            )

        print(f"Average performance across folds for simpleNN_{AE_name}:")
        for metric, values in fold_results.items():
            if "mca" in metric or not values:
                continue
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values, ddof=1) if len(values) > 1 else 0.0:.4f}")


if __name__ == "__main__":
    main()
