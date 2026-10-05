"""

The third experiment involves training the autoencoder and
classifier networks separately (similar to the first experiment).
With the exception that in this experiment, we employed
synthetic sample generation for the minority class.
Also, in this setup we have included 5-fold cross validation.

"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from colorama import Fore, Style

# Options (CycleGAN style)
from options.train_options import TrainOptions

# Custom modules
from utils.data_setup import CustomImageDatasetV1, create_labeled_dataloaders_from_paths
from engine.utils import create_output_dir
from engine.models.autoencoder import shallowConvAE, ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1
from engine.models.classifier import Classifier256, Classifier512, Classifier1024, Classifier2048
from engine.engine_exp_3 import train
from utils.fold_handle import get_fold_train_val_test_data, add_external_imgs_train_only, resolve_split_csv

from torch.utils.data import DataLoader


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
        transforms.RandomPerspective(),
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


def make_models(AE_name: str, num_classes: int, device: torch.device):
    if AE_name == "shallowConvAE":
        AE = shallowConvAE().to(device)
        classifier = Classifier256(256, num_classes, dropout_rate=0.4).to(device)

    elif AE_name == "convAE":
        AE = ConvAutoencoder().to(device)
        classifier = Classifier512(512, num_classes, dropout_rate=0.4).to(device)

    elif AE_name == "deepConvAE":
        AE = DeepConvAutoencoder().to(device)
        classifier = Classifier1024(1024, num_classes, dropout_rate=0.4).to(device)

    elif AE_name == "deepConvAEV1":
        AE = DeepConvAutoencoderV1().to(device)
        classifier = Classifier2048(2048, num_classes, dropout_rate=0.4).to(device)

    else:
        raise ValueError(f"Unknown AE_name: {AE_name}")

    return AE, classifier


def load_pretrained_ae(AE: torch.nn.Module, ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained AE checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    AE.load_state_dict(state)
    return AE


def main():
    # ---- parse options ----
    opt = TrainOptions().parse()
    if opt.name is None:
        opt.name = "EXP_3"

    # ---- colors ----
    c_ = Fore.GREEN
    i_ = Fore.BLUE
    b_ = Fore.MAGENTA
    sr_ = Style.RESET_ALL

    # ---- seed/device ----
    seed_everything(opt.seed)
    device = torch.device(opt.device)

    # ---- paths ----
    data_path = Path(opt.data_path)
    l_image_path = data_path / opt.labeled_dir
    split_csv_path = resolve_split_csv(opt.split_csv, l_image_path)
    extra_img_dir = (l_image_path / opt.extra_img_dir) if opt.extra_img_dir else None

    # ---- transforms ----
    train_transform, test_transform = build_transforms()

    experiment_number = 0

    # ---- loop through AEs ----
    for AE_name in opt.aes:

        fold_results = {
            "test_loss": [],
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

            # ---- labeled loaders ----
            l_train_dataloader, l_val_dataloader, class_names, class_weights = create_labeled_dataloaders_from_paths(
                train_paths=train_paths,
                test_paths=val_paths,
                train_transform=train_transform,
                test_transform=test_transform,
                batch_size=opt.l_batch_size,
                num_workers=opt.num_workers,
            )

            # Build TEST loader with the same class_to_idx as TRAIN for consistent label indices.
            class_to_idx = {c: i for i, c in enumerate(class_names)}
            test_dataset = CustomImageDatasetV1(
                image_label_pairs=test_paths,
                transform=test_transform,
                class_to_idx=class_to_idx,
                strict_labels=True,
            )
            l_test_dataloader = DataLoader(
                test_dataset,
                batch_size=opt.l_batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=True,
            )

            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

            # ---- info ----
            experiment_number += 1
            print(f"{b_}[INFO] Experiment number: {experiment_number}{sr_}")
            print(f"{i_}[INFO] Labelled train dl length: {len(l_train_dataloader)}{sr_}")
            print(f"[INFO] Labelled val dl length: {len(l_val_dataloader)}")
            print(f"[INFO] Labelled test dl length: {len(l_test_dataloader)}")
            print(f"[INFO] Class weights: {weights_tensor}")
            print(f"{i_}[INFO] Class names: {class_names}{sr_}")
            print("-" * 50 + "\n")

            # ---- models ----
            AE, classifier = make_models(AE_name, len(class_names), device)

            # Load pretrained AE (trained separately in ONLY_AE)
            pretrained_ckpt = (
                Path(opt.checkpoints_dir)
                / "ONLY_AE"
                / AE_name
                / "model_weights"
                / "AE_100_epochs.pth"
            )
            AE = load_pretrained_ae(AE, pretrained_ckpt, device)

            print(f"[INFO] Model: {opt.name}")
            print(f"[INFO] Exact model: {opt.classifier_name}_{AE_name}")
            print(f"[INFO] Epochs: {opt.epochs}")
            print(f"[INFO] CL LR: {opt.cl_lr}")
            print(sr_)

            # ---- outputs (fold-aware so you don't overwrite) ----
            base_out = (
                Path(opt.checkpoints_dir)
                / opt.name
                / f"{opt.classifier_name}_{AE_name}"
                / f"{opt.extra_img_n}"
                / f"fold_{fold}"
            )
            _, model_weights_dir = create_output_dir(base_out)

            # ---- losses/optim/sched ----
            classification_criterion = nn.CrossEntropyLoss()

            optimizer_B = torch.optim.Adam(
                classifier.parameters(),
                lr=opt.cl_lr,
                weight_decay=opt.weight_decay,
            )
            scheduler_B = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_B,
                mode="max",
                factor=0.5,
                patience=10,
                threshold_mode="rel",
                cooldown=1,
                min_lr=1e-6,
            )

            # ---- train ----
            train(
                AE=AE,
                classifier=classifier,
                train_dataloader_B=l_train_dataloader,
                test_dataloader_B=l_test_dataloader,
                val_dataloader_B=l_val_dataloader,
                optimizer_B=optimizer_B,
                loss_fn_B=classification_criterion,
                epochs=opt.epochs,
                device=device,
                scheduler_B=scheduler_B,
                model_weights_dir=model_weights_dir,
                fold_results=fold_results,
                fold=fold,
                num_classes=len(class_names),
                CL_lr=opt.cl_lr,
                patience=opt.patience,
                save_models=opt.save_models,
            )

            print(f"{c_}Experiment #{experiment_number} completed successfully.{sr_}")
            print("-" * 50 + "\n")

        # ---- fold averages ----
        print(f"Average performance across folds for {opt.classifier_name}_{AE_name}:")
        for metric, values in fold_results.items():
            mean = np.mean(values) if len(values) else float("nan")
            std = np.std(values) if len(values) else float("nan")
            if isinstance(mean, np.ndarray) or isinstance(std, np.ndarray):
                mean_s = np.array2string(np.asarray(mean), precision=4, separator=", ")
                std_s = np.array2string(np.asarray(std), precision=4, separator=", ")
                print(f"{metric}: {mean_s} ± {std_s}")
            else:
                print(f"{metric}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
