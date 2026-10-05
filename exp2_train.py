"""

The second experiment involves training the autoencoder
and classifier in a loop, providing the autoencoder with
classification feedback. This experiment does not include
synthetic sample generation for the minority class.

"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from colorama import Fore, Style

# Options (CycleGAN style)
from options.train_options import TrainOptions

# Custom modules
from utils.data_setup import create_unlabeled_dataloaders, create_labeled_dataloaders_from_paths
from engine.utils import create_output_dir, weights_init
from engine.models.autoencoder import shallowConvAE, ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1
from engine.models.classifier import Classifier256, Classifier512, Classifier1024, Classifier2048
from engine.engine_exp_2 import train
from utils.fold_handle import get_fold_train_val_test_data, add_external_imgs_train_only, resolve_split_csv

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
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
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

    AE.apply(weights_init)
    return AE, classifier


def main():
    # ---- parse options ----
    opt = TrainOptions().parse()

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
    u_image_path = data_path / opt.unlabeled_dir
    split_csv_path = resolve_split_csv(opt.split_csv, l_image_path)
    extra_img_dir = (l_image_path / opt.extra_img_dir) if opt.extra_img_dir else None

    u_train_path = u_image_path / opt.u_train_subdir
    u_val_path   = u_image_path / opt.u_val_subdir
    u_test_path  = u_image_path / opt.u_test_subdir

    # ---- transforms ----
    train_transform, test_transform = build_transforms()

    # ---- unlabeled dataloaders ----
    u_train_dataloader, u_val_dataloader, u_test_dataloader = create_unlabeled_dataloaders(
        train_dir=u_train_path,
        val_dir=u_val_path,
        test_dir=u_test_path,
        train_transform=train_transform,
        test_transform=test_transform,
        train_batch_size=opt.u_train_batch_size,
        val_batch_size=opt.u_val_batch_size,
        test_batch_size=opt.u_test_batch_size,
        num_workers=opt.num_workers
    )

    experiment_number = 0

    # Loop through each number of epochs
    for AE_name in opt.aes:

        fold_results = {
            "AE_test_loss": [],
            "test_loss": [],
            "test_acc": [],
            "test_acc_mca": [],
            "test_f1": [],
            "test_f1_mca": [],
            "test_auroc": [],
            "test_auroc_mca": [],
            "test_precision": [],
            "test_recall": []
        }
        
        for fold in opt.folds:
            print(f"\n{Fore.CYAN}[INFO] Fold {fold}{Style.RESET_ALL}")

            train_paths, val_paths, test_paths = get_fold_train_val_test_data(
                split_csv_path,
                l_image_path,
                fold_idx=fold,
                val_ratio=0.2,
                seed=opt.seed
            )

            # ---- add external ileum to TRAIN only ----
            added_subset = []
            if extra_img_dir is not None and opt.extra_img_n > 0:
                train_paths, added_subset = add_external_imgs_train_only(
                    train_paths, val_paths, test_paths,
                    extra_dir=extra_img_dir,
                    n_extra=opt.extra_img_n,
                    seed=opt.extra_seed,
                    label="ileum"
                )
            print(f"[INFO] Added {len(added_subset)} external Ileum samples to TRAIN only (fold={fold}).")

            # ---- labeled loaders ----
            l_train_dataloader, l_val_dataloader, class_names, class_weights = create_labeled_dataloaders_from_paths(
                train_paths=train_paths,
                test_paths=val_paths,
                train_transform=train_transform,
                test_transform=test_transform,
                batch_size=opt.l_batch_size,
                num_workers=opt.num_workers
            )

            l_test_dataloader, _, _, _ = create_labeled_dataloaders_from_paths(
                train_paths=test_paths,
                test_paths=None,   # no fake dataset
                train_transform=test_transform,
                test_transform=test_transform,
                batch_size=opt.l_batch_size,
                num_workers=opt.num_workers,
                verbose=False
                )

            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

            # ---- info ----
            experiment_number += 1
            print(f"{b_}[INFO] Experiment number: {experiment_number}{sr_}")
            print(f"{i_}[INFO] Labelled train dl length: {len(l_train_dataloader)}{sr_}")
            print(f"[INFO] Labelled val dl length: {len(l_val_dataloader)}")
            print(f"[INFO] Labelled test dl length: {len(l_test_dataloader)}")
            print(f"[INFO] Unlabeled train dl length: {len(u_train_dataloader)}")
            print(f"[INFO] Unlabeled val dl length: {len(u_val_dataloader)}")
            print(f"[INFO] Unlabeled test dl length: {len(u_test_dataloader)}")
            print(f"[INFO] Class weights: {weights_tensor}")
            print(f"{i_}[INFO] Class names: {class_names}{sr_}")
            print("-" * 50 + "\n")

            # ---- models ----
            AE, classifier = make_models(AE_name, len(class_names), device)

            print(f"[INFO] Model: {opt.name}")
            print(f"[INFO] Exact model: {opt.classifier_name}_{AE_name}")
            print(f"[INFO] Epochs: {opt.epochs}")
            print(f"[INFO] CL LR: {opt.cl_lr}")
            print(sr_)

            # ---- outputs (fold-aware so you don't overwrite) ----
            base_out = Path(opt.checkpoints_dir) / opt.name/f"{opt.extra_img_n}" / f"{opt.classifier_name}_{AE_name}"/ f"fold_{fold}"
            training_progress_dir, model_weights_dir = create_output_dir(base_out)

            # ---- losses ----
            reconstruction_criterion = nn.MSELoss()
            classification_criterion = nn.CrossEntropyLoss()
            AE_classification_criterion = nn.CrossEntropyLoss(weight=weights_tensor)


            # ---- optim/sched ----
            optimizer_A = torch.optim.Adam(AE.parameters(), lr=opt.ae_lr, weight_decay=opt.weight_decay)
            optimizer_B = torch.optim.Adam(classifier.parameters(), lr=opt.cl_lr, weight_decay=opt.weight_decay)

            scheduler_A = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_A, mode='min', factor=0.5, patience=10,
                threshold_mode='rel', cooldown=1, min_lr=1e-6
            )
            scheduler_B = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_B, mode='max', factor=0.5, patience=10,
                threshold_mode='rel', cooldown=1, min_lr=1e-6
            )

            # ---- train ----
            train(
                AE=AE,
                classifier=classifier,
                train_dataloader_A=u_train_dataloader,
                val_dataloader_A=u_val_dataloader,
                test_dataloader_A=u_test_dataloader,
                train_dataloader_B=l_train_dataloader,
                test_dataloader_B=l_test_dataloader,
                val_dataloader_B=l_val_dataloader,
                optimizer_A=optimizer_A,
                optimizer_B=optimizer_B,
                loss_fn_A=reconstruction_criterion,
                loss_fn_B=classification_criterion,
                loss_fn_C=AE_classification_criterion,
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
                save_models=opt.save_models
            )
                            
            print(f"{c_}Experiment #{experiment_number} completed successfully.{sr_}")
            print("-" * 50 + "\n")

        # ---- fold averages ----
        print(f"Average performance across folds for {opt.classifier_name}_{AE_name}:")
        for metric, values in fold_results.items():
            mean = np.mean(values) if len(values) else float("nan")
            std = np.std(values) if len(values) else float("nan")
            print(f"{metric}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()