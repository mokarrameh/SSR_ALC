"""SSR_ALC HPO runner.

Runs a random-sampled search over:
    - alpha
    - beta
    - gamma

Search space is read from YAML config: train.search_space
Trials are trained using engine.engine_hpo.train
"""

from pathlib import Path
import csv
import random
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from colorama import Fore, Style

# Options (CycleGAN style)
from options.train_options import TrainOptions

from utils.config_utils import load_yaml_config

# Custom modules
from utils.data_setup import create_unlabeled_dataloaders, create_labeled_dataloaders_from_paths
from engine.utils import create_output_dir, weights_init
from engine.models.autoencoder import shallowConvAE, ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1
from engine.models.classifier import Classifier256, Classifier512, Classifier1024, Classifier2048
from engine.engine_hpo import train
from utils.fold_handle import get_fold_train_val_test_data, add_external_imgs_train_only, resolve_split_csv


def seed_everything(seed: int):
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


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _make_fold_results() -> Dict[str, List[Any]]:
    return {
        "AE_test_loss": [],
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


def _prepare_labeled_loaders_for_fold(
    *,
    opt,
    fold: int,
    split_csv_path: Path,
    l_image_path: Path,
    extra_img_dir: Optional[Path],
    train_transform,
    test_transform,
    device: torch.device,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    List[str],
    torch.Tensor
]:
    train_paths, val_paths, test_paths = get_fold_train_val_test_data(
        split_csv_path,
        l_image_path,
        fold_idx=fold,
        val_ratio=0.2,
        seed=opt.seed,
    )

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

    l_train_dataloader, l_val_dataloader, class_names, class_weights = create_labeled_dataloaders_from_paths(
        train_paths=train_paths,
        test_paths=val_paths,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=opt.l_batch_size,
        num_workers=opt.num_workers,
    )

    l_test_dataloader, _, _, _ = create_labeled_dataloaders_from_paths(
        train_paths=test_paths,
        test_paths=None,
        train_transform=test_transform,
        test_transform=test_transform,
        batch_size=opt.l_batch_size,
        num_workers=opt.num_workers,
        verbose=False,
    )

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return l_train_dataloader, l_val_dataloader, l_test_dataloader, class_names, weights_tensor


def run_one_trial(
    *,
    opt,
    AE_name: str,
    fold: int,
    device: torch.device,
    u_train_dataloader,
    u_val_dataloader,
    u_test_dataloader,
    l_train_dataloader,
    l_val_dataloader,
    l_test_dataloader,
    class_names: Sequence[str],
    weights_tensor: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    trial_out_dir: Path,
) -> Dict[str, Any]:
    AE, classifier = make_models(AE_name, len(class_names), device)

    training_progress_dir, model_weights_dir = create_output_dir(trial_out_dir)

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    AE_classification_criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer_A = torch.optim.Adam(AE.parameters(), lr=opt.ae_lr, weight_decay=opt.weight_decay)
    optimizer_B = torch.optim.Adam(classifier.parameters(), lr=opt.cl_lr, weight_decay=opt.weight_decay)

    scheduler_A = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_A, mode="min", factor=0.5, patience=10, threshold_mode="rel", cooldown=1, min_lr=1e-6
    )
    scheduler_B = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_B, mode="max", factor=0.5, patience=10, threshold_mode="rel", cooldown=1, min_lr=1e-6
    )

    fold_results = _make_fold_results()
    stats = train(
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
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        smote_seed=getattr(opt, "smote_seed", opt.seed),
        AE_lr=opt.ae_lr,
        CL_lr=opt.cl_lr,
        patience=opt.patience,
    )
    return stats


def main():
    # ---- parse options ----
    opt = TrainOptions().parse()

    # ---- colors ----
    c_ = Fore.GREEN
    e_ = Fore.RED
    i_ = Fore.BLUE
    b_ = Fore.MAGENTA
    sr_ = Style.RESET_ALL

    # ---- seed/device ----
    seed_everything(opt.seed)
    device = torch.device(opt.device)  # safe because BaseOptions resolves "auto"

    # load yaml to get search_space
    yaml_data = load_yaml_config(opt.config) if getattr(opt, "config", None) else {}
    train_cfg = (yaml_data.get("train") or {})
    search_space = (train_cfg.get("search_space") or {})

    # paths
    data_path = Path(opt.data_path)
    l_image_path = data_path / opt.labeled_dir
    u_image_path = data_path / opt.unlabeled_dir
    split_csv_path = resolve_split_csv(opt.split_csv, l_image_path)
    extra_img_dir = (l_image_path / opt.extra_img_dir) if opt.extra_img_dir else None

    u_train_path = u_image_path / opt.u_train_subdir
    u_val_path = u_image_path / opt.u_val_subdir
    u_test_path = u_image_path / opt.u_test_subdir

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
        num_workers=opt.num_workers,
    )

    # HPO space (from YAML; fallback to opt if missing)
    space = {
        "alpha": [float(x) for x in _as_list(search_space.get("alpha", opt.alpha))],
        "beta": [float(x) for x in _as_list(search_space.get("beta", opt.beta))],
        "gamma": [float(x) for x in _as_list(search_space.get("gamma", opt.gamma))],
    }

    all_cfgs = list(product(space["alpha"], space["beta"], space["gamma"]))
    if len(all_cfgs) == 0:
        raise RuntimeError("HPO: empty search space. Check train.search_space in YAML.")

    trials = all_cfgs

    folds = opt.folds if opt.folds else [0, 1, 2, 3, 4]

    print(f"[HPO] Selecting by validation macro-F1 returned as stats['best_val_f1']")
    print(f"[HPO] Trials: {len(trials)}/{len(all_cfgs)}")
    print(f"[HPO] Folds: {folds}")
    print(f"[HPO] AEs: {opt.aes}")

    # ---- run HPO ----
    for AE_name in opt.aes:
        for fold in folds:
            print(f"\n{Fore.CYAN}[HPO] AE={AE_name} fold={fold}{Style.RESET_ALL}")

            l_train_dataloader, l_val_dataloader, l_test_dataloader, class_names, weights_tensor = \
                _prepare_labeled_loaders_for_fold(
                    opt=opt,
                    fold=fold,
                    split_csv_path=split_csv_path,
                    l_image_path=l_image_path,
                    extra_img_dir=extra_img_dir,
                    train_transform=train_transform,
                    test_transform=test_transform,
                    device=device,
                )

            # IMPORTANT: per-AE + per-fold output dir (prevents overwrites)
            out_dir = Path(opt.checkpoints_dir) / opt.name / f"{AE_name}" / f"fold_{fold}"
            out_dir.mkdir(parents=True, exist_ok=True)
            trials_csv = out_dir / "trials.csv"
            best_yaml = out_dir / "best_config.yaml"

            best = {"best_val_f1": -float("inf"), "cfg": None, "stats": None}

            with open(trials_csv, "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "trial",
                        "alpha",
                        "beta",
                        "gamma",
                        "status",
                        "error",
                        "best_val_f1",
                        "best_val_epoch",
                        "best_ae_val_loss",
                        "best_ae_epoch",
                        "last_test_f1",
                    ],
                )
                w.writeheader()

                for t, (alpha, beta, gamma) in enumerate(trials, start=1):
                    trial_tag = f"a{alpha}_b{beta}_g{gamma}"
                    print(f"\n[HPO] Trial {t}/{len(trials)}: {trial_tag}")

                    trial_out_dir = out_dir / "runs" / trial_tag

                    status = "ok"
                    err_msg = ""
                    try:
                        stats = run_one_trial(
                            opt=opt,
                            AE_name=AE_name,
                            fold=fold,
                            device=device,
                            u_train_dataloader=u_train_dataloader,
                            u_val_dataloader=u_val_dataloader,
                            u_test_dataloader=u_test_dataloader,
                            l_train_dataloader=l_train_dataloader,
                            l_val_dataloader=l_val_dataloader,
                            l_test_dataloader=l_test_dataloader,
                            class_names=class_names,
                            weights_tensor=weights_tensor,
                            alpha=float(alpha),
                            beta=float(beta),
                            gamma=float(gamma),
                            trial_out_dir=trial_out_dir,
                        )
                    except Exception as ex:
                        status = "fail"
                        err_msg = f"{type(ex).__name__}: {ex}"
                        print(f"{Fore.RED}[HPO] Trial failed: {err_msg}{Style.RESET_ALL}")
                        stats = {
                            "best_val_f1": -float("inf"),
                            "best_val_epoch": None,
                            "best_ae_val_loss": None,
                            "best_ae_epoch": None,
                            "last_test_f1": None,
                        }

                    w.writerow(
                        {
                            "trial": t,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "status": status,
                            "error": err_msg,
                            "best_val_f1": stats.get("best_val_f1"),
                            "best_val_epoch": stats.get("best_val_epoch"),
                            "best_ae_val_loss": stats.get("best_ae_val_loss"),
                            "best_ae_epoch": stats.get("best_ae_epoch"),
                            "last_test_f1": stats.get("last_test_f1"),
                        }
                    )

                    bv = float(stats.get("best_val_f1", -float("inf")))
                    if bv > float(best["best_val_f1"]):
                        best["best_val_f1"] = bv
                        best["cfg"] = (float(alpha), float(beta), float(gamma))
                        best["stats"] = stats

            if best["cfg"] is None:
                print(f"{e_}[HPO] No successful trials for AE={AE_name} fold={fold}{sr_}")
                continue

            alpha_b, beta_b, gamma_b = best["cfg"]
            best_yaml.write_text(
                "train:\n"
                f"  alpha: {alpha_b}\n"
                f"  beta: {beta_b}\n"
                f"  gamma: {gamma_b}\n",
                encoding="utf-8",
            )

            print(
                f"\n[HPO] Best val_f1={best['best_val_f1']:.4f} with "
                f"a={alpha_b}, b={beta_b}, g={gamma_b}"
            )
            print(f"[HPO] Saved: {trials_csv}")
            print(f"[HPO] Saved: {best_yaml}")


if __name__ == "__main__":
    main()