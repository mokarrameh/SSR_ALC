"""
Unsupervised autoencoder pretraining (ONLY_AE).

Trains each autoencoder on the unlabeled HyperKvasir subset with a pure
reconstruction (MSE) loss. The resulting weights are loaded by Experiments 1
and 3 (`exp1_train.py`, `exp3_train.py`) from

    <checkpoints_dir>/ONLY_AE/<AE_name>/model_weights/AE_<epochs>_epochs.pth

Restored from the original `train_only_ae.py` and ported to the YAML/CLI
options used by the other scripts. Hyperparameters are unchanged
(see configs/only_ae.yaml).

Example:
    python train_only_ae.py --config configs/only_ae.yaml --data_path /path/to/data
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, Style
from torchvision import transforms

from options.train_options import TrainOptions
from utils.data_setup import create_unlabeled_dataloaders
from engine.utils import create_output_dir, weights_init
from engine.models.autoencoder import ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1
from engine.engine_only_ae import train


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


AE_FACTORY = {
    "convAE": ConvAutoencoder,             # 512-d latent
    "deepConvAE": DeepConvAutoencoder,     # 1024-d latent
    "deepConvAEV1": DeepConvAutoencoderV1, # 2048-d latent
}


def main():
    opt = TrainOptions().parse()
    seed_everything(opt.seed)
    device = torch.device(opt.device)

    u_image_path = Path(opt.data_path) / opt.unlabeled_dir
    train_transform, test_transform = build_transforms()

    u_train_dataloader, u_val_dataloader, u_test_dataloader = create_unlabeled_dataloaders(
        train_dir=u_image_path / opt.u_train_subdir,
        val_dir=u_image_path / opt.u_val_subdir,
        test_dir=u_image_path / opt.u_test_subdir,
        train_transform=train_transform,
        test_transform=test_transform,
        train_batch_size=opt.u_train_batch_size,
        val_batch_size=opt.u_val_batch_size,
        test_batch_size=opt.u_test_batch_size,
        num_workers=opt.num_workers,
    )
    print(f"{Fore.BLUE}[INFO] Unlabeled train/val/test batches: "
          f"{len(u_train_dataloader)}/{len(u_val_dataloader)}/{len(u_test_dataloader)}{Style.RESET_ALL}")

    for AE_name in opt.aes:
        if AE_name not in AE_FACTORY:
            raise ValueError(f"Unknown AE name: {AE_name}. Choose from {list(AE_FACTORY)}")
        AE = AE_FACTORY[AE_name]().to(device)
        AE.apply(weights_init)

        print(f"{Fore.MAGENTA}[INFO] ONLY_AE: {AE_name} | epochs={opt.epochs} | lr={opt.ae_lr}{Style.RESET_ALL}")
        training_progress_dir, model_weights_dir = create_output_dir(Path(opt.checkpoints_dir) / opt.name / AE_name)

        optimizer_A = torch.optim.Adam(AE.parameters(), lr=opt.ae_lr, weight_decay=opt.weight_decay)
        scheduler_A = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_A, mode="min", factor=0.5, patience=10,
            threshold_mode="rel", cooldown=1, min_lr=1e-6,
        )

        fold_results = {"AE_test_loss": []}
        train(
            AE=AE,
            train_dataloader_A=u_train_dataloader,
            val_dataloader_A=u_val_dataloader,
            test_dataloader_A=u_test_dataloader,
            optimizer_A=optimizer_A,
            loss_fn_A=nn.MSELoss(),
            epochs=opt.epochs,
            device=device,
            scheduler_A=scheduler_A,
            training_progress_dir=training_progress_dir,
            model_weights_dir=model_weights_dir,
            fold_results=fold_results,
            AE_lr=opt.ae_lr,
            patience=opt.patience,
        )
        print(f"{Fore.GREEN}[INFO] {AE_name} done. Test reconstruction loss: "
              f"{fold_results['AE_test_loss'][-1]:.4f}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
