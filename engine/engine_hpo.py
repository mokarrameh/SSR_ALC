"""
Contains functions for training and testing the experiment 4.
"""
import torch
import os 
from tqdm.auto import tqdm
import numpy as np
import csv
from pathlib import Path
import copy 
import math

from .metrics import make_metrics, update_metrics, compute_and_reset_metrics
from .utils import save_model
from .utils import EarlyStoppingF1
from .utils import display_random_images

# For colored terminal text
from colorama import Fore, Back, Style

from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Constants and Configurations
c_ = Fore.GREEN # improvements
w_ = Fore.YELLOW # warning
e_ = Fore.RED # error
i_ = Fore.BLUE # info
g_ = Fore.CYAN # changes
b_ = Fore.MAGENTA # begining
sr_ = Style.RESET_ALL


def alpha_schedule(epoch, warmup_epochs=20, max_alpha=10.0):
    if epoch < warmup_epochs:
        return max_alpha * (epoch / warmup_epochs)
    return max_alpha


def train_step(
    AE: torch.nn.Module, 
    classifier: torch.nn.Module, 
    dataloader_A: torch.utils.data.DataLoader, 
    dataloader_B: torch.utils.data.DataLoader, 
    loss_fn_A: torch.nn.Module, 
    loss_fn_B: torch.nn.Module, 
    loss_fn_C: torch.nn.Module,
    optimizer_A: torch.optim.Optimizer,
    optimizer_B: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 6,
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
    smote_seed: int = 42,
    smote_k_neighbors: int = 2,
    smote_sampling_strategy: str = "minority",
    smote_batch_size: int = 32,
    drop_last: bool = True,
    debug: bool = False
    ):
    """
    Runs one training epoch for the Autoencoder (AE) and classifier models.

    The training proceeds in two stages:
      1. The AE is trained on both unlabeled data (from `dataloader_A`) and 
         labeled data (from `dataloader_B`) using a weighted combination of 
         reconstruction loss and classification loss.
      2. Features extracted from the AE's encoder on labeled data are oversampled 
         using SMOTE to balance minority classes, and the classifier is trained 
         on the resampled feature set.

    Args:
        AE (torch.nn.Module): Autoencoder model (encoder used for feature extraction).
        classifier (torch.nn.Module): Classifier model operating on AE-encoded features.
        dataloader_A (torch.utils.data.DataLoader): DataLoader for unlabeled data 
            (used only for AE reconstruction).
        dataloader_B (torch.utils.data.DataLoader): DataLoader for labeled data 
            (used for both AE and classifier training).
        loss_fn_A (torch.nn.Module): Loss function for AE reconstruction (e.g. MSELoss).
        loss_fn_B (torch.nn.Module): Loss function for classifier training (e.g. CrossEntropyLoss).
        loss_fn_C (torch.nn.Module): Auxiliary classification loss applied jointly with AE training.
        optimizer_A (torch.optim.Optimizer): Optimizer for the AE parameters.
        optimizer_B (torch.optim.Optimizer): Optimizer for the classifier parameters.
        device (torch.device): Target device for computation (e.g. "cuda" or "cpu").
        num_classes (int, optional): Number of classes in the classification task. Defaults to 6.
        alpha (float, optional): Weight for classification loss `loss_fn_C` in AE training. Defaults to 1.
        beta (float, optional): Weight for reconstruction loss on labeled data. Defaults to 10.
        gamma (float, optional): Weight for reconstruction loss on unlabeled data. Defaults to 5.

    Returns:
        tuple:
            AE_train_loss (float): Average AE reconstruction loss per batch.
            train_loss (float): Average classifier loss per batch (after SMOTE).
            train_acc (float): Macro-averaged accuracy across all classes.
            train_acc_mca (ndarray): Per-class accuracy values.
            train_f1_score (float): Macro-averaged F1 score.
            train_f1_score_mca (ndarray): Per-class F1 scores.
            train_auroc (float): Macro-averaged AUROC score.
            train_auroc_mca (ndarray): Per-class AUROC scores.
            train_precision (float): Macro-averaged precision.
            train_recall (float): Macro-averaged recall.
    """
    # Put model in train mode
    AE.train()
    classifier.train()

    metrics = make_metrics(num_classes=num_classes, device=device)

    # --- accumulators ---
    unlabeled_recon_sum = 0.0
    labeled_recon_sum = 0.0
    n_ae_steps = 0

    clf_loss_sum = 0.0
    n_clf_steps = 0

    # Collect features for SMOTE
    feat_list = []
    lab_list = []
    
    # Use min length (explicit)
    steps = min(len(dataloader_A), len(dataloader_B))
    if steps == 0:
        raise RuntimeError("train_step: empty dataloader_A or dataloader_B.")

    # ------------------ Stage 1: AE training + collect labeled features ------------------
    for step_idx, (data_A, (data_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):
        if step_idx >= steps:
            break
        
        optimizer_A.zero_grad(set_to_none=True)

        data_A = data_A.to(device, non_blocking=True)
        data_B = data_B.to(device, non_blocking=True)
        labels_B = labels_B.to(device, non_blocking=True)

        # Unlabeled recon
        _, decoded_A = AE(data_A)
        loss_unlab = loss_fn_A(decoded_A, data_A)
        unlabeled_recon_sum += float(loss_unlab.item())

        # Labeled forward: encoded + recon + classification feedback
        encoded_B, decoded_B = AE(data_B)
        logits_B = classifier(encoded_B)

        loss_cls_feedback = loss_fn_C(logits_B, labels_B)
        loss_lab_recon = loss_fn_A(decoded_B, data_B)
        labeled_recon_sum += float(loss_lab_recon.item())

        total_loss = alpha * loss_cls_feedback + beta * loss_lab_recon + gamma * loss_unlab
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(AE.parameters(), max_norm=1.0)
        optimizer_A.step()

        # Collect features/labels for SMOTE (detach!)
        feat_list.append(encoded_B.detach().cpu().numpy())
        lab_list.append(labels_B.detach().cpu().numpy())

        n_ae_steps += 1

    if n_ae_steps == 0:
        raise RuntimeError("train_step: No AE steps ran (check dataloaders).")

    # Define AE_train_loss in a consistent way:
    # mean reconstruction loss across (unlabeled + labeled) per AE step
    AE_train_loss = (unlabeled_recon_sum + labeled_recon_sum) / (2.0 * max(1, n_ae_steps))

    # ------------------ Stage 2: SMOTE + classifier training ------------------
    labeled_features = np.concatenate(feat_list, axis=0)
    labeled_labels = np.concatenate(lab_list, axis=0)

    if debug:
        print(f"\n[DEBUG] labeled_features shape: {labeled_features.shape}")
        print(f"[DEBUG] labeled_labels shape: {labeled_labels.shape}")
        print(f"[DEBUG] labeled_labels counts: {Counter(labeled_labels)}")

    # Robust SMOTE: k_neighbors must be < minority_count
    counts = Counter(labeled_labels.tolist())
    minority_count = min(counts.values()) if len(counts) > 0 else 0

    # If minority_count < 2, SMOTE cannot run
    if minority_count < 2:
        raise RuntimeError(
            f"SMOTE cannot run: smallest class has {minority_count} sample(s). "
            "Increase labeled samples or reduce filtering."
        )

    k_eff = min(smote_k_neighbors, minority_count - 1)
    k_eff = max(1, k_eff)

    smote = SMOTE(
        random_state=smote_seed,
        k_neighbors=k_eff,
        sampling_strategy=smote_sampling_strategy,
    )
    X_resampled, y_resampled = smote.fit_resample(labeled_features, labeled_labels)
    
    if debug:
        print(f"[DEBUG] SMOTE k_neighbors used: {k_eff}")
        print(f"[DEBUG] Resampled dataset counts: {Counter(y_resampled)}")

    X_tensor = torch.from_numpy(X_resampled).float()
    y_tensor = torch.from_numpy(y_resampled).long()

    smote_dataset = TensorDataset(X_tensor, y_tensor)
    smote_loader = DataLoader(
        smote_dataset,
        batch_size=smote_batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,   # keep simple (features already in RAM)
        pin_memory=True,
    )

    for feats, labels in smote_loader:
        optimizer_B.zero_grad(set_to_none=True)

        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = classifier(feats)
        loss = loss_fn_B(logits, labels)

        clf_loss_sum += float(loss.item())
        n_clf_steps += 1

        # Metrics: use class indices for acc/f1/precision/recall
        update_metrics(metrics, logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer_B.step()

    if n_clf_steps == 0:
        raise RuntimeError("train_step: No classifier steps ran (SMOTE loader empty).")

    train_loss = clf_loss_sum / max(1, n_clf_steps)
    train_acc, train_acc_mca, train_f1, train_f1_mca, train_auroc, train_auroc_mca, train_precision, train_recall = \
        compute_and_reset_metrics(metrics)

    return (
        AE_train_loss,
        train_loss,
        train_acc,
        train_acc_mca,
        train_f1,
        train_f1_mca,
        train_auroc,
        train_auroc_mca,
        train_precision,
        train_recall,
    )


def val_step(
    AE: torch.nn.Module,
    classifier: torch.nn.Module,
    dataloader_A: torch.utils.data.DataLoader,
    dataloader_B: torch.utils.data.DataLoader,
    loss_fn_A: torch.nn.Module,
    loss_fn_B: torch.nn.Module,
    device: torch.device,
    num_classes: int = 6,
):
    """
    Validation:
      - AE recon loss on unlabeled + labeled (reported as mean of the two per step)
      - classifier metrics on labeled
    """

    AE.eval()
    classifier.eval()
    
    metrics = make_metrics(num_classes=num_classes, device=device)

    unlabeled_recon_sum = 0.0
    labeled_recon_sum = 0.0
    n_ae_steps = 0

    clf_loss_sum = 0.0
    n_clf_steps = 0

    steps = min(len(dataloader_A), len(dataloader_B))
    if len(dataloader_B) > len(dataloader_A):
        raise ValueError(
            f"val_step: labeled loader has {len(dataloader_B)} batches but the unlabeled loader only "
            f"{len(dataloader_A)}; zipping them would silently skip labeled samples. "
            "Use a larger unlabeled subset or a smaller unlabeled batch size."
        )
    if steps == 0:
        raise RuntimeError("val_step: empty dataloader_A or dataloader_B.")

    with torch.inference_mode():
        for step_idx, (data_A, (data_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):
            if step_idx >= steps:
                break

            data_A = data_A.to(device, non_blocking=True)
            data_B = data_B.to(device, non_blocking=True)
            labels_B = labels_B.to(device, non_blocking=True)

            # AE on unlabeled
            _, decoded_A = AE(data_A)
            loss_unlab = loss_fn_A(decoded_A, data_A)
            unlabeled_recon_sum += float(loss_unlab.item())

            # AE + classifier on labeled
            encoded_B, decoded_B = AE(data_B)
            logits = classifier(encoded_B)

            loss_cls = loss_fn_B(logits, labels_B)
            clf_loss_sum += float(loss_cls.item())
            n_clf_steps += 1

            loss_lab_recon = loss_fn_A(decoded_B, data_B)
            labeled_recon_sum += float(loss_lab_recon.item())

            # metrics
            update_metrics(metrics, logits, labels_B)

            n_ae_steps += 1

    AE_val_loss = (unlabeled_recon_sum + labeled_recon_sum) / (2.0 * max(1, n_ae_steps))
    val_loss = clf_loss_sum / max(1, n_clf_steps)
    # consistent averaging
    val_acc, val_acc_mca, val_f1, val_f1_mca, val_auroc, val_auroc_mca, val_precision, val_recall = \
        compute_and_reset_metrics(metrics)

    return (
        AE_val_loss,
        val_loss,
        val_acc,
        val_acc_mca,
        val_f1,
        val_f1_mca,
        val_auroc,
        val_auroc_mca,
        val_precision,
        val_recall,
    )


def test_step(
    AE: torch.nn.Module,
    classifier: torch.nn.Module,
    dataloader_A: torch.utils.data.DataLoader,
    dataloader_B: torch.utils.data.DataLoader,
    loss_fn_A: torch.nn.Module,
    loss_fn_B: torch.nn.Module,
    device: torch.device,
    num_classes: int = 6,
):
    """
    Test:
      - AE recon loss on unlabeled + labeled (reported as mean of the two per step)
      - classifier metrics on labeled
    """
    import numpy as np

    AE.eval()
    classifier.eval()

    metrics = make_metrics(num_classes=num_classes, device=device)

    unlabeled_recon_sum = 0.0
    labeled_recon_sum = 0.0
    n_ae_steps = 0

    clf_loss_sum = 0.0
    n_clf_steps = 0

    steps = min(len(dataloader_A), len(dataloader_B))
    if len(dataloader_B) > len(dataloader_A):
        raise ValueError(
            f"test_step: labeled loader has {len(dataloader_B)} batches but the unlabeled loader only "
            f"{len(dataloader_A)}; zipping them would silently skip labeled samples. "
            "Use a larger unlabeled subset or a smaller unlabeled batch size."
        )
    if steps == 0:
        raise RuntimeError("test_step: empty dataloader_A or dataloader_B.")

    with torch.inference_mode():
        for step_idx, (data_A, (data_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):
            if step_idx >= steps:
                break

            data_A = data_A.to(device, non_blocking=True)
            data_B = data_B.to(device, non_blocking=True)
            labels_B = labels_B.to(device, non_blocking=True)

            # AE on unlabeled
            _, decoded_A = AE(data_A)
            loss_unlab = loss_fn_A(decoded_A, data_A)
            unlabeled_recon_sum += float(loss_unlab.item())

            # AE + classifier on labeled
            encoded_B, decoded_B = AE(data_B)
            logits = classifier(encoded_B)

            loss_cls = loss_fn_B(logits, labels_B)
            clf_loss_sum += float(loss_cls.item())
            n_clf_steps += 1

            loss_lab_recon = loss_fn_A(decoded_B, data_B)
            labeled_recon_sum += float(loss_lab_recon.item())

            # metrics
            update_metrics(metrics, logits, labels_B)

            n_ae_steps += 1

    AE_test_loss = (unlabeled_recon_sum + labeled_recon_sum) / (2.0 * max(1, n_ae_steps))
    test_loss = clf_loss_sum / max(1, n_clf_steps)

    test_acc, test_acc_mca, test_f1, test_f1_mca, test_auroc, test_auroc_mca, test_precision, test_recall = \
        compute_and_reset_metrics(metrics)

    return (
        AE_test_loss,
        test_loss,
        test_acc,
        test_acc_mca,
        test_f1,
        test_f1_mca,
        test_auroc,
        test_auroc_mca,
        test_precision,
        test_recall,
    )


def train(
    AE: torch.nn.Module, 
    classifier: torch.nn.Module, 
    train_dataloader_A: torch.utils.data.DataLoader,
    val_dataloader_A: torch.utils.data.DataLoader,
    test_dataloader_A: torch.utils.data.DataLoader, 
    train_dataloader_B: torch.utils.data.DataLoader, 
    test_dataloader_B: torch.utils.data.DataLoader,
    val_dataloader_B: torch.utils.data.DataLoader,
    optimizer_A: torch.optim.Optimizer,
    optimizer_B: torch.optim.Optimizer,
    loss_fn_A: torch.nn.Module,
    loss_fn_B: torch.nn.Module,
    loss_fn_C: torch.nn.Module,
    epochs: int,
    device: torch.device,
    scheduler_A,
    scheduler_B,
    training_progress_dir,
    model_weights_dir,
    fold_results,
    fold,
    num_classes=None,
    alpha=1,
    beta=10,
    gamma=5,
    smote_k_neighbors: int = 2,
    smote_seed: int = 42,
    smote_sampling_strategy: str = "minority",
    smote_batch_size: int = 32,
    AE_lr=5e-4,
    CL_lr=0.0001,
    patience=20):
    """
    Trains, validates, and tests the Autoencoder (AE) and classifier models over multiple epochs.

    This function performs:
        - Training with `train_step` on both unlabeled (A) and labeled (B) datasets.
        - Validation with `val_step` to monitor AE reconstruction loss and classifier metrics.
        - Early stopping based on validation F1 score.
        - Learning rate scheduling for both AE and classifier optimizers.
        - SMOTE-based oversampling for class imbalance during classifier training.
        - Saving of best model weights and reconstruction images.
        - Logging of metrics into CSV files for each epoch and final test results.

    Args:
        AE (torch.nn.Module): Autoencoder model (encoder used for feature extraction).
        classifier (torch.nn.Module): Classifier model operating on AE-encoded features.
        train_dataloader_A (DataLoader): DataLoader for unlabeled training data (AE training).
        val_dataloader_A (DataLoader): DataLoader for unlabeled validation data (AE validation).
        test_dataloader_A (DataLoader): DataLoader for unlabeled test data (AE evaluation).
        train_dataloader_B (DataLoader): DataLoader for labeled training data (classifier training).
        val_dataloader_B (DataLoader): DataLoader for labeled validation data (classifier validation).
        test_dataloader_B (DataLoader): DataLoader for labeled test data (classifier evaluation).
        optimizer_A (torch.optim.Optimizer): Optimizer for AE parameters.
        optimizer_B (torch.optim.Optimizer): Optimizer for classifier parameters.
        loss_fn_A (torch.nn.Module): Loss function for AE reconstruction.
        loss_fn_B (torch.nn.Module): Loss function for classifier on SMOTE-resampled features.
        loss_fn_C (torch.nn.Module): Loss function for classifier on original labeled data.
        epochs (int): Number of epochs to train.
        device (torch.device): Target device for computation ('cpu' or 'cuda').
        scheduler_A: Learning rate scheduler for AE optimizer.
        scheduler_B: Learning rate scheduler for classifier optimizer.
        training_progress_dir (str / Path): Directory to save reconstructed images during training.
        model_weights_dir (str / Path): Directory to save model checkpoints and CSV logs.
        fold_results (dict): Dictionary to store training/validation/test metrics for cross-validation folds.
        fold (int): Current fold index for cross-validation.
        num_classes (int, optional): Number of classes for classification. Defaults to None.
        alpha (float, optional): Weight for classifier loss on original labeled data. Defaults to 1.
        beta (float, optional): Weight for AE reconstruction loss on labeled data. Defaults to 10.
        gamma (float, optional): Weight for AE reconstruction loss on unlabeled data. Defaults to 5.
        AE_lr (float, optional): Initial learning rate for AE optimizer. Defaults to 5e-4.
        CL_lr (float, optional): Initial learning rate for classifier optimizer. Defaults to 1e-4.
        patience (int, optional): Number of epochs with no improvement to trigger early stopping. Defaults to 20.

    Returns:
        None: Results are saved in `fold_results` dictionary and CSV files in `model_weights_dir`.

    Notes:
        - The function performs gradient clipping for both AE and classifier models.
        - Early stopping is applied based on validation F1 score (`val_f1`).
        - Best AE and classifier models are saved whenever validation loss or F1 improves.
        - The number of validation steps is limited by the smaller of the unlabeled and labeled dataloaders.
        - Final test metrics are computed using `test_step` and saved in `fold_results`.
    """
    # ---- normalize paths ----
    training_progress_dir = Path(training_progress_dir)
    model_weights_dir = Path(model_weights_dir)
    training_progress_dir.mkdir(parents=True, exist_ok=True)
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = model_weights_dir / f"{fold}.csv"
    test_file_path = model_weights_dir / f"{fold}_test.csv"

    fieldnames = [
        "Epoch",
        "AE_train_loss",
        "AE_val_loss",
        "train_loss",
        "train_acc",
        "train_acc_mca",
        "train_f1",
        "train_f1_mca",
        "train_auroc",
        "train_auroc_mca",
        "train_precision",
        "train_recall",
        "val_loss",
        "val_acc",
        "val_acc_mca",
        "val_f1",
        "val_f1_mca",
        "val_auroc",
        "val_auroc_mca",
        "val_precision",
        "val_recall",
    ]

    tst_fieldnames = [
        "Fold",
        "AE_test_loss",
        "test_loss",
        "test_acc",
        "test_acc_mca",
        "test_f1",
        "test_f1_mca",
        "test_auroc",
        "test_auroc_mca",
        "test_precision",
        "test_recall",
    ]

    # ---- create CSV headers if needed ----
    if not csv_file_path.exists():
        with csv_file_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    if not test_file_path.exists():
        with test_file_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=tst_fieldnames)
            w.writeheader()

    # ---- best trackers ----
    best_f1 = -np.inf
    best_epoch_cl = -1

    best_loss = np.inf
    best_epoch_ae = -1

    # Keep best weights in memory so test uses BEST (not last)
    best_AE_state = copy.deepcopy(AE.state_dict())
    best_CL_state = copy.deepcopy(classifier.state_dict())

    classifier.to(device)
    AE.to(device)

    early_stopping_f1 = EarlyStoppingF1(patience=patience, min_delta=0.001)

    # Track LR with tolerance
    prev_ae_lr = float(AE_lr)
    prev_cl_lr = float(CL_lr)

    # ---- training loop ----
    for epoch in tqdm(range(epochs)):
        (
            AE_train_loss,
            train_loss,
            train_acc,
            train_acc_mca,
            train_f1,
            train_f1_mca,
            train_auroc,
            train_auroc_mca,
            train_precision,
            train_recall,
        ) = train_step(
            AE=AE,
            classifier=classifier,
            dataloader_A=train_dataloader_A,
            dataloader_B=train_dataloader_B,
            loss_fn_A=loss_fn_A,
            loss_fn_B=loss_fn_B,
            loss_fn_C=loss_fn_C,
            optimizer_A=optimizer_A,
            optimizer_B=optimizer_B,
            device=device,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smote_seed=smote_seed,
            smote_k_neighbors=smote_k_neighbors,
            smote_sampling_strategy=smote_sampling_strategy,
            smote_batch_size=smote_batch_size,
        )

        (
            AE_val_loss,
            val_loss,
            val_acc,
            val_acc_mca,
            val_f1,
            val_f1_mca,
            val_auroc,
            val_auroc_mca,
            val_precision,
            val_recall,
        ) = val_step(
            AE=AE,
            classifier=classifier,
            dataloader_A=val_dataloader_A,
            dataloader_B=val_dataloader_B,
            loss_fn_A=loss_fn_A,
            loss_fn_B=loss_fn_B,
            device=device,
            num_classes=num_classes
        )

        # schedulers
        if scheduler_A is not None:
            scheduler_A.step(AE_val_loss)
        if scheduler_B is not None:
            scheduler_B.step(val_f1)

        # ---- logging (kept similar to yours) ----
        print(
            f"{sr_}Epoch: {epoch+1}\n"
            f"\t|AE_train_loss: {AE_train_loss:.4f} | AE_val_loss: {AE_val_loss:.4f}"
        )
        print(
            f"{sr_}Epoch: {epoch+1}\n"
            f"\t|train_loss: {train_loss:.4f}\n"
            f"\t|train_acc: {train_acc:.4f} | train_f1: {train_f1:.4f} | train_auroc: {train_auroc:.4f} | "
            f"train_precision: {train_precision:.4f} | train_recall: {train_recall:.4f}\n"
            f"\t|val_loss: {val_loss:.4f}\n"
            f"\t|val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | val_auroc: {val_auroc:.4f} | "
            f"val_precision: {val_precision:.4f} | val_recall: {val_recall:.4f}"
        )

        for i, f1 in enumerate(train_f1_mca):
            print(f"Class {i} Train F1: {float(f1):.4f}")
        for i, f1 in enumerate(val_f1_mca):
            print(f"Class {i} Validation F1: {float(f1):.4f}")

        # ---- best AE by AE_val_loss ----
        if AE_val_loss < best_loss:
            print(f"{c_}AE val Loss Improved ({best_loss:0.4f} ---> {AE_val_loss:0.4f})")
            best_loss = float(AE_val_loss)
            best_epoch_ae = epoch + 1

            # Save recon image (Path-safe)
            display_random_images(
                data_loader=val_dataloader_A,
                AE=AE,
                file_recon=str(training_progress_dir / f"epoch{epoch + 1}_fold{fold}_val_recon.png"),
                display_real=False,
                device=device,
            )
            print(f"{sr_}")

        # ---- best CL by val_f1 ----
        if val_f1 > best_f1:
            print(f"{c_}CL Val F1 Improved ({best_f1:0.4f} ---> {val_f1:0.4f})")
            best_f1 = float(val_f1)
            best_epoch_cl = epoch + 1

            # keep best in memory
            best_CL_state = copy.deepcopy(classifier.state_dict())
            best_AE_state = copy.deepcopy(AE.state_dict())

            # also save to disk (your naming kept)
            save_model(
                model_state_dict=best_CL_state,
                target_dir=model_weights_dir,
                model_name=f"CL_{epochs}_epochs_{fold}fold.pth",
            )
            save_model(
                model_state_dict=best_AE_state,
                target_dir=model_weights_dir,
                model_name=f"AE_{epochs}_epochs_{fold}fold.pth",
            )
            print(f"{sr_}")

        # ---- append epoch row ----
        with csv_file_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(
                {
                    "Epoch": epoch + 1,
                    "AE_train_loss": AE_train_loss,
                    "AE_val_loss": AE_val_loss,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_acc_mca": train_acc_mca,
                    "train_f1": train_f1,
                    "train_f1_mca": train_f1_mca,
                    "train_auroc": train_auroc,
                    "train_auroc_mca": train_auroc_mca,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_acc_mca": val_acc_mca,
                    "val_f1": val_f1,
                    "val_f1_mca": val_f1_mca,
                    "val_auroc": val_auroc,
                    "val_auroc_mca": val_auroc_mca,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                }
            )

        # ---- LR change print (tolerant) ----
        if scheduler_A is not None:
            curr = float(scheduler_A.get_last_lr()[-1])
            if not math.isclose(curr, prev_ae_lr, rel_tol=0.0, abs_tol=1e-12):
                print(f"{g_}AE learning rate changed: {prev_ae_lr} -> {curr}{sr_}")
                prev_ae_lr = curr

        if scheduler_B is not None:
            curr = float(scheduler_B.get_last_lr()[-1])
            if not math.isclose(curr, prev_cl_lr, rel_tol=0.0, abs_tol=1e-12):
                print(f"{g_}CL learning rate changed: {prev_cl_lr} -> {curr}")
                prev_cl_lr = curr

                # In HPO mode, keep alpha fixed for the whole trial.
                # (The previous code tried to schedule alpha but referenced an undefined function.)
                print(f"{g_}alpha kept fixed at: {alpha}{sr_}")

        # ---- early stopping by val_f1 ----
        early_stopping_f1(val_f1)
        if early_stopping_f1.early_stop:
            print(f"{w_}Early stopping at epoch {epoch+1}{sr_}")
            break

    # ---- IMPORTANT: load BEST weights before testing ----
    AE.load_state_dict(best_AE_state)
    classifier.load_state_dict(best_CL_state)

    (
        AE_test_loss,
        test_loss,
        test_acc,
        test_acc_mca,
        test_f1,
        test_f1_mca,
        test_auroc,
        test_auroc_mca,
        test_precision,
        test_recall,
    ) = test_step(
        AE=AE,
        classifier=classifier,
        dataloader_A=test_dataloader_A,
        dataloader_B=test_dataloader_B,
        loss_fn_A=loss_fn_A,
        loss_fn_B=loss_fn_B,
        device=device,
        num_classes=num_classes,
    )

    display_random_images(
        data_loader=test_dataloader_A,
        AE=AE,
        file_recon=str(training_progress_dir / f"fold{fold}_test_recon.png"),
        display_real=False,
        device=device,
    )

    # record fold results
    fold_results["AE_test_loss"].append(AE_test_loss)
    fold_results["test_loss"].append(test_loss)
    fold_results["test_acc"].append(test_acc)
    fold_results["test_acc_mca"].append(test_acc_mca)
    fold_results["test_f1"].append(test_f1)
    fold_results["test_f1_mca"].append(test_f1_mca)
    fold_results["test_auroc"].append(test_auroc)
    fold_results["test_auroc_mca"].append(test_auroc_mca)
    fold_results["test_precision"].append(test_precision)
    fold_results["test_recall"].append(test_recall)

    # write test row
    with test_file_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tst_fieldnames)
        w.writerow(
            {
                "Fold": fold,
                "AE_test_loss": AE_test_loss,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_acc_mca": test_acc_mca,
                "test_f1": test_f1,
                "test_f1_mca": test_f1_mca,
                "test_auroc": test_auroc,
                "test_auroc_mca": test_auroc_mca,
                "test_precision": test_precision,
                "test_recall": test_recall,
            }
        )

    print(f"{c_}[INFO] Final Val Results:")
    print(f"Best CL Epoch: {best_epoch_cl}\n\t|Best CL F1: {best_f1:.4f}")
    print(f"Best AE Epoch: {best_epoch_ae}\n\t|Best AE Loss: {best_loss:.4f}")

    print(f"{c_}[INFO] Final Test Results:")
    for metric, values in fold_results.items():
        if "mca" not in metric:
            print(f"{metric}: {values[-1]:.4f}")
    print(f"{sr_}")

    return {
    "best_val_f1": best_f1,
    "best_val_epoch": best_epoch_cl,
    "best_ae_val_loss": best_loss,
    "best_ae_epoch": best_epoch_ae,
    "last_test_f1": fold_results["test_f1"][-1] if len(fold_results["test_f1"]) else None,
}
