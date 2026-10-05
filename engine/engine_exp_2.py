"""
Contains functions for training and testing the experiment 2.
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
from typing import Union

# Constants and Configurations
c_ = Fore.GREEN # improvements
w_ = Fore.YELLOW # warning
e_ = Fore.RED # error
i_ = Fore.BLUE # info
g_ = Fore.CYAN # changes
b_ = Fore.MAGENTA # begining
sr_ = Style.RESET_ALL

def choose_smote_k(y: Union[list, np.ndarray], *, base_k: int = 2, max_k: int = 10) -> int:
    """
    Choose k_neighbors based on the minority class count in y.
    Scales up slowly and clamps to valid range.
    """
    counts = Counter(y)
    n_min = min(counts.values())

    # scale rule (pick one):
    # k = base_k + int(math.log2(n_min)) - 1
    k = base_k + int(math.sqrt(n_min)) - 1   # grows gently

    # clamp to SMOTE-safe range
    k = max(1, min(k, max_k, n_min - 1))
    return k


def alpha_schedule(epoch, warmup_epochs=20, max_alpha=10.0):
    if epoch < warmup_epochs:
        return max_alpha * (epoch / warmup_epochs)
    return max_alpha


def train_step(AE: torch.nn.Module, 
               classifier: torch.nn.Module, 
               dataloader_A: torch.utils.data.DataLoader, 
               dataloader_B: torch.utils.data.DataLoader, 
               loss_fn_A: torch.nn.Module, 
               loss_fn_C: torch.nn.Module, 
               optimizer_A: torch.optim.Optimizer,
               optimizer_B: torch.optim.Optimizer,
               device: torch.device,
               num_classes=6,
               alpha=1,
               beta=10,
               gamma=5):    
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

        n_ae_steps += 1

        # ------------------ Classifier Training ------------------
        # Use encoded buffer and detach from computation graph for classifier-only training
        optimizer_B.zero_grad()
        
        logits = classifier(encoded_B.detach())
        loss = loss_fn_C(logits, labels_B)
        clf_loss_sum += float(loss.item())
        n_clf_steps += 1

        # Metrics: use class indices for acc/f1/precision/recall
        update_metrics(metrics, logits, labels_B)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer_B.step()

    # Define AE_train_loss in a consistent way:
    # mean reconstruction loss across (unlabeled + labeled) per AE step
    AE_train_loss = (unlabeled_recon_sum + labeled_recon_sum) / (2.0 * max(1, n_ae_steps))

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


def val_step(AE: torch.nn.Module, 
              classifier: torch.nn.Module,
        dataloader_A: torch.utils.data.DataLoader,
        dataloader_B: torch.utils.data.DataLoader,
        loss_fn_A: torch.nn.Module,
        loss_fn_B: torch.nn.Module,
        device: torch.device,
        num_classes=6):

    AE.eval()
    classifier.eval()

    unlabeled_recon_sum = 0.0
    labeled_recon_sum = 0.0
    n_ae_steps = 0

    clf_loss_sum = 0.0
    n_clf_steps = 0

    metrics = make_metrics(num_classes=num_classes, device=device)

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

            # ----- AE UNLABELED -----
            _, decoded_A = AE(data_A)
            loss_unlab = loss_fn_A(decoded_A, data_A)
            unlabeled_recon_sum += float(loss_unlab.item())

            # ----- AE + CLASSIFIER LABELED -----
            encoded_B, decoded_B = AE(data_B)

            loss_lab_recon = loss_fn_A(decoded_B, data_B)
            labeled_recon_sum += float(loss_lab_recon.item())

            logits = classifier(encoded_B)

            loss = loss_fn_B(logits, labels_B)
            clf_loss_sum += float(loss.item())
            n_clf_steps += 1

            update_metrics(metrics, logits, labels_B)

            n_ae_steps += 1

    AE_val_loss = (unlabeled_recon_sum + labeled_recon_sum) / (2.0 * max(1, n_ae_steps))
    val_loss = clf_loss_sum / max(1, n_clf_steps)

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
        num_classes=6):

    AE.eval()
    classifier.eval()

    unlabeled_recon_sum = 0.0
    labeled_recon_sum = 0.0
    n_ae_steps = 0

    clf_loss_sum = 0.0
    n_clf_steps = 0

    metrics = make_metrics(num_classes=num_classes, device=device)

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

            # ----- AE UNLABELED -----
            _, decoded_A = AE(data_A)
            loss_unlab = loss_fn_A(decoded_A, data_A)
            unlabeled_recon_sum += float(loss_unlab.item())

            # ----- AE + CLASSIFIER LABELED -----
            encoded_B, decoded_B = AE(data_B)

            loss_lab_recon = loss_fn_A(decoded_B, data_B)
            labeled_recon_sum += float(loss_lab_recon.item())

            logits = classifier(encoded_B)

            loss = loss_fn_B(logits, labels_B)
            clf_loss_sum += float(loss.item())
            n_clf_steps += 1

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


def train(AE: torch.nn.Module, 
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
          AE_lr=5e-4,
          CL_lr=0.0001,
          patience=20,
          save_models=True,
    ):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
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
            loss_fn_C=loss_fn_C,
            optimizer_A=optimizer_A,
            optimizer_B=optimizer_B,
            device=device,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
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
            if save_models:
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