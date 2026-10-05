"""
Contains functions for training and testing experiment 3.
"""
import copy
import csv
import math
import os
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .metrics import make_metrics, update_metrics, compute_and_reset_metrics
from .utils import save_model, EarlyStoppingF1

# For colored terminal text
from colorama import Fore, Style

# Constants and Configurations
c_ = Fore.GREEN # improvements
w_ = Fore.YELLOW # warning
e_ = Fore.RED # error
i_ = Fore.BLUE # info
g_ = Fore.CYAN # changes
b_ = Fore.MAGENTA # begining
sr_ = Style.RESET_ALL


def choose_smote_k(y: Union[list, np.ndarray], *, base_k: int = 10, max_k: int = 10) -> int:
    """Pick a SMOTE-safe k_neighbors based on minority class count.

    This preserves the old behavior (k=10) whenever possible, but clamps
    to a valid value when a fold has too few minority samples.
    """
    counts = Counter(y)
    n_min = min(counts.values())

    # Keep exp3 historical default (10) but clamp to SMOTE-safe range.
    k = base_k + int(math.sqrt(n_min)) - 1
    k = max(1, min(k, max_k, n_min - 1))
    return k

def train_step(
    AE: torch.nn.Module,
    classifier: torch.nn.Module,
    dataloader_B: torch.utils.data.DataLoader,
    loss_fn_B: torch.nn.Module,
    optimizer_B: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 6,
    smote_seed: int = 42,
    smote_sampling_strategy: str = "minority",
    smote_batch_size: int = 32,
    drop_last: bool = True,
    debug: bool = False,
):
    """One epoch: extract AE embeddings, apply SMOTE, train classifier."""

    classifier.train()
    AE.eval()  # frozen feature extractor

    metrics = make_metrics(num_classes=num_classes, device=device)

    feat_list = []
    lab_list = []

    with torch.inference_mode():
        for imgs, labels in dataloader_B:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            encoded, _ = AE(imgs)
            feat_list.append(encoded.detach().cpu().numpy())
            lab_list.append(labels.detach().cpu().numpy())

    if len(feat_list) == 0:
        raise RuntimeError("train_step: empty dataloader_B.")

    labeled_features = np.concatenate(feat_list, axis=0)
    labeled_labels = np.concatenate(lab_list, axis=0)

    if debug:
        print(f"\n[DEBUG] labeled_features shape: {labeled_features.shape}")
        print(f"[DEBUG] labeled_labels counts: {Counter(labeled_labels)}")

    # ---- SMOTE (robust to small minority folds) ----
    counts = Counter(labeled_labels)
    if len(counts) < 2:
        # Single-class fold (shouldn't happen with stratified split, but be defensive)
        X_resampled, y_resampled = labeled_features, labeled_labels
    else:
        n_min = min(counts.values())
        if n_min < 2:
            # SMOTE cannot run with only 1 sample in minority class.
            X_resampled, y_resampled = labeled_features, labeled_labels
        else:
            k_dynamic = choose_smote_k(labeled_labels, base_k=10, max_k=10)
            smote = SMOTE(
                random_state=smote_seed,
                k_neighbors=k_dynamic,
                sampling_strategy=smote_sampling_strategy,
            )
            X_resampled, y_resampled = smote.fit_resample(labeled_features, labeled_labels)

            if debug:
                print(f"[DEBUG] SMOTE k_neighbors used: {k_dynamic}")
                print(f"[DEBUG] Resampled counts: {Counter(y_resampled)}")

    X_tensor = torch.from_numpy(X_resampled).float()
    y_tensor = torch.from_numpy(y_resampled).long()

    smote_dataset = TensorDataset(X_tensor, y_tensor)
    smote_loader = DataLoader(
        smote_dataset,
        batch_size=smote_batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,
    )

    clf_loss_sum = 0.0
    n_steps = 0

    for feats, labels in smote_loader:
        optimizer_B.zero_grad(set_to_none=True)

        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = classifier(feats)
        loss = loss_fn_B(logits, labels)

        clf_loss_sum += float(loss.item())
        n_steps += 1

        update_metrics(metrics, logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer_B.step()

    if n_steps == 0:
        raise RuntimeError("train_step: SMOTE loader empty (no classifier steps).")

    train_loss = clf_loss_sum / max(1, n_steps)
    train_acc, train_acc_mca, train_f1, train_f1_mca, train_auroc, train_auroc_mca, train_precision, train_recall = \
        compute_and_reset_metrics(metrics)

    return (
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

def test_step(AE: torch.nn.Module, 
              classifier: torch.nn.Module, 
              dataloader_B: torch.utils.data.DataLoader, 
              loss_fn_B: torch.nn.Module,
              device: torch.device,
              num_classes=6) -> float:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    AE.eval()
    classifier.eval()

    metrics = make_metrics(num_classes=num_classes, device=device)

    loss_sum = 0.0
    n_steps = 0

    with torch.inference_mode():
        for imgs, labels in dataloader_B:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            encoded, _ = AE(imgs)
            logits = classifier(encoded)

            loss = loss_fn_B(logits, labels)
            loss_sum += float(loss.item())
            n_steps += 1

            update_metrics(metrics, logits, labels)

    test_loss = loss_sum / max(1, n_steps)
    test_acc, test_acc_mca, test_f1, test_f1_mca, test_auroc, test_auroc_mca, test_precision, test_recall = \
        compute_and_reset_metrics(metrics)

    return (
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

def val_step(AE: torch.nn.Module, 
              classifier: torch.nn.Module, 
              dataloader_B: torch.utils.data.DataLoader, 
              loss_fn_B: torch.nn.Module,
              device: torch.device,
              num_classes=6) -> float:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    AE.eval()
    classifier.eval()

    metrics = make_metrics(num_classes=num_classes, device=device)

    loss_sum = 0.0
    n_steps = 0

    with torch.inference_mode():
        for imgs, labels in dataloader_B:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            encoded, _ = AE(imgs)
            logits = classifier(encoded)

            loss = loss_fn_B(logits, labels)
            loss_sum += float(loss.item())
            n_steps += 1

            update_metrics(metrics, logits, labels)

    val_loss = loss_sum / max(1, n_steps)
    val_acc, val_acc_mca, val_f1, val_f1_mca, val_auroc, val_auroc_mca, val_precision, val_recall = \
        compute_and_reset_metrics(metrics)

    return (
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

def train(AE: torch.nn.Module, 
          classifier: torch.nn.Module, 
          train_dataloader_B: torch.utils.data.DataLoader, 
          test_dataloader_B: torch.utils.data.DataLoader,
          val_dataloader_B: torch.utils.data.DataLoader,
          optimizer_B: torch.optim.Optimizer,
          loss_fn_B: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler_B,
          model_weights_dir,
          fold_results,
          fold,
          num_classes=None,
          CL_lr=0.0001,
          patience=20,
          save_models: bool = True):
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

    model_weights_dir = Path(model_weights_dir)
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = model_weights_dir / f"{fold}.csv"
    test_file_path = model_weights_dir / f"{fold}_test.csv"

    fieldnames = [
        "Epoch",
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

    if not csv_file_path.exists():
        with csv_file_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    if not test_file_path.exists():
        with test_file_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=tst_fieldnames)
            w.writeheader()
    
    best_f1 = -np.inf
    best_epoch_cl = -1
    best_CL_state = copy.deepcopy(classifier.state_dict())
    
    # Make sure model on target device
    classifier.to(device)
    AE.to(device)

    early_stopping_f1 = EarlyStoppingF1(patience=patience, min_delta=0.001)

    prev_cl_lr = float(CL_lr)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):

        train_loss, train_acc, train_acc_mca, train_f1, train_f1_mca, train_auroc, train_auroc_mca, train_precision, train_recall = train_step(AE=AE, 
                                                                                                                                classifier=classifier, 
                                                                                                                                dataloader_B=train_dataloader_B, 
                                                                                                                                loss_fn_B = loss_fn_B,
                                                                                                                                optimizer_B=optimizer_B,
                                                                                                                                device=device,
                                                                                                                                num_classes=num_classes)

        val_loss, val_acc, val_acc_mca, val_f1, val_f1_mca, val_auroc, val_auroc_mca, val_precision, val_recall = val_step(AE=AE, 
                                                                                                                    classifier=classifier, 
                                                                                                                    dataloader_B=val_dataloader_B, 
                                                                                                                    loss_fn_B=loss_fn_B,
                                                                                                                    device=device,
                                                                                                                    num_classes=num_classes)
        
        if scheduler_B is not None:
            scheduler_B.step(val_f1)

        print(
          f"{sr_}Epoch: {epoch+1}\n"
          f"\t|train_loss: {train_loss:.4f}\n"
          f"\t|train_acc: {train_acc:.4f} | "
          f"train_f1: {train_f1:.4f} | "
          f"train_auroc: {train_auroc:.4f} | "
          f"train_precision: {train_precision:.4f} | "
          f"train_recall: {train_recall:.4f}\n"
          f"\t|val_loss: {val_loss:.4f} \n"
          f"\t|val_acc: {val_acc:.4f} | "
          f"val_f1: {val_f1:.4f} | "
          f"val_auroc: {val_auroc:.4f} | "
          f"val_precision: {val_precision:.4f} | "
          f"val_recall: {val_recall:.4f}"
        )
        
        # ---- best CL by val_f1 ----
        if val_f1 > best_f1:
            print(f"{c_}CL Val F1 Improved ({best_f1:0.4f} ---> {val_f1:0.4f})")
            best_f1 = float(val_f1)
            best_epoch_cl = epoch + 1

            best_CL_state = copy.deepcopy(classifier.state_dict())

            if save_models:
                CL_save_filepath = f"CL_{epochs}_epochs_{fold}_fold.pth"
                save_model(
                    model_state_dict=best_CL_state,
                    target_dir=model_weights_dir,
                    model_name=CL_save_filepath,
                )
            
            print(f"{sr_}")

        # Write the results to CSV file
        with csv_file_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            row = {
                "Epoch": epoch + 1,
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
                "val_recall": val_recall
            }
            w.writerow(row)

        # ---- LR change print (tolerant) ----
        if scheduler_B is not None:
            curr = float(scheduler_B.get_last_lr()[-1])
            if not math.isclose(curr, prev_cl_lr, rel_tol=0.0, abs_tol=1e-12):
                print(f"{g_}CL learning rate changed: {prev_cl_lr} -> {curr}{sr_}")
                prev_cl_lr = curr

        # Check early stopping conditions
        early_stopping_f1(val_f1)
        
        if early_stopping_f1.early_stop:
            print(f"{w_}Early stopping at epoch {epoch+1}{sr_}")
            break


    # ---- IMPORTANT: test BEST weights ----
    classifier.load_state_dict(best_CL_state)

    test_loss, test_acc, test_acc_mca, test_f1, test_f1_mca, test_auroc, test_auroc_mca, test_precision, test_recall = test_step(
        AE=AE,
        classifier=classifier,
        dataloader_B=test_dataloader_B,
        loss_fn_B=loss_fn_B,
        device=device,
        num_classes=num_classes,
    )

    fold_results["test_loss"].append(test_loss)
    fold_results["test_acc"].append(test_acc)
    fold_results["test_acc_mca"].append(test_acc_mca)
    fold_results["test_f1"].append(test_f1)
    fold_results["test_f1_mca"].append(test_f1_mca)
    fold_results["test_auroc"].append(test_auroc)
    fold_results["test_auroc_mca"].append(test_auroc_mca)
    fold_results["test_precision"].append(test_precision)
    fold_results["test_recall"].append(test_recall)

        # Write the results to CSV file
    with test_file_path.open("a", newline="") as f:
        tst_csv_writer = csv.DictWriter(f, fieldnames=tst_fieldnames)
        row = {
            "Fold": fold,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_acc_mca": test_acc_mca,
            "test_f1": test_f1,
            "test_f1_mca": test_f1_mca,
            "test_auroc": test_auroc,
            "test_auroc_mca": test_auroc_mca,
            "test_precision": test_precision,
            "test_recall": test_recall
        }
        tst_csv_writer.writerow(row)

    print(f"{c_}[INFO] Final Results:")
    print(
          f"Best CL Epoch: {best_epoch_cl}\n"
          f"\t|Best CL F1: {best_f1:.4f}"
        )
    
    print(f"{c_}[INFO] Final Test Results:")
    for metric, values in fold_results.items():
        if "mca" not in metric:
            print(f"{metric}: {values[-1]:.4f}")
          
    print(f"{sr_}")
