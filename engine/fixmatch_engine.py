from itertools import cycle
import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from .metrics import make_metrics, update_metrics, compute_and_reset_metrics
from .utils import save_model
from .utils import EarlyStoppingF1
from pathlib import Path
import torch
import os 
from tqdm.auto import tqdm
import numpy as np
import csv
from pathlib import Path
import copy 
import math
from colorama import Fore, Back, Style
# Constants and Configurations
c_ = Fore.GREEN # improvements
w_ = Fore.YELLOW # warning
e_ = Fore.RED # error
i_ = Fore.BLUE # info
g_ = Fore.CYAN # changes
b_ = Fore.MAGENTA # begining
sr_ = Style.RESET_ALL


def supervised_loss_fn(logits, targets, class_weights=None):
    if class_weights is not None:
        return F.cross_entropy(logits, targets, weight=class_weights)
    return F.cross_entropy(logits, targets)


def fixmatch_unsupervised_loss(logits_weak, logits_strong, tau=0.95):
    """FixMatch unsupervised loss.

    Args:
        logits_weak: logits on weakly augmented unlabeled images
        logits_strong: logits on strongly augmented unlabeled images
        tau: confidence threshold
    """
    probs_weak = torch.softmax(logits_weak.detach(), dim=1)
    max_probs, pseudo_labels = torch.max(probs_weak, dim=1)

    mask = (max_probs >= tau).float()

    per_sample_loss = F.cross_entropy(logits_strong, pseudo_labels, reduction="none")

    if mask.sum().item() > 0:
        unsup_loss = (per_sample_loss * mask).sum() / mask.sum()
    else:
        unsup_loss = torch.zeros((), device=logits_strong.device, dtype=logits_strong.dtype)

    stats = {
        "mask_ratio": mask.mean().item(),
        "num_kept": int(mask.sum().item()),
        "num_total": int(mask.numel()),
    }
    return unsup_loss, stats


def _apply_transform_batch(transform, batch):
    """Apply a torchvision-style transform to a batch tensor.

    Supports two common cases:
      - transform can take a full batch tensor (rare)
      - transform can take a single image tensor [C,H,W]
    """
    if transform is None:
        return batch

    try:
        out = transform(batch)
        if isinstance(out, torch.Tensor):
            return out
    except Exception:
        pass

    # Fall back to per-sample application.
    if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
        raise ValueError("Expected a 4D tensor batch [B,C,H,W] for per-sample transforms.")

    transformed = [transform(img) for img in batch]
    return torch.stack(transformed, dim=0)


@torch.no_grad()
def evaluate_classifier(model, loader, num_classes, device, average="macro"):
    model.eval()

    metrics = make_metrics(num_classes=num_classes, device=device)

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[:2]
        else:
            raise ValueError("Eval loader must return (images, labels).")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        update_metrics(metrics, logits, y)

    test_acc, test_acc_mca, test_f1, test_f1_mca, test_auroc, test_auroc_mca, test_precision, test_recall = compute_and_reset_metrics(metrics)

    return (
        test_acc,
        test_acc_mca,
        test_f1,
        test_f1_mca,
        test_auroc,
        test_auroc_mca,
        test_precision,
        test_recall,
    )


def train_one_epoch_fixmatch(
    model,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    device,
    weak_transform,
    strong_transform,
    num_classes,
    lambda_u=1.0,
    tau=0.95,
    class_weights=None,
    scheduler=None
):
    model.train()

    metrics = make_metrics(num_classes=num_classes, device=device)

    total_loss = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    total_mask_ratio = 0.0
    num_steps = 0

    unlabeled_iter = cycle(unlabeled_loader)

    for batch_l in labeled_loader:
        # ---- labeled batch ----
        if isinstance(batch_l, (list, tuple)) and len(batch_l) >= 2:
            x_l, y_l = batch_l[:2]
        else:
            raise ValueError("Labeled loader must return (images, labels).")

        x_l = x_l.to(device, non_blocking=True)
        y_l = y_l.to(device, non_blocking=True)

        # ---- unlabeled batch ----
        batch_u = next(unlabeled_iter)
        if isinstance(batch_u, (list, tuple)) and len(batch_u) >= 2:
            x_u_w, x_u_s = batch_u[:2]
            x_u_w = x_u_w.to(device, non_blocking=True)
            x_u_s = x_u_s.to(device, non_blocking=True)
        else:
            x_u = batch_u[0] if isinstance(batch_u, (list, tuple)) else batch_u
            x_u = x_u.to(device, non_blocking=True)

            # apply weak/strong transforms (may be Identity)
            x_u_w = _apply_transform_batch(weak_transform, x_u)
            x_u_s = _apply_transform_batch(strong_transform, x_u)

        optimizer.zero_grad()

        # supervised branch
        logits_l = model(x_l)
        sup_loss = supervised_loss_fn(logits_l, y_l, class_weights=class_weights)

        # unlabeled branch
        logits_u_w = model(x_u_w)
        logits_u_s = model(x_u_s)

        unsup_loss, stats = fixmatch_unsupervised_loss(
            logits_u_w,
            logits_u_s,
            tau=tau
        )

        loss = sup_loss + lambda_u * unsup_loss
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        total_mask_ratio += stats["mask_ratio"]
        num_steps += 1

        update_metrics(metrics, logits_l, y_l)


    train_loss = total_loss / max(num_steps, 1)
    sup_loss = total_sup_loss / max(num_steps, 1)
    unsup_loss = total_unsup_loss / max(num_steps, 1)
    mask_ratio = total_mask_ratio / max(num_steps, 1)


    train_acc, train_acc_mca, train_f1, train_f1_mca, train_auroc, train_auroc_mca, train_precision, train_recall = \
        compute_and_reset_metrics(metrics)


    return (
        train_loss,
        sup_loss,
        unsup_loss,
        mask_ratio,
        train_acc,
        train_acc_mca,
        train_f1,
        train_f1_mca,
        train_auroc,
        train_auroc_mca,
        train_precision,
        train_recall,
    )


def fit_fixmatch(
    model,
    train_labeled_loader,
    train_unlabeled_loader,
    val_loader,
    test_loader,
    optimizer,
    device,
    weak_transform,
    strong_transform,
    training_progress_dir,
    model_weights_dir,
    fold_results,
    fold,
    num_classes=6,
    num_epochs=100,
    lambda_u=1.0,
    tau=0.95,
    class_weights=None,
    scheduler=None,
    monitor="f1_macro",
    patience=20,
):

# ---- normalize paths ----
    training_progress_dir = Path(training_progress_dir)
    model_weights_dir = Path(model_weights_dir)
    training_progress_dir.mkdir(parents=True, exist_ok=True)
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = model_weights_dir / f"{fold}.csv"
    test_file_path = model_weights_dir / f"{fold}_test.csv"

    fieldnames = [
        "Epoch",
        "train_loss",
        "sup_loss",
        "unsup_loss",
        "mask_ratio",
        "train_acc",
        "train_acc_mca",
        "train_f1",
        "train_f1_mca",
        "train_auroc",
        "train_auroc_mca",
        "train_precision",
        "train_recall",
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


    best_state = None
     # ---- best trackers ----
    best_f1 = -np.inf
    best_epoch_cl = -1

    early_stopping_f1 = EarlyStoppingF1(patience=patience, min_delta=0.001)

    model.to(device)

    for epoch in tqdm(range(1, num_epochs + 1)):
        (
            train_loss,
            sup_loss,
            unsup_loss,
            mask_ratio,
            train_acc,
            train_acc_mca,
            train_f1,
            train_f1_mca,
            train_auroc,
            train_auroc_mca,
            train_precision,
            train_recall,
        ) = train_one_epoch_fixmatch(
            model=model,
            labeled_loader=train_labeled_loader,
            unlabeled_loader=train_unlabeled_loader,
            optimizer=optimizer,
            device=device,
            weak_transform=weak_transform,
            strong_transform=strong_transform,
            num_classes=num_classes,
            lambda_u=lambda_u,
            tau=tau,
            class_weights=class_weights,
            scheduler=scheduler
        )

        (
            val_acc,
            val_acc_mca,
            val_f1,
            val_f1_mca,
            val_auroc,
            val_auroc_mca,
            val_precision,
            val_recall,
        ) = evaluate_classifier(model, val_loader, num_classes, device=device)

        # ---- logging ----
        print(
            f"{sr_}Epoch: {epoch}\n"
            f"\t|train_acc: {train_acc:.4f} | train_f1: {train_f1:.4f} | train_auroc: {train_auroc:.4f} | "
            f"train_precision: {train_precision:.4f} | train_recall: {train_recall:.4f}\n"
            f"\t|val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | val_auroc: {val_auroc:.4f} | "
            f"val_precision: {val_precision:.4f} | val_recall: {val_recall:.4f}"
        )

        for i, f1 in enumerate(train_f1_mca):
            print(f"Class {i} Train F1: {float(f1):.4f}")
        for i, f1 in enumerate(val_f1_mca):
            print(f"Class {i} Validation F1: {float(f1):.4f}")

                # ---- best CL by val_f1 ----
        if val_f1 > best_f1:
            print(f"{c_}CL Val F1 Improved ({best_f1:0.4f} ---> {val_f1:0.4f})")
            best_f1 = float(val_f1)
            best_epoch_cl = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"{sr_}")
        
        # ---- append epoch row ----
        with csv_file_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(
                {
                    "Epoch": epoch,
                    "train_loss": train_loss,
                    "sup_loss": sup_loss,
                    "unsup_loss": unsup_loss,
                    "mask_ratio": mask_ratio,
                    "train_acc": train_acc,
                    "train_acc_mca": train_acc_mca,
                    "train_f1": train_f1,
                    "train_f1_mca": train_f1_mca,
                    "train_auroc": train_auroc,
                    "train_auroc_mca": train_auroc_mca,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
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

        # ---- early stopping by val_f1 ----
        early_stopping_f1(val_f1)
        if early_stopping_f1.early_stop:
            print(f"{w_}Early stopping at epoch {epoch+1}{sr_}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    (
        test_acc,
        test_acc_mca,
        test_f1,
        test_f1_mca,
        test_auroc,
        test_auroc_mca,
        test_precision,
        test_recall,
    )  = evaluate_classifier(model, test_loader, num_classes, device=device)


    # record fold results
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
    print(f"Best Epoch: {best_epoch_cl}\n\t|Best F1: {best_f1:.4f}")

    print(f"{c_}[INFO] Final Test Results:")
    for metric, values in fold_results.items():
        if "mca" in metric:
            continue
        if not values:
            print(f"{metric}: n/a")
            continue
        print(f"{metric}: {values[-1]:.4f}")
    print(f"{sr_}")

