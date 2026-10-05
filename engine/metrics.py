import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)

def make_metrics(num_classes: int, device: torch.device):
    """
    Returns a dict of torchmetrics objects moved to device.
    Uses:
      - class indices for acc/f1/precision/recall
      - probabilities for AUROC
    """
    metrics = {
        "acc": MulticlassAccuracy(num_classes=num_classes, average="macro").to(device),
        "acc_mca": MulticlassAccuracy(num_classes=num_classes, average=None).to(device),

        "f1": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "f1_mca": MulticlassF1Score(num_classes=num_classes, average=None).to(device),

        "auroc": MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None).to(device),
        "auroc_mca": MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None).to(device),

        "precision": MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro").to(device),
    }
    return metrics


def update_metrics(metrics: dict, logits: torch.Tensor, labels: torch.Tensor):
    """
    Updates all metrics consistently.
    - logits: [B, C]
    - labels: [B]
    """
    preds = torch.argmax(logits, dim=1)

    metrics["acc"].update(preds, labels)
    metrics["acc_mca"].update(preds, labels)

    metrics["f1"].update(preds, labels)
    metrics["f1_mca"].update(preds, labels)

    metrics["precision"].update(preds, labels)
    metrics["recall"].update(preds, labels)

    probs = torch.softmax(logits, dim=1)
    metrics["auroc"].update(probs, labels)
    metrics["auroc_mca"].update(probs, labels)


def compute_and_reset_metrics(metrics: dict):
    """
    Computes metrics -> numpy, then resets them.
    Returns:
      acc, acc_mca, f1, f1_mca, auroc, auroc_mca, precision, recall
    """
    acc = float(metrics["acc"].compute().detach().cpu().numpy())
    acc_mca = metrics["acc_mca"].compute().detach().cpu().numpy()

    f1 = float(metrics["f1"].compute().detach().cpu().numpy())
    f1_mca = metrics["f1_mca"].compute().detach().cpu().numpy()

    auroc = float(metrics["auroc"].compute().detach().cpu().numpy())
    auroc_mca = metrics["auroc_mca"].compute().detach().cpu().numpy()

    precision = float(metrics["precision"].compute().detach().cpu().numpy())
    recall = float(metrics["recall"].compute().detach().cpu().numpy())

    for m in metrics.values():
        m.reset()

    return acc, acc_mca, f1, f1_mca, auroc, auroc_mca, precision, recall
