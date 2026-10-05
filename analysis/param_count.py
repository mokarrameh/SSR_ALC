#!/usr/bin/env python3
"""Count trainable parameters per component (paper Table 16).

Example:
    python analysis/param_count.py --ae convAE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn as nn  # noqa: E402

from engine.models.autoencoder import ConvAutoencoder, DeepConvAutoencoder, DeepConvAutoencoderV1  # noqa: E402
from engine.models.classifier import Classifier512, Classifier1024, Classifier2048  # noqa: E402

MODELS = {
    "convAE": (ConvAutoencoder, Classifier512, 512),
    "deepConvAE": (DeepConvAutoencoder, Classifier1024, 1024),
    "deepConvAEV1": (DeepConvAutoencoderV1, Classifier2048, 2048),
}


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ae", choices=list(MODELS), default="convAE")
    p.add_argument("--num_classes", type=int, default=6)
    args = p.parse_args()

    ae_cls, clf_cls, dim = MODELS[args.ae]
    ae = ae_cls()
    clf = clf_cls(dim, args.num_classes, dropout_rate=0.4)
    parts = {
        "Encoder (conv blocks)": count(ae.encoder),
        "FC bottleneck (fc1 + fc2)": count(ae.fc1) + count(ae.fc2),
        "Decoder (ConvTranspose)": count(ae.decoder),
        "Classifier (MLP)": count(clf),
    }
    total = sum(parts.values())
    print(f"{args.ae} ({dim}-d latent)")
    for name, n in parts.items():
        print(f"  {name:28s} {n / 1e6:8.2f} M  {100 * n / total:6.2f} %")
    print(f"  {'Autoencoder total':28s} {count(ae) / 1e6:8.2f} M")
    print(f"  {'Autoencoder + classifier':28s} {total / 1e6:8.2f} M")

    try:
        import torchvision
        fm = torchvision.models.resnet18(weights=None)
        fm.fc = nn.Linear(fm.fc.in_features, args.num_classes)
        print(f"FixMatch ResNet-18: {count(fm) / 1e6:.2f} M")
    except Exception as exc:  # pragma: no cover
        print(f"(skipping ResNet-18 count: {exc})")


if __name__ == "__main__":
    main()
