import argparse
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Training options (CycleGAN style)."""

    def __init__(self):
        super().__init__()
        self.isTrain = True

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)

        # Training hypers
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--patience", type=int, default=20)
        parser.add_argument("--ae_lr", type=float, default=1e-4)
        parser.add_argument("--cl_lr", type=float, default=1e-4)

        # Loss weights
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=10.0)
        parser.add_argument("--gamma", type=float, default=5.0)

        # SMOTE (used inside engine_hpo)
        parser.add_argument("--smote_k_neighbors", type=int, default=2)
        parser.add_argument("--smote_seed", type=int, default=42)

        # hpo mode settings
        parser.add_argument("--trials", type=int, default=0, help="Number of HPO trials.") 
        

        return parser