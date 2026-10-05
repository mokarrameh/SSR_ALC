from __future__ import annotations

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

from utils.config_utils import load_yaml_config, merge_yaml_into_defaults


class BaseOptions:

    def __init__(self) -> None:
        self.initialized = False
        self.parser = None
        self.isTrain = None

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Common / infra
        parser.add_argument("--config", type=str, default=None, help="Path to configs.yaml")
        parser.add_argument("--name", type=str, default=None, help="Experiment name")
        parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
        parser.add_argument("--num_workers", type=int, default=1)

        # Data paths
        parser.add_argument(
            "--data_path",
            type=str,
            default=None,
            help="Path to data directory containing labeled/ and unlabeled/",
        )
        parser.add_argument("--labeled_dir", type=str, default="labeled")
        parser.add_argument("--unlabeled_dir", type=str, default="unlabeled")
        parser.add_argument(
            "--split_csv",
            type=str,
            default=None,
            help="Split CSV filename or path (relative to <data_path>/<labeled_dir> unless absolute).",
        )
        parser.add_argument("--u_train_subdir", type=str, default="train")
        parser.add_argument("--u_val_subdir", type=str, default="val")
        parser.add_argument("--u_test_subdir", type=str, default="test")

        # Dataloaders
        parser.add_argument("--l_batch_size", type=int, default=32)
        parser.add_argument("--u_train_batch_size", type=int, default=256)
        parser.add_argument("--u_val_batch_size", type=int, default=64)
        parser.add_argument("--u_test_batch_size", type=int, default=64)

        # CV
        parser.add_argument("--num_folds", type=int, default=5)
        parser.add_argument("--folds", type=int, nargs="*", default=None,
                            help="Explicit folds, e.g. --folds 0 1. None => all folds")

        # External images
        parser.add_argument("--extra_img_dir", type=str, default=None)
        parser.add_argument("--extra_img_n", type=int, default=0)
        parser.add_argument("--extra_seed", type=int, default=123)

        # Model selection (you already loop AEs; keep list)
        parser.add_argument("--classifier_name", type=str, default="simpleNN")
        parser.add_argument("--aes", type=str, nargs="+", default=["convAE", "deepConvAE", "deepConvAEV1"],
                            help="List of AE variants to run")
        parser.add_argument("--save_models", action="store_true", help="Whether to save model weights after training")

        self.initialized = True
        return parser

    def gather_options(self) -> Tuple[argparse.Namespace, Dict[str, Any]]:
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        else:
            parser = self.parser

        # Phase 1: read --config
        opt_partial, _ = parser.parse_known_args()
        yaml_data = load_yaml_config(opt_partial.config) if opt_partial.config else {}

        # Merge YAML defaults into argparse defaults
        defaults = vars(opt_partial)
        section = "train" if self.isTrain else "test"
        merged_defaults = merge_yaml_into_defaults(defaults, yaml_data, section)

        # Phase 2: rebuild parser with merged defaults, parse fully (CLI overrides)
        parser2 = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser2 = self.initialize(parser2)
        parser2.set_defaults(**merged_defaults)

        opt = parser2.parse_args()
        self.parser = parser2
        return opt, yaml_data

    def print_options(self, opt: argparse.Namespace) -> str:
        msg = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            msg += f"{k:>25}: {v}\n"
        msg += "----------------- End -------------------\n"
        print(msg)
        return msg

    def save_options(self, opt: argparse.Namespace) -> Path:
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        expr_dir.mkdir(parents=True, exist_ok=True)

        txt_path = expr_dir / ("train_opt.txt" if self.isTrain else "test_opt.txt")
        txt_path.write_text(self.print_options(opt), encoding="utf-8")

        if yaml is not None:
            y_path = expr_dir / ("train_opt.yaml" if self.isTrain else "test_opt.yaml")
            with y_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(vars(opt), f, sort_keys=True)

        return expr_dir

    def parse(self) -> argparse.Namespace:
        opt, _ = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.name is None:
            raise ValueError("Missing required experiment name. Provide --name or set global.name in the YAML config.")

        if not opt.data_path:
            raise ValueError(
                "Missing required data path. Provide --data_path or set global.data_path in the YAML config."
            )

        if not opt.split_csv:
            raise ValueError(
                "Missing required split CSV. Provide --split_csv or set global.split_csv in the YAML config."
            )

        # resolve folds
        if opt.folds is None:
            opt.folds = list(range(opt.num_folds))

        # resolve device
        if opt.device == "auto":
            import torch
            opt.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.save_options(opt)
        return opt