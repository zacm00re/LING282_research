"""Load configs/default.yaml and apply command-line overrides.

Usage in any script:
    from carti_asr.config import load_config
    cfg = load_config("configs/default.yaml", overrides=["model.name=openai/whisper-base"])
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml


def _set_nested(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)  # "1e-5" -> float, "true" -> bool, etc.


def load_config(path: str | Path = "configs/default.yaml", overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        _set_nested(cfg, key, value)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def add_common_args(parser):
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set model.name=openai/whisper-base")
    return parser
