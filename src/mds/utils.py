"""Seeding, device selection, and lightweight run logging."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable.

    Full determinism (`deterministic=True`) slows GPU training noticeably, so it
    is off by default; seeds alone make splits, sampling and init repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(preference: str = "auto") -> torch.device:
    """Pick cuda -> mps -> cpu when `preference` is "auto"."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled(device: torch.device, requested: bool) -> bool:
    """Mixed precision only on CUDA: MPS/CPU autocast is slower or less stable here."""
    return requested and device.type == "cuda"


def get_logger(name: str = "mds", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class CSVLogger:
    """Append one row per epoch to a CSV file (the minimum run log we always keep)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row)
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writerow(row)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
