"""Shared fixtures: a tiny synthetic "marrow" folder (no real data needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SYNTH_CLASSES = {"Blast": 12, "Neutrophil": 16, "Erythroblast": 10, "Rare_cell": 3}


def make_synthetic_folder(root: Path, n_patients: int = 6, size: int = 40, seed: int = 0) -> Path:
    """Class folders with JPEGs named like `P003_slide2_00017.jpg`.

    Each fake patient gets its own background tint so colour-based proxy
    grouping has a real signal to find.
    """
    rng = np.random.default_rng(seed)
    tints = rng.integers(150, 250, size=(n_patients, 3))
    idx = 0
    for cls, n in SYNTH_CLASSES.items():
        (root / cls).mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            p = idx % n_patients
            arr = np.tile(tints[p], (size, size, 1)).astype(np.uint8)
            c = size // 2
            arr[c - 8 : c + 8, c - 8 : c + 8] = rng.integers(40, 120, size=3)
            name = f"P{p:03d}_slide{p % 3}_{idx:05d}.jpg"
            Image.fromarray(arr).save(root / cls / name, quality=90)
            idx += 1
    return root


@pytest.fixture
def synth_root(tmp_path: Path) -> Path:
    return make_synthetic_folder(tmp_path / "raw")
