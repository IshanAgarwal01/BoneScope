"""App logic on a tiny random-weight bundle (no server is launched)."""

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mds.config import Config
from mds.models import create_model

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"
CLASSES = ["Blast", "Erythroblast", "Neutrophil", "Platelet"]


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("bonescope_app", APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bonescope_app"] = module  # dataclasses look the module up while loading
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(app, tmp_path):
    cfg = Config()
    cfg.model.name, cfg.model.pretrained, cfg.data.img_size = "convnext_atto", False, 64
    torch.manual_seed(0)
    model = create_model(cfg, len(CLASSES))
    path = tmp_path / "bundle.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "classes": CLASSES,
            "config": cfg.to_dict(),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "temperature": 1.5,
            "blast_classes": ["Blast"],
            "non_nucleated_classes": ["Platelet"],
            "split_caveat": "test",
        },
        path,
    )
    return app.load_bundle(path)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (50, 50, 3), dtype=np.uint8))


def test_single_prediction(app, bundle):
    top3, heat, status = app.predict_single(bundle, _img())
    assert len(top3) == 3 and set(top3) <= set(CLASSES)
    assert all(0 <= p <= 1 for p in top3.values())
    assert heat.shape == (64, 64, 3) and heat.dtype == np.uint8
    assert "Not for clinical use" in status
    # random 4-class model is near-uniform -> low-confidence banner
    assert "Low confidence" in status


def test_batch_prediction(app, bundle, tmp_path):
    zpath = tmp_path / "cells.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for i in range(12):
            buf = tmp_path / f"c{i}.png"
            _img(i).save(buf)
            z.write(buf, f"cells/c{i}.png")
        z.writestr("__MACOSX/._junk.png", b"junk")
        z.writestr("notes.txt", "ignored")
    table, summary = app.predict_batch(bundle, zpath)
    assert table["cells"].sum() == 12
    assert "Not for clinical use" in summary
    assert ("WHO 2022" in summary) or ("unavailable" in summary)
    assert "Only 12 cells" in summary


def test_zip_without_images_is_rejected(app, tmp_path):
    zpath = tmp_path / "empty.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("readme.txt", "no images")
    with pytest.raises(ValueError):
        app.read_zip_images(zpath)


def test_demo_builds_with_and_without_model(app, bundle):
    assert app.build_demo(bundle) is not None
    assert app.build_demo(None) is not None
