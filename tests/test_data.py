import numpy as np
import pytest
import torch
from PIL import Image

from mds.data import (
    FixedScalePad,
    MarrowDataset,
    ReinhardStainNormalizer,
    build_transforms,
    index_images,
    sqrt_inverse_sampler,
)
from mds.splits import encode_labels


@pytest.fixture
def frame(synth_root):
    df = index_images(synth_root)
    df["label"], classes = encode_labels(df["label_name"])
    return df, classes


@pytest.mark.parametrize("train", [True, False])
def test_dataset_shapes_and_labels(synth_root, frame, train):
    df, classes = frame
    ds = MarrowDataset(df, synth_root, build_transforms(64, train=train))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert classes[y] == df.loc[0, "label_name"]
    assert len(ds) == len(df)


def test_eval_transform_is_deterministic(synth_root, frame):
    df, _ = frame
    ds = MarrowDataset(df, synth_root, build_transforms(64, train=False))
    assert torch.equal(ds[3][0], ds[3][0])


def _cell_crop(side: int, cell: int) -> Image.Image:
    """Light background with a dark square 'cell' in the middle."""
    img = Image.new("RGB", (side, side), (230, 220, 225))
    o = (side - cell) // 2
    img.paste((20, 20, 60), (o, o, o + cell, o + cell))
    return img


def _dark_width(img: Image.Image) -> int:
    return int((np.asarray(img).sum(-1) < 200).any(0).sum())


def test_fixed_scale_keeps_relative_size():
    pad = FixedScalePad(img_size=100, canvas_px=200)
    small, big = pad(_cell_crop(50, 20)), pad(_cell_crop(100, 40))
    assert small.size == big.size == (100, 100)
    # one global scale (0.5): the big cell stays twice as wide as the small one
    assert 1.8 < _dark_width(big) / _dark_width(small) < 2.3  # bilinear edges blur ±1px
    # whereas plain resizing would make both cells the same width
    resize = build_transforms(100, train=False).transforms[0]
    a, b = _dark_width(resize(_cell_crop(50, 20))), _dark_width(resize(_cell_crop(100, 40)))
    assert abs(a - b) <= 2


def test_sqrt_inverse_sampler_upweights_rare_classes():
    labels = [0] * 900 + [1] * 100
    sampler = sqrt_inverse_sampler(labels, seed=0)
    drawn = np.bincount([labels[i] for i in sampler], minlength=2)
    # expected share of class 1: 100/sqrt(100) / (900/sqrt(900) + 100/sqrt(100)) = 0.25
    assert 0.2 < drawn[1] / drawn.sum() < 0.3


def test_reinhard_runs_and_keeps_shape(synth_root, frame):
    df, _ = frame
    ref = Image.open(synth_root / df.loc[0, "path"])
    norm = ReinhardStainNormalizer(ref)
    out = norm(Image.open(synth_root / df.loc[5, "path"]))
    assert out.size == ref.size and out.mode == "RGB"
