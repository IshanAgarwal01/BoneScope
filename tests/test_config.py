from pathlib import Path

import pytest

from mds.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["base", "convnext_tiny", "efficientnet_b3", "dinov2_probe"])
def test_all_configs_load(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg.seed == 42
    assert cfg.loss.label_smoothing == 0.1


def test_inheritance_overrides_only_what_changes():
    cfg = load_config(CONFIGS / "efficientnet_b3.yaml")
    assert cfg.model.name == "efficientnet_b3.ra2_in1k"
    assert cfg.data.img_size == 288
    assert cfg.train.weight_decay == 0.05  # inherited from base


def test_cli_overrides_parse_yaml_scalars():
    overrides = ["train.epochs=1", "train.lr_head=5e-4", "loss.name=focal"]
    cfg = load_config(CONFIGS / "base.yaml", overrides)
    assert cfg.train.epochs == 1
    assert cfg.train.lr_head == pytest.approx(5e-4)
    assert cfg.loss.name == "focal"


def test_typos_are_rejected():
    with pytest.raises(KeyError):
        load_config(CONFIGS / "base.yaml", ["train.epoch=1"])
    with pytest.raises(ValueError):
        load_config(CONFIGS / "base.yaml", ["loss.name=cross_entropy"])


def test_sampler_and_weighted_loss_not_combined():
    with pytest.raises(ValueError):
        load_config(CONFIGS / "base.yaml", ["data.sampler=sqrt_inverse", "loss.name=cb_ce"])


def test_yaml_scientific_notation_without_dot(tmp_path):
    (tmp_path / "c.yaml").write_text("train:\n  lr_head: 1e-3\n")
    assert load_config(tmp_path / "c.yaml").train.lr_head == pytest.approx(1e-3)
