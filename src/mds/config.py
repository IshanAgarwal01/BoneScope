"""YAML config loading into typed dataclasses.

Every hyperparameter lives in a YAML file so a run can be reproduced from its
saved config alone. Configs can `inherit:` a parent file and override fields,
and the CLI can override single values with `--set section.key=value`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

VALID_LOSSES = {"ce", "cb_ce", "focal"}
VALID_SAMPLERS = {"none", "sqrt_inverse"}
VALID_SPLIT_TYPES = {"patient", "slide", "proxy", "stratified"}


@dataclass
class DataConfig:
    root: str = "data/raw"
    splits_csv: str = "splits/splits.csv"
    split_type: str = "proxy"
    img_size: int = 224
    num_workers: int = 4
    stain_norm: bool = False
    sampler: str = "none"


@dataclass
class ModelConfig:
    name: str = "convnext_tiny.fb_in22k_ft_in1k"
    pretrained: bool = True
    drop_path: float = 0.1
    linear_probe_only: bool = False


@dataclass
class LossConfig:
    name: str = "ce"
    label_smoothing: float = 0.1
    cb_beta: float = 0.999
    focal_gamma: float = 2.0


@dataclass
class TrainConfig:
    batch_size: int = 64
    probe_epochs: int = 2
    probe_lr: float = 1e-3
    epochs: int = 30
    warmup_epochs: int = 1
    lr_head: float = 1e-3
    lr_last: float = 3e-4
    lr_middle: float = 1e-4
    lr_early: float = 3e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    patience: int = 5
    amp: bool = True
    device: str = "auto"


@dataclass
class OutputConfig:
    dir: str = "runs"
    run_name: str | None = None
    wandb: bool = False


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Fail fast on typos in categorical options."""
        if self.loss.name not in VALID_LOSSES:
            raise ValueError(f"loss.name must be one of {VALID_LOSSES}, got {self.loss.name!r}")
        if self.data.sampler not in VALID_SAMPLERS:
            raise ValueError(f"data.sampler must be one of {VALID_SAMPLERS}")
        if self.data.split_type not in VALID_SPLIT_TYPES:
            raise ValueError(f"data.split_type must be one of {VALID_SPLIT_TYPES}")
        if self.data.sampler != "none" and self.loss.name == "cb_ce":
            # Re-weighting twice (sampler + loss) over-corrects for imbalance;
            # the experiment plan changes one thing at a time.
            raise ValueError("Use either a weighted loss or a balanced sampler, not both")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_with_inheritance(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return _deep_merge(_read_with_inheritance(path.parent / parent), raw)


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value with YAML rules (true, null, 3, 0.5 ...).

    PyYAML follows YAML 1.1, which reads `1e-3` (no dot) as a string, so
    numeric-looking strings are converted explicitly.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply `section.key=value` strings on top of a raw config dict."""
    out = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        node = out
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _parse_scalar(value)
    return out


def _build(cls: type, values: dict, where: str) -> Any:
    types = {f.name: f.type for f in fields(cls)}
    unknown = set(values) - set(types)
    if unknown:
        raise KeyError(f"Unknown config keys in {where}: {sorted(unknown)}")
    # YAML 1.1 reads `1e-3` as a string; coerce declared float fields.
    coerced = {
        k: float(v) if types[k] == "float" and isinstance(v, int | str) else v
        for k, v in values.items()
    }
    return cls(**coerced)


def from_dict(raw: dict) -> Config:
    """Build a validated Config, rejecting unknown keys (catches typos)."""
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "loss": LossConfig,
        "train": TrainConfig,
        "output": OutputConfig,
    }
    top_unknown = set(raw) - set(sections) - {"seed"}
    if top_unknown:
        raise KeyError(f"Unknown top-level config keys: {sorted(top_unknown)}")
    kwargs: dict[str, Any] = {"seed": raw.get("seed", 42)}
    for name, cls in sections.items():
        kwargs[name] = _build(cls, raw.get(name) or {}, name)
    cfg = Config(**kwargs)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (following `inherit:`) and apply CLI overrides."""
    raw = _read_with_inheritance(Path(path))
    raw = apply_overrides(raw, overrides or [])
    return from_dict(raw)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
