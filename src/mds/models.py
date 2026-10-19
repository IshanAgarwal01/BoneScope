"""timm backbones, freezing for the linear probe, and discriminative LR groups.

"Discriminative learning rates": early layers detect generic edges/colours
that ImageNet pretraining already got right, so they get a small LR (3e-5);
the newly initialised head knows nothing yet and gets a large one (1e-3).
Without this, the first fine-tuning steps can wreck useful pretrained features.
"""

from __future__ import annotations

import re

import timm
import torch
from torch import nn

from mds.config import Config, TrainConfig

BUCKETS = ("early", "middle", "last", "head")
_STAGE_RX = re.compile(r"^(stages|blocks|layers)\.(\d+)\.")
_EARLY_PREFIXES = ("stem", "conv_stem", "bn1", "patch_embed", "cls_token", "pos_embed", "reg_token")
_HEAD_PREFIXES = ("head", "classifier", "fc_norm", "head_drop")


def create_model(cfg: Config, num_classes: int) -> nn.Module:
    """Build a timm model with a fresh `num_classes` head (class count from data)."""
    kwargs: dict = {"drop_path_rate": cfg.model.drop_path}
    if cfg.model.name.startswith("vit"):
        # DINOv2 defaults to 518 px; ViTs in timm resample position embeddings.
        kwargs["img_size"] = cfg.data.img_size
    return timm.create_model(
        cfg.model.name, pretrained=cfg.model.pretrained, num_classes=num_classes, **kwargs
    )


def model_mean_std(model: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Normalisation constants the backbone was pretrained with."""
    pc = getattr(model, "pretrained_cfg", {}) or {}
    return tuple(pc.get("mean", (0.485, 0.456, 0.406))), tuple(pc.get("std", (0.229, 0.224, 0.225)))


def classifier_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.get_classifier().parameters())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the final linear classifier."""
    head_ids = {id(p) for p in classifier_parameters(model)}
    for p in model.parameters():
        p.requires_grad = trainable or id(p) in head_ids


def set_probe_mode(model: nn.Module) -> None:
    """Linear-probe mode: frozen backbone in eval mode.

    eval() matters for EfficientNet: BatchNorm in train mode would keep updating
    its running statistics even with frozen weights.
    """
    model.eval()
    model.get_classifier().train()


def _stage_counts(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in names:
        m = _STAGE_RX.match(n)
        if m:
            counts[m.group(1)] = max(counts.get(m.group(1), 0), int(m.group(2)) + 1)
    return counts


def bucket_of(name: str, n_stages: dict[str, int]) -> str:
    """Assign a parameter to early / middle / last / head by its depth.

    With n stages: first n//2 -> early, last max(1, n//4) -> last, rest -> middle.
    ConvNeXt (4 stages): 0-1 early, 2 middle, 3 last.
    EfficientNet (7 blocks): 0-2 early, 3-5 middle, 6 + conv_head last.
    ViT (12 blocks): 0-5 early, 6-8 middle, 9-11 + final norm last.
    """
    if name.startswith(_HEAD_PREFIXES):
        return "head"
    if name.startswith(_EARLY_PREFIXES):
        return "early"
    m = _STAGE_RX.match(name)
    if m:
        i, n = int(m.group(2)), n_stages[m.group(1)]
        if i >= n - max(1, n // 4):
            return "last"
        return "early" if i < n // 2 else "middle"
    # post-stage layers (final norm, conv_head, bn2) sit just before the head
    return "last"


def param_groups(model: nn.Module, tc: TrainConfig) -> list[dict]:
    """AdamW parameter groups: one LR per depth bucket, and no weight decay on
    biases/norm weights/tokens (decaying those hurts and is standard to skip)."""
    lrs = {"early": tc.lr_early, "middle": tc.lr_middle, "last": tc.lr_last, "head": tc.lr_head}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_stages = _stage_counts([n for n, _ in named])
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for n, p in named:
        no_decay = p.ndim <= 1 or n.endswith(("cls_token", "pos_embed", "reg_token"))
        groups.setdefault((bucket_of(n, n_stages), no_decay), []).append(p)
    return [
        {
            "name": f"{bucket}{'_no_decay' if nd else ''}",
            "params": params,
            "lr": lrs[bucket],
            "weight_decay": 0.0 if nd else tc.weight_decay,
        }
        for (bucket, nd), params in sorted(groups.items())
    ]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a best.pt written by train.py (no pretrained download)."""
    from mds.config import from_dict

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = from_dict(ckpt["config"])
    cfg.model.pretrained = False
    model = create_model(cfg, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt
