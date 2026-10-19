"""Grad-CAM: where in the crop did the model look?

Grad-CAM weights the last convolutional feature maps by how much they push
the score of a chosen class, giving a coarse heat map. In this project it is a
sanity check, not a proof of reasoning: heat on the nucleus/cytoplasm is
reassuring; heat on the background or a neighbouring cell is "shortcut"
evidence (e.g. the model recognising a slide's stain instead of the cell).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn


def _swin_reshape(t: torch.Tensor) -> torch.Tensor:
    # timm Swin blocks are channels-last: (B, H, W, C) -> (B, C, H, W)
    return t.permute(0, 3, 1, 2)


def _vit_reshape(model: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    n_prefix = getattr(model, "num_prefix_tokens", 1)

    def reshape(t: torch.Tensor) -> torch.Tensor:
        tokens = t[:, n_prefix:, :]
        side = int(round(tokens.shape[1] ** 0.5))
        return tokens.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)

    return reshape


def target_layers_for(model: nn.Module) -> tuple[list[nn.Module], Callable | None]:
    """Layer(s) to hook and an optional reshape for transformer outputs.

    ConvNeXt: the last block of the last stage (highest-level spatial features).
    EfficientNet: conv_head, the final 1x1 conv before pooling.
    Swin: last block's norm1 (needs channels-last -> channels-first reshape).
    ViT/DINOv2: last block's norm1 (tokens reshaped to a grid; prefix tokens dropped).
    """
    if hasattr(model, "stages") and hasattr(model.stages[-1], "blocks"):  # ConvNeXt
        return [model.stages[-1].blocks[-1]], None
    if hasattr(model, "conv_head"):  # EfficientNet
        return [model.conv_head], None
    if hasattr(model, "layers") and hasattr(model.layers[-1], "blocks"):  # Swin
        return [model.layers[-1].blocks[-1].norm1], _swin_reshape
    if hasattr(model, "blocks") and hasattr(model.blocks[-1], "norm1"):  # ViT
        return [model.blocks[-1].norm1], _vit_reshape(model)
    raise ValueError(f"No Grad-CAM target layer rule for {type(model).__name__}")


class CamExplainer:
    """Reusable Grad-CAM wrapper (hooks are registered once)."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
        layers, reshape = target_layers_for(model)
        self.cam = GradCAM(model=model, target_layers=layers, reshape_transform=reshape)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Heat map(s) in [0, 1], shape (B, H, W). Default target: predicted class."""
        if x.ndim == 3:
            x = x[None]
        targets = None if class_idx is None else [ClassifierOutputTarget(class_idx)] * len(x)
        return self.cam(input_tensor=x, targets=targets)

    def close(self) -> None:
        self.cam.activations_and_grads.release()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heat map onto an RGB image (float in [0, 1] or uint8). Returns uint8."""
    img = rgb.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if cam.shape != img.shape[:2]:
        cam = np.asarray(
            torch.nn.functional.interpolate(
                torch.as_tensor(cam)[None, None], size=img.shape[:2], mode="bilinear"
            )[0, 0]
        )
    return show_cam_on_image(img, cam, use_rgb=True, image_weight=1 - alpha)


def border_heat_fraction(cam: np.ndarray, border: float = 0.15) -> float:
    """Share of heat in the outer frame of the crop.

    The cell sits in the centre (the crop is its box + 10%), so the outer ~15%
    is background or neighbouring cells. The frame is ~51% of the area, so a
    uniform map scores ~0.5; well above that suggests a background shortcut.
    """
    h, w = cam.shape
    b_h, b_w = max(1, int(h * border)), max(1, int(w * border))
    total = float(cam.sum())
    if total <= 0:
        # An all-zero map (e.g. a 1x1 feature map at tiny input sizes) carries no
        # location information; NaN avoids reading it as "perfectly centred".
        return float("nan")
    inner = float(cam[b_h : h - b_h, b_w : w - b_w].sum())
    return 1.0 - inner / total


def denormalize(x: torch.Tensor, mean, std) -> np.ndarray:
    """Undo Normalize for display: (C,H,W) tensor -> (H,W,C) float in [0,1]."""
    m = torch.as_tensor(mean)[:, None, None]
    s = torch.as_tensor(std)[:, None, None]
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
