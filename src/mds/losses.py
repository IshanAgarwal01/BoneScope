"""Loss functions for heavily imbalanced cell classes.

Some marrow cell types are 50x more common than others. Plain cross-entropy
then mostly learns the common ones, which macro-F1 (our primary metric,
averaging every class equally) punishes. Options, tried one at a time:
  ce     cross-entropy + label smoothing (baseline)
  cb_ce  class-balanced CE: weight classes by 1 / "effective number" of samples
  focal  down-weight easy, confidently-correct examples
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mds.config import LossConfig


def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Cui et al. (CVPR 2019) class-balanced weights.

    Effective number E_n = (1 - beta^n) / (1 - beta). Extra images of a class
    add diminishing information (many are near-duplicates), so weights grow
    more gently than plain 1/n. Weights are normalised to average 1 so the loss
    scale matches plain CE. Classes absent from training get weight 0.
    """
    counts = np.asarray(counts, dtype=np.float64)
    effective = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = np.where(counts > 0, 1.0 / np.maximum(effective, 1e-12), 0.0)
    present = counts > 0
    weights[present] *= present.sum() / weights[present].sum()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017): CE scaled by (1 - p_true)^gamma.

    A cell the model already gets right with p=0.95 contributes ~400x less
    than with plain CE (gamma=2), so training effort shifts to hard cases,
    e.g. dysplastic vs normal granulocytes. Label smoothing is not combined
    with focal loss here, so the experiment changes one thing at a time.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits.float(), dim=1)
        log_pt = log_p.gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - log_pt.exp()) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_loss(cfg: LossConfig, class_counts: np.ndarray) -> nn.Module:
    """Create the training loss from config and training-set class counts."""
    if cfg.name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    if cfg.name == "cb_ce":
        weights = class_balanced_weights(class_counts, cfg.cb_beta)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    if cfg.name == "focal":
        return FocalLoss(gamma=cfg.focal_gamma)
    raise ValueError(f"Unknown loss {cfg.name!r}")
