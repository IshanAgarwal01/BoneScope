import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mds.config import LossConfig
from mds.losses import FocalLoss, build_loss, class_balanced_weights


def test_cb_weights_favour_rare_classes_and_average_one():
    w = class_balanced_weights(np.array([1000, 100, 10]), beta=0.999)
    assert w[2] > w[1] > w[0]
    assert w.mean().item() == pytest.approx(1.0, rel=1e-5)


def test_cb_weights_beta_zero_is_uniform():
    w = class_balanced_weights(np.array([1000, 10]), beta=0.0)
    assert torch.allclose(w, torch.ones(2))


def test_cb_weights_zero_count_class_gets_zero():
    w = class_balanced_weights(np.array([50, 0, 5]))
    assert w[1] == 0 and torch.isfinite(w).all()


def test_focal_gamma_zero_equals_ce():
    logits, y = torch.randn(16, 5), torch.randint(0, 5, (16,))
    assert FocalLoss(0.0)(logits, y).item() == pytest.approx(F.cross_entropy(logits, y).item())


def test_focal_downweights_easy_examples():
    easy = torch.tensor([[8.0, 0.0, 0.0]])
    hard = torch.tensor([[0.5, 0.0, 0.0]])
    y = torch.tensor([0])
    ratio_easy = FocalLoss(2.0)(easy, y) / F.cross_entropy(easy, y)
    ratio_hard = FocalLoss(2.0)(hard, y) / F.cross_entropy(hard, y)
    assert ratio_easy < 0.01 < ratio_hard


@pytest.mark.parametrize("name", ["ce", "cb_ce", "focal"])
def test_build_loss_backpropagates(name):
    loss_fn = build_loss(LossConfig(name=name), np.array([30, 5, 1]))
    logits = torch.randn(8, 3, requires_grad=True)
    loss = loss_fn(logits, torch.randint(0, 3, (8,)))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
