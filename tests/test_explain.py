import numpy as np
import pytest
import timm
import torch

from mds.explain import CamExplainer, border_heat_fraction, overlay, target_layers_for


@pytest.mark.parametrize(
    "name,kwargs,size",
    [
        ("convnext_atto", {}, 64),
        ("efficientnet_b0", {}, 64),
        ("vit_tiny_patch16_224", {"img_size": 64}, 64),
        ("swin_tiny_patch4_window7_224", {}, 224),
    ],
)
def test_gradcam_runs_per_architecture(name, kwargs, size):
    torch.manual_seed(0)
    model = timm.create_model(name, pretrained=False, num_classes=3, **kwargs)
    explainer = CamExplainer(model)
    cam = explainer(torch.randn(2, 3, size, size), class_idx=1)
    explainer.close()
    assert cam.shape == (2, size, size)
    assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        target_layers_for(torch.nn.Linear(3, 3))


def test_border_heat_fraction():
    centre = np.zeros((100, 100))
    centre[40:60, 40:60] = 1
    assert border_heat_fraction(centre) == pytest.approx(0.0)
    assert border_heat_fraction(np.ones((100, 100))) == pytest.approx(1 - 0.7**2, abs=0.01)
    edge = np.zeros((100, 100))
    edge[:5] = 1
    assert border_heat_fraction(edge) == pytest.approx(1.0)
    assert np.isnan(border_heat_fraction(np.zeros((10, 10))))


def test_overlay_resizes_and_returns_uint8():
    out = overlay(np.random.rand(64, 64, 3), np.random.rand(8, 8))
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
