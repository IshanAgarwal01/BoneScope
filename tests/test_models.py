import pytest
import timm

from mds.config import Config
from mds.models import _stage_counts, bucket_of, param_groups, set_backbone_trainable


@pytest.mark.parametrize(
    "name,expect",
    [
        (
            "convnext_atto",
            {
                "stem.0.weight": "early",
                "stages.1.blocks.0.conv_dw.weight": "early",
                "stages.2.blocks.0.conv_dw.weight": "middle",
                "stages.3.blocks.0.conv_dw.weight": "last",
                "head.fc.weight": "head",
            },
        ),
        (
            "efficientnet_b0",
            {
                "conv_stem.weight": "early",
                "blocks.2.0.conv_pw.weight": "early",
                "blocks.4.0.conv_pw.weight": "middle",
                "blocks.6.0.conv_pw.weight": "last",
                "conv_head.weight": "last",
                "classifier.weight": "head",
            },
        ),
    ],
)
def test_buckets(name, expect):
    model = timm.create_model(name, pretrained=False, num_classes=4)
    names = [n for n, _ in model.named_parameters()]
    counts = _stage_counts(names)
    for param, bucket in expect.items():
        assert param in names
        assert bucket_of(param, counts) == bucket


def test_param_groups_cover_every_param_once_with_lrs():
    cfg = Config()
    model = timm.create_model("convnext_atto", pretrained=False, num_classes=4)
    groups = param_groups(model, cfg.train)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    lr = {g["name"]: g["lr"] for g in groups}
    assert (
        lr["head"] == 1e-3 and lr["last"] == 3e-4 and lr["middle"] == 1e-4 and lr["early"] == 3e-5
    )
    assert all(g["weight_decay"] == 0 for g in groups if g["name"].endswith("no_decay"))


def test_freeze_leaves_only_classifier_trainable():
    model = timm.create_model("convnext_atto", pretrained=False, num_classes=4)
    set_backbone_trainable(model, False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["head.fc.weight", "head.fc.bias"]
