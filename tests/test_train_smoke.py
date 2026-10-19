"""1-2 epoch CPU training runs on synthetic images with tiny random-init models.

Proves the full pipeline (splits -> loaders -> probe -> fine-tune -> checkpoint
-> test predictions) runs end to end. The numbers mean nothing.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from mds.data import index_images
from mds.splits import make_folds, make_inner_val
from mds.train import main


@pytest.fixture
def splits_csv(synth_root, tmp_path):
    df = index_images(synth_root)
    df["group"] = df["path"].str.extract(r"/(P\d+)_")[0]
    splits = pd.concat(
        [make_folds(df, "patient"), make_folds(df.drop(columns="group"), "stratified")],
        ignore_index=True,
    )
    out = tmp_path / "splits" / "splits.csv"
    out.parent.mkdir()
    splits.to_csv(out, index=False)
    make_inner_val(splits).to_csv(out.with_name("splits_inner_val.csv"), index=False)
    return out


CASES = {
    "convnext_ce": ["model.name=convnext_atto", "loss.name=ce"],
    "effnet_focal_sampler": [
        "model.name=efficientnet_b0",
        "loss.name=focal",
        "data.sampler=sqrt_inverse",
    ],
    "vit_probe_cb": [
        "model.name=vit_tiny_patch16_224",
        "model.linear_probe_only=true",
        "loss.name=cb_ce",
        "train.epochs=0",
        "train.probe_epochs=2",
    ],
}


@pytest.mark.parametrize("case", list(CASES))
def test_one_epoch_cpu(case, synth_root, splits_csv, tmp_path):
    overrides = [
        f"data.root={synth_root}",
        f"data.splits_csv={splits_csv}",
        "data.split_type=patient",
        "data.img_size=32",
        "data.num_workers=0",
        "model.pretrained=false",
        "train.batch_size=8",
        "train.probe_epochs=1",
        "train.epochs=1",
        "train.device=cpu",
        f"output.dir={tmp_path / 'runs'}",
        f"output.run_name={case}",
        *CASES[case],
    ]
    cfg_path = str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    metrics = main(["--config", cfg_path, "--fold", "0", "--set", *overrides])

    run = tmp_path / "runs" / case / "patient" / "fold0"
    for name in [
        "best.pt",
        "log.csv",
        "classes.json",
        "config.yaml",
        "metrics.json",
        "test_preds.npz",
        "val_preds.npz",
    ]:
        assert (run / name).exists(), name
    assert json.loads((run / "classes.json").read_text()) == sorted(
        ["Blast", "Erythroblast", "Neutrophil", "Rare_cell"]
    )
    log = pd.read_csv(run / "log.csv")
    assert len(log) == 2 and set(log.columns) >= {"val_macro_f1", "val_top3", "lr_head"}
    assert 0.0 <= metrics["macro_f1"] <= 1.0 and math.isfinite(metrics["nll"])
    assert metrics["split_caveat"] == "patient-level split"
