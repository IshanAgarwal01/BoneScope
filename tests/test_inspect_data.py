import importlib.util
from pathlib import Path

import numpy as np
from conftest import SYNTH_CLASSES

from mds.data import (
    border_color_stats,
    build_proxy_features,
    cluster_class_purity,
    cluster_proxy_groups,
    filename_skeleton,
    index_images,
    keyword_hits,
    token_position_stats,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_index_reads_classes_from_folders(synth_root):
    df = index_images(synth_root)
    assert df["label_name"].value_counts().to_dict() == SYNTH_CLASSES
    assert not df["path"].str.startswith("/").any()  # relative, portable paths


def test_filename_skeleton():
    assert filename_skeleton("P012_S3_cell0045") == "AD_AD_AD"


def test_token_stats_find_patient_like_position(synth_root):
    stems = [Path(p).stem for p in index_images(synth_root)["path"]]
    stats = {t.position: t for t in token_position_stats(stems)}
    assert stats[0].n_unique == 6  # P000..P005 -> patient-ID candidate
    assert stats[2].n_unique == len(stems)  # running index
    assert keyword_hits(stems)["slide"] == len(stems)


def test_border_stats_capture_background_not_centre():
    arr = np.full((40, 40, 3), 200, np.uint8)
    arr[10:30, 10:30] = 50
    from PIL import Image

    stats = border_color_stats(Image.fromarray(arr))
    assert np.allclose(stats[:3], 200) and np.allclose(stats[3:], 0)


def test_proxy_groups_recover_background_tints():
    rng = np.random.default_rng(0)
    groups_true = np.repeat(np.arange(4), 25)
    colors = np.hstack([rng.normal(groups_true[:, None] * 30, 1, (100, 3)), np.zeros((100, 3))])
    emb = rng.normal(size=(100, 16))
    feats = build_proxy_features(emb, colors, n_pca=4, color_weight=4.0)
    pred = cluster_proxy_groups(feats, n_groups=4)
    # each true group should map to exactly one cluster
    assert cluster_class_purity(pred, groups_true) == 1.0


def test_inspect_report_end_to_end(synth_root, tmp_path):
    inspect = _load_script("inspect_data")
    text = inspect.build_report(synth_root, tmp_path / "reports", grid=True)
    assert "4 classes, 41 images" in text
    assert "Rare_cell (3)" in text  # rare class flagged
    assert (tmp_path / "reports" / "figures" / "class_grid.png").exists()
    assert (tmp_path / "reports" / "class_counts.csv").exists()
