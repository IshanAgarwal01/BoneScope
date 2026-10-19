import numpy as np
import pytest

from mds.calibration import expected_calibration_error, fit_temperature, softmax_np
from mds.evaluate import compute_metrics, lineage_accuracy, normalized_confusion, top_confusions


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 1])
    probs = np.eye(4)[y] * 0.97 + 0.01
    m = compute_metrics(y, probs)
    assert m["macro_f1"] == m["accuracy"] == m["balanced_acc"] == m["top3_acc"] == 1.0


def test_top3_counts_second_and_third_choice():
    probs = np.array([[0.5, 0.3, 0.15, 0.05], [0.5, 0.3, 0.15, 0.05]])
    m = compute_metrics(np.array([2, 3]), probs)
    assert m["top3_acc"] == 0.5 and m["accuracy"] == 0.0


def test_lineage_accuracy_forgives_same_family_errors():
    classes = ["myelocyte", "metamyelocyte", "lymphocyte"]
    lineage = {
        "myelocyte": "granulocytic",
        "metamyelocyte": "granulocytic",
        "lymphocyte": "lymphoid",
    }
    y, pred = np.array([0, 1, 2]), np.array([1, 0, 0])
    assert lineage_accuracy(y, pred, classes, lineage) == pytest.approx(2 / 3)


def test_confusion_rows_sum_to_one_and_top_confusions():
    y, pred = np.array([0, 0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 1, 2])
    cm = normalized_confusion(y, pred, 4)
    assert np.allclose(cm[:3].sum(1), 1) and cm[3].sum() == 0
    tc = top_confusions(y, pred, ["a", "b", "c", "d"])
    assert tc.iloc[0][["true", "predicted", "count"]].tolist() == ["a", "b", 2]


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, c = 4000, 5
    y = rng.integers(0, c, n)
    # a classifier that is right ~60% of the time but ~99% confident
    pred = np.where(rng.random(n) < 0.6, y, rng.integers(0, c, n))
    logits = np.eye(c)[pred] * 6.0 + rng.normal(0, 0.3, (n, c))
    before = expected_calibration_error(softmax_np(logits), y)
    t = fit_temperature(logits, y)
    after = expected_calibration_error(softmax_np(logits, t), y)
    assert t > 1.5 and after < before / 3
    # temperature never changes the predicted class
    assert (softmax_np(logits, t).argmax(1) == logits.argmax(1)).all()


def test_temperature_is_bounded_on_separable_validation():
    y = np.array([0, 1, 2] * 10)
    logits = np.eye(3)[y] * 5.0  # perfectly separable: unbounded NLL wants T -> 0
    t = fit_temperature(logits, y)
    assert 0.25 <= t < 0.3
