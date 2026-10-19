import numpy as np
import pytest

from mds.clinical import (
    crossed_thresholds,
    dysplasia_percentages,
    estimate_blast_percentage,
    run_virtual_patients,
    sample_virtual_patient,
    who_category,
)

CLASSES = ["blast", "lymphocyte", "neutrophil", "erythroblast", "platelet"]
BLAST, NON_NUC = ["blast"], ["platelet"]


def onehot(labels):
    return np.eye(len(CLASSES))[labels]


@pytest.mark.parametrize(
    "pct,who,icc",
    [
        (0.0, "MDS-LB", None),
        (4.99, "MDS-LB", None),
        (5.0, "MDS-IB1", "MDS-EB"),
        (9.99, "MDS-IB1", "MDS-EB"),
        (10.0, "MDS-IB2", "MDS/AML"),
        (19.99, "MDS-IB2", "MDS/AML"),
        (20.0, "AML", "AML"),
        (85.0, "AML", "AML"),
    ],
)
def test_who_thresholds(pct, who, icc):
    cat = who_category(pct)
    assert cat.who == who
    if icc:
        assert cat.icc == icc
        assert cat.label == f"{who} (ICC: {icc})"


def test_who_rejects_invalid():
    with pytest.raises(ValueError):
        who_category(float("nan"))
    with pytest.raises(ValueError):
        who_category(-1)


def test_hard_count_excludes_non_nucleated_from_denominator():
    # 10 blasts, 90 other nucleated, 50 platelets -> 10 / 100 = 10%, not 10 / 150
    labels = [0] * 10 + [1] * 30 + [2] * 30 + [3] * 30 + [4] * 50
    est = estimate_blast_percentage(onehot(labels), CLASSES, BLAST, NON_NUC, "hard")
    assert est.blast_pct == pytest.approx(10.0)
    assert est.n_nucleated == 100
    assert est.category.who == "MDS-IB2"


def test_soft_count_uses_probabilities():
    probs = np.array(
        [
            [0.6, 0.4, 0.0, 0.0, 0.0],  # 0.6 blast
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5],  # half nucleated
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    est = estimate_blast_percentage(probs, CLASSES, BLAST, NON_NUC, "soft")
    assert est.blast_pct == pytest.approx(100 * 0.6 / 3.5)
    hard = estimate_blast_percentage(probs, CLASSES, BLAST, NON_NUC, "hard")
    # row 2 is a tie; argmax takes the first (neutrophil), so 4 nucleated, 1 blast
    assert hard.blast_pct == pytest.approx(25.0)


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    labels = rng.choice(4, size=500, p=[0.08, 0.3, 0.32, 0.3])
    a = estimate_blast_percentage(onehot(labels), CLASSES, BLAST, NON_NUC, seed=3)
    b = estimate_blast_percentage(onehot(labels), CLASSES, BLAST, NON_NUC, seed=3)
    assert a.ci_low <= a.blast_pct <= a.ci_high
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    # binomial SE at p~0.08, n=500 is ~1.2 points -> CI width ~4.8
    assert 3.0 < a.ci_high - a.ci_low < 7.0


def test_borderline_flag():
    assert crossed_thresholds(3.0, 6.0) == [5.0]
    assert crossed_thresholds(8.0, 21.0) == [10.0, 20.0]
    assert crossed_thresholds(5.5, 9.5) == []
    labels = [0] * 25 + [1] * 475  # 5.0% exactly: CI must straddle 5
    est = estimate_blast_percentage(onehot(labels), CLASSES, BLAST, NON_NUC)
    assert est.borderline and est.crossed == [5.0]
    assert "BORDERLINE" in est.summary() and "Not for clinical use" in est.summary()


def test_input_validation():
    with pytest.raises(ValueError):
        estimate_blast_percentage(onehot([0, 1]), CLASSES, ["myeloblast"], NON_NUC)
    with pytest.raises(ValueError):
        estimate_blast_percentage(onehot([0, 1]), CLASSES, BLAST, ["blast"])
    with pytest.raises(ValueError):
        estimate_blast_percentage(onehot([4, 4]), CLASSES, BLAST, NON_NUC)  # no nucleated cells


def test_virtual_patient_has_exact_blast_fraction():
    rng = np.random.default_rng(0)
    labels = rng.choice(5, size=3000, p=[0.1, 0.25, 0.3, 0.25, 0.1])
    idx, true_pct, replaced = sample_virtual_patient(
        labels, CLASSES, BLAST, NON_NUC, 7.0, 500, non_nucleated_extra=40, rng=rng
    )
    picked = labels[idx]
    assert len(idx) == 540 and not replaced
    assert (picked == 0).sum() == 35 and (picked == 4).sum() == 40
    assert true_pct == pytest.approx(7.0)


def test_perfect_classifier_recovers_virtual_patients():
    rng = np.random.default_rng(0)
    labels = rng.choice(5, size=4000, p=[0.1, 0.25, 0.3, 0.25, 0.1])
    df = run_virtual_patients(onehot(labels), labels, CLASSES, BLAST, NON_NUC, n_boot=200)
    assert np.allclose(df["hard_pct"], df["true_pct"])
    assert np.allclose(df["soft_pct"], df["true_pct"])
    assert df["hard_category_correct"].all()
    assert df["true_category"].tolist()[0].startswith("MDS-LB")
    assert df["true_category"].tolist()[-1].startswith("AML")


def test_dysplasia_percentages():
    classes = ["neutrophil", "dys_neutrophil", "erythroblast", "dys_erythroblast"]
    preds = np.array([0] * 85 + [1] * 15 + [2] * 95 + [3] * 5)
    df = dysplasia_percentages(
        preds,
        classes,
        {
            "granulocytic": (["neutrophil", "dys_neutrophil"], ["dys_neutrophil"]),
            "erythroid": (["erythroblast", "dys_erythroblast"], ["dys_erythroblast"]),
        },
    ).set_index("lineage")
    assert df.loc["granulocytic", "dysplasia_pct"] == pytest.approx(15.0)
    assert df.loc["granulocytic", "significant_ge_10pct"]
    assert not df.loc["erythroid", "significant_ge_10pct"]
