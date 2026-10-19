"""From per-cell predictions to a blast percentage and a WHO 2022 blast range.

EDUCATIONAL DEMO ONLY. Not a medical device; never use for diagnosis.

Why this is only a demo on this dataset:
  * A real marrow blast count is a 500-cell differential of *consecutive*
    nucleated cells on a smear. This dataset is a curated collection of
    crops, so per-patient class ratios in it do not reflect the marrow.
  * WHO 2022 MDS classification also uses peripheral-blood blasts, Auer rods,
    cytogenetics and mutations (e.g. SF3B1, TP53, del(5q)); none are here.
  * Ring sideroblasts need an iron (Prussian blue) stain, which this
    Wright-Giemsa dataset does not have, so MDS-SF3B1/ring-sideroblast
    categories are out of scope entirely.
We therefore demonstrate the arithmetic on synthetic "virtual patients" whose
true blast fraction we control.

Blast % = blasts / all nucleated marrow cells x 100. Non-nucleated elements
(e.g. mature red cells, platelets, smudge cells if present) are excluded from
the denominator; which classes those are is decided from the real class names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

DISCLAIMER = "Research demo. Not for clinical use. Trained on Wright-Giemsa single-cell crops."
WHO_THRESHOLDS: tuple[float, ...] = (5.0, 10.0, 20.0)


@dataclass(frozen=True)
class BlastCategory:
    who: str
    icc: str
    blast_range: str

    @property
    def label(self) -> str:
        return f"{self.who} (ICC: {self.icc})"


# Thresholds are applied to the continuous percentage: "5-9%" means 5 <= x < 10.
_CATEGORIES = (
    (5.0, BlastCategory("MDS-LB", "MDS, NOS / other <5% blast categories", "<5%")),
    (10.0, BlastCategory("MDS-IB1", "MDS-EB", "5-9%")),
    (20.0, BlastCategory("MDS-IB2", "MDS/AML", "10-19%")),
    (float("inf"), BlastCategory("AML", "AML", ">=20%")),
)


def who_category(blast_pct: float) -> BlastCategory:
    """WHO 2022 blast-based range with the ICC 2022 equivalent.

    Uses marrow blast % only. Real classification also depends on blood
    blasts, Auer rods and genetics (WHO 2022 allows AML with defining genetic
    abnormalities below 20% blasts), so this is the blast-count axis only.
    """
    if not np.isfinite(blast_pct) or blast_pct < 0:
        raise ValueError(f"blast_pct must be a finite non-negative number, got {blast_pct}")
    for upper, cat in _CATEGORIES:
        if blast_pct < upper:
            return cat
    raise AssertionError("unreachable")


def crossed_thresholds(
    ci_low: float, ci_high: float, thresholds: Sequence[float] = WHO_THRESHOLDS
) -> list[float]:
    """Thresholds lying inside the CI: the category could plausibly be either side."""
    return [t for t in thresholds if ci_low < t <= ci_high]


def _indices(classes: Sequence[str], names: Sequence[str], what: str) -> np.ndarray:
    missing = set(names) - set(classes)
    if missing:
        raise ValueError(f"{what} classes not in model classes: {sorted(missing)}")
    return np.array([classes.index(n) for n in names], dtype=int)


def _hard_pct(preds: np.ndarray, is_blast: np.ndarray, is_nucleated: np.ndarray) -> float:
    nucleated = is_nucleated[preds].sum()
    return float(100.0 * is_blast[preds].sum() / nucleated) if nucleated else float("nan")


def _soft_pct(p_blast: np.ndarray, p_nucleated: np.ndarray) -> float:
    denom = p_nucleated.sum()
    return float(100.0 * p_blast.sum() / denom) if denom > 0 else float("nan")


@dataclass
class BlastEstimate:
    method: str
    n_cells: int
    n_nucleated: float
    blast_pct: float
    ci_low: float
    ci_high: float
    category: BlastCategory
    borderline: bool
    crossed: list[float] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def summary(self) -> str:
        flag = (
            f" BORDERLINE: 95% CI crosses {', '.join(f'{t:g}%' for t in self.crossed)}."
            if self.borderline
            else ""
        )
        return (
            f"Blast {self.blast_pct:.1f}% (95% CI {self.ci_low:.1f}-{self.ci_high:.1f}%, "
            f"{self.method} count, {self.n_nucleated:.0f} nucleated of {self.n_cells} cells) -> "
            f"{self.category.label}, range {self.category.blast_range}.{flag} {self.disclaimer}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.label
        return d


def estimate_blast_percentage(
    probs: np.ndarray,
    classes: Sequence[str],
    blast_classes: Sequence[str],
    non_nucleated_classes: Sequence[str] = (),
    method: str = "hard",
    n_boot: int = 1000,
    seed: int = 0,
) -> BlastEstimate:
    """Blast % with a percentile-bootstrap 95% CI over cells.

    hard: count argmax predictions (what a person tallying cells does).
    soft: sum probabilities, so a cell that is 60% blast / 40% promyelocyte
          contributes 0.6 blasts. Less jumpy for uncertain cells, and it
          benefits directly from temperature-scaled (calibrated) probabilities.

    The bootstrap resamples cells with replacement 1,000 times; the CI shows
    how much the estimate would move with a different random 500 cells. It
    does NOT include classifier bias: a model that systematically misses
    blasts can produce a tight CI around the wrong value.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(classes):
        raise ValueError("probs must be (n_cells, n_classes)")
    n = len(probs)
    if n == 0:
        raise ValueError("no cells")
    blast_idx = _indices(classes, blast_classes, "blast")
    non_nuc_idx = _indices(classes, non_nucleated_classes, "non-nucleated")
    if np.intersect1d(blast_idx, non_nuc_idx).size:
        raise ValueError("a class cannot be both blast and non-nucleated")

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    if method == "hard":
        is_blast = np.zeros(len(classes), bool)
        is_blast[blast_idx] = True
        is_nuc = np.ones(len(classes), bool)
        is_nuc[non_nuc_idx] = False
        preds = probs.argmax(1)
        point = _hard_pct(preds, is_blast, is_nuc)
        n_nuc = float(is_nuc[preds].sum())
        b_blast = is_blast[preds][boot_idx].sum(1)
        b_nuc = is_nuc[preds][boot_idx].sum(1)
    elif method == "soft":
        p_blast = probs[:, blast_idx].sum(1)
        p_nuc = 1.0 - probs[:, non_nuc_idx].sum(1)
        point = _soft_pct(p_blast, p_nuc)
        n_nuc = float(p_nuc.sum())
        b_blast = p_blast[boot_idx].sum(1)
        b_nuc = p_nuc[boot_idx].sum(1)
    else:
        raise ValueError("method must be 'hard' or 'soft'")
    if not np.isfinite(point):
        raise ValueError("no nucleated cells predicted; blast % is undefined")
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = 100.0 * b_blast / b_nuc
    boots = boots[np.isfinite(boots)]
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    crossed = crossed_thresholds(lo, hi)
    return BlastEstimate(
        method=method,
        n_cells=n,
        n_nucleated=n_nuc,
        blast_pct=point,
        ci_low=lo,
        ci_high=hi,
        category=who_category(point),
        borderline=bool(crossed),
        crossed=crossed,
    )


def dysplasia_percentages(
    preds: np.ndarray,
    classes: Sequence[str],
    lineages: dict[str, tuple[Sequence[str], Sequence[str]]],
) -> pd.DataFrame:
    """Share of dysplastic cells per lineage (WHO significance line: >=10%).

    `lineages` maps lineage -> (all classes in that lineage, dysplastic subset).
    Strong caveat: dysplastic granulocytes are the hardest class in this
    dataset (authors report F1 around 0.31), so these numbers are noisy.
    """
    rows = []
    for lineage, (members, dysplastic) in lineages.items():
        m = _indices(classes, members, lineage)
        d = _indices(classes, dysplastic, lineage)
        n_lin = int(np.isin(preds, m).sum())
        n_dys = int(np.isin(preds, d).sum())
        pct = 100.0 * n_dys / n_lin if n_lin else float("nan")
        rows.append(
            {
                "lineage": lineage,
                "cells": n_lin,
                "dysplastic": n_dys,
                "dysplasia_pct": pct,
                "significant_ge_10pct": bool(n_lin and pct >= 10.0),
            }
        )
    return pd.DataFrame(rows)


# --- virtual patients --------------------------------------------------------


def sample_virtual_patient(
    labels: np.ndarray,
    classes: Sequence[str],
    blast_classes: Sequence[str],
    non_nucleated_classes: Sequence[str],
    target_pct: float,
    n_nucleated: int = 500,
    non_nucleated_extra: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Draw held-out cells with a known blast fraction.

    Returns (cell indices, exact true blast %, sampled_with_replacement).
    `n_nucleated` nucleated cells are drawn (round(target% x n) blasts, the
    rest other nucleated cells, keeping their natural mix), plus optional
    non-nucleated cells that must NOT count in the denominator.
    """
    rng = rng or np.random.default_rng(0)
    blast = _indices(classes, blast_classes, "blast")
    non_nuc = _indices(classes, non_nucleated_classes, "non-nucleated")
    blast_pool = np.flatnonzero(np.isin(labels, blast))
    other_pool = np.flatnonzero(~np.isin(labels, blast) & ~np.isin(labels, non_nuc))
    non_nuc_pool = np.flatnonzero(np.isin(labels, non_nuc))

    n_blast = int(round(n_nucleated * target_pct / 100.0))
    n_other = n_nucleated - n_blast
    replaced = False

    def draw(pool: np.ndarray, k: int) -> np.ndarray:
        nonlocal replaced
        if k == 0:
            return np.empty(0, int)
        if len(pool) == 0:
            raise ValueError("pool is empty: cannot build this virtual patient")
        replace = k > len(pool)
        replaced |= replace
        return rng.choice(pool, size=k, replace=replace)

    idx = np.concatenate(
        [
            draw(blast_pool, n_blast),
            draw(other_pool, n_other),
            draw(non_nuc_pool, non_nucleated_extra),
        ]
    )
    rng.shuffle(idx)
    return idx, 100.0 * n_blast / n_nucleated, replaced


def run_virtual_patients(
    probs: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    blast_classes: Sequence[str],
    non_nucleated_classes: Sequence[str] = (),
    targets: Sequence[float] = (2, 4, 7, 9, 12, 18, 25),
    n_nucleated: int = 500,
    non_nucleated_extra: int = 0,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """One virtual patient per target fraction, estimated hard and soft."""
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        idx, true_pct, replaced = sample_virtual_patient(
            labels, classes, blast_classes, non_nucleated_classes, target,
            n_nucleated, non_nucleated_extra, rng,
        )  # fmt: skip
        row = {
            "target_pct": target,
            "true_pct": true_pct,
            "true_category": who_category(true_pct).label,
            "n_cells": len(idx),
            "with_replacement": replaced,
        }
        for method in ("hard", "soft"):
            est = estimate_blast_percentage(
                probs[idx], classes, blast_classes, non_nucleated_classes, method, n_boot, seed
            )
            row |= {
                f"{method}_pct": est.blast_pct,
                f"{method}_ci_low": est.ci_low,
                f"{method}_ci_high": est.ci_high,
                f"{method}_category": est.category.label,
                f"{method}_borderline": est.borderline,
                f"{method}_category_correct": est.category == who_category(true_pct),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_virtual_patients(df: pd.DataFrame, path: str, title: str = "") -> None:
    """True vs predicted blast % with 95% CIs and the WHO 2022 thresholds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    top = max(30.0, float(df[["true_pct", "hard_ci_high", "soft_ci_high"]].max().max()) + 2)
    ax.plot([0, top], [0, top], ls="--", c="grey", lw=1, label="perfect agreement")
    for t in WHO_THRESHOLDS:
        ax.axhline(t, c="#c44", lw=0.8, alpha=0.6)
        ax.axvline(t, c="#c44", lw=0.8, alpha=0.6)
    for method, offset, marker in (("hard", -0.25, "o"), ("soft", 0.25, "s")):
        ax.errorbar(
            df["true_pct"] + offset,
            df[f"{method}_pct"],
            yerr=[
                df[f"{method}_pct"] - df[f"{method}_ci_low"],
                df[f"{method}_ci_high"] - df[f"{method}_pct"],
            ],
            fmt=marker,
            capsize=3,
            label=f"{method} count (95% bootstrap CI)",
        )
    for t, name in zip((2.5, 7.5, 15, 25), ("MDS-LB", "MDS-IB1", "MDS-IB2", "AML"), strict=True):
        if t < top:
            ax.text(top * 0.98, t, name, ha="right", va="center", fontsize=8, color="#a33")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("true blast % (virtual patient)")
    ax.set_ylabel("predicted blast %")
    ax.set_title(title or "Virtual patients: predicted vs true blast %")
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
