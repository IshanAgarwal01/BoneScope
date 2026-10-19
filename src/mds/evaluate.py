"""Metrics, per-class reports, confusion matrices and calibration plots.

Macro-F1 is the primary metric: it averages F1 over classes, so a rare class
like dysplastic granulocytes counts as much as common neutrophils. Plain
accuracy is reported only to compare with the published baselines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from mds.calibration import expected_calibration_error, reliability_bins

SPLIT_CAVEAT = {
    "patient": "patient-level split",
    "slide": "slide-level split; patients may span slides",
    "proxy": "approximate group split (proxy clusters); may be optimistic",
    "stratified": "image-level split; may be optimistic",
}


def top_k_accuracy(labels: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    if probs.shape[1] <= k:
        return 1.0
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def lineage_accuracy(
    labels: np.ndarray, preds: np.ndarray, classes: list[str], lineage_of: dict[str, str]
) -> float:
    """Accuracy after collapsing classes to lineages (erythroid, granulocytic, ...).

    Confusing two granulocyte maturation stages is a much smaller error than
    calling a blast a lymphocyte; this metric shows how many errors are of the
    "right family, wrong stage" kind.
    """
    lin = np.array([lineage_of[c] for c in classes])
    return float((lin[labels] == lin[preds]).mean())


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    classes: list[str] | None = None,
    lineage_of: dict[str, str] | None = None,
) -> dict[str, float]:
    """Headline metrics from integer labels and an (N, C) probability matrix."""
    labels = np.asarray(labels, dtype=int)
    preds = probs.argmax(axis=1)
    all_labels = np.arange(probs.shape[1])
    present = np.unique(labels)
    out = {
        # macro-F1 over classes present in this split: a class with no test
        # images has undefined recall and would add an artificial 0.
        "macro_f1": float(
            f1_score(labels, preds, labels=present, average="macro", zero_division=0)
        ),
        "macro_f1_all_classes": float(
            f1_score(labels, preds, labels=all_labels, average="macro", zero_division=0)
        ),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "top3_acc": top_k_accuracy(labels, probs, 3),
        "accuracy": float(accuracy_score(labels, preds)),
        "ece": expected_calibration_error(probs, labels),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
    }
    if lineage_of is not None and classes is not None:
        out["lineage_acc"] = lineage_accuracy(labels, preds, classes, lineage_of)
    return out


def per_class_report(labels: np.ndarray, preds: np.ndarray, classes: list[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        labels, preds, labels=np.arange(len(classes)), zero_division=0
    )
    return (
        pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})
        .sort_values("support", ascending=False)
        .reset_index(drop=True)
    )


def normalized_confusion(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-normalised: row i shows where true class i's cells went (rows sum to 1)."""
    cm = confusion_matrix(labels, preds, labels=np.arange(n_classes)).astype(float)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)


def top_confusions(
    labels: np.ndarray, preds: np.ndarray, classes: list[str], k: int = 15
) -> pd.DataFrame:
    """Most frequent (true -> predicted) mistakes, with share of the true class."""
    cm = confusion_matrix(labels, preds, labels=np.arange(len(classes)))
    rows = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j]:
                rows.append(
                    {
                        "true": classes[i],
                        "predicted": classes[j],
                        "count": int(cm[i, j]),
                        "share_of_true": cm[i, j] / cm[i].sum(),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["true", "predicted", "count", "share_of_true"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k).reset_index(drop=True)


def plot_confusion(cm_norm: np.ndarray, classes: list[str], path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(classes)
    size = max(8, 0.55 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    sns.heatmap(
        cm_norm,
        annot=n <= 30,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        cbar_kws={"shrink": 0.6, "label": "share of true class"},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\n(educational project, not for clinical use)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path, n_bins: int = 15
) -> None:
    """Reliability diagram for one or more (probs, labels) sets, e.g. before/after T."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    for name, (probs, labels) in curves.items():
        conf, acc, counts = reliability_bins(probs, labels, n_bins)
        keep = counts > 0
        ece = expected_calibration_error(probs, labels, n_bins)
        ax.plot(conf[keep], acc[keep], marker="o", label=f"{name} (ECE {ece:.3f})")
    ax.set_xlabel("confidence (top-1 probability)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title("Reliability diagram")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
