"""Phase 5-6 analysis of one cross-validated run.

Pools out-of-fold test predictions (every image is predicted by the one model
that never saw it), fits a temperature per fold on that fold's validation set,
and writes the per-class report, confusion matrix, top confusions,
calibration before/after, blast precision/recall and virtual patients.

Usage:
    python scripts/analyze_run.py --run runs/convnext_tiny/proxy \
        --blast-classes <name> [<name> ...] --non-nucleated <name> [...] \
        [--lineage-map configs/lineage_map.yaml] [--out reports]

Blast and non-nucleated class names must be chosen from the real class list
(team decision at Checkpoint 1); nothing is guessed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mds.calibration import expected_calibration_error, fit_temperature, softmax_np
from mds.clinical import DISCLAIMER, plot_virtual_patients, run_virtual_patients
from mds.evaluate import (
    SPLIT_CAVEAT,
    compute_metrics,
    normalized_confusion,
    per_class_report,
    plot_confusion,
    plot_reliability,
    top_confusions,
)


def load_folds(run: Path) -> tuple[list[str], list[dict]]:
    folds = []
    classes: list[str] | None = None
    for d in sorted(run.glob("fold*")):
        if not (d / "test_preds.npz").exists():
            continue
        c = json.loads((d / "classes.json").read_text())
        if classes is not None and c != classes:
            raise SystemExit(f"{d} has a different class list from earlier folds")
        classes = c
        val = np.load(d / "val_preds.npz", allow_pickle=True)
        test = np.load(d / "test_preds.npz", allow_pickle=True)
        t = fit_temperature(val["logits"], val["labels"])
        folds.append(
            {
                "fold": int(d.name.removeprefix("fold")),
                "temperature": t,
                "logits": test["logits"],
                "labels": test["labels"],
                "paths": test["paths"],
            }
        )
    if not folds:
        raise SystemExit(f"No fold*/test_preds.npz under {run}")
    return classes, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="runs/<name>/<split_type>")
    parser.add_argument("--blast-classes", nargs="+", required=True)
    parser.add_argument("--non-nucleated", nargs="*", default=[])
    parser.add_argument("--lineage-map", type=Path)
    parser.add_argument("--out", default="reports", type=Path)
    parser.add_argument("--n-virtual", default=500, type=int, help="nucleated cells per patient")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    split_type = args.run.name
    caveat = SPLIT_CAVEAT.get(split_type, split_type)
    name = f"{args.run.parent.name}_{split_type}"
    figs = args.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    lineage_of = yaml.safe_load(args.lineage_map.read_text()) if args.lineage_map else None

    classes, folds = load_folds(args.run)
    logits = np.concatenate([f["logits"] for f in folds])
    labels = np.concatenate([f["labels"] for f in folds])
    raw = softmax_np(logits)
    cal = np.concatenate([softmax_np(f["logits"], f["temperature"]) for f in folds])
    preds = raw.argmax(1)

    # per-fold metrics (mean ± std) before/after temperature scaling
    per_fold = []
    for f in folds:
        m = compute_metrics(f["labels"], softmax_np(f["logits"]), classes, lineage_of)
        m["ece_calibrated"] = expected_calibration_error(
            softmax_np(f["logits"], f["temperature"]), f["labels"]
        )
        m |= {"fold": f["fold"], "temperature": f["temperature"]}
        per_fold.append(m)
    pf = pd.DataFrame(per_fold).set_index("fold")

    report = per_class_report(labels, preds, classes)
    report.to_csv(args.out / f"per_class_{name}.csv", index=False)
    cm = normalized_confusion(labels, preds, len(classes))
    plot_confusion(cm, classes, figs / f"confusion_{name}.png", f"{name}: out-of-fold ({caveat})")
    confusions = top_confusions(labels, preds, classes)
    plot_reliability(
        {"uncalibrated": (raw, labels), "temperature-scaled": (cal, labels)},
        figs / f"reliability_{name}.png",
    )

    blast_rows = report[report["class"].isin(args.blast_classes)]
    vp = run_virtual_patients(
        cal, labels, classes, args.blast_classes, args.non_nucleated,
        n_nucleated=args.n_virtual, seed=args.seed,
    )  # fmt: skip
    vp.round(2).to_csv(args.out / f"virtual_patients_{name}.csv", index=False)
    plot_virtual_patients(
        vp, str(figs / f"virtual_patients_{name}.png"), f"Virtual patients ({name}, {caveat})"
    )

    def mean_sd(col: str) -> str:
        v = pf[col].astype(float)
        return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"

    metric_cols = ["macro_f1", "balanced_acc", "top3_acc", "accuracy", "ece", "ece_calibrated"]
    if "lineage_acc" in pf:
        metric_cols.insert(4, "lineage_acc")
    md = [
        f"# Analysis: `{name}`",
        "",
        f"> {DISCLAIMER}",
        f"> Split: **{caveat}**. {len(folds)} folds, {len(labels):,} out-of-fold test images.",
        "",
        "## Headline metrics (mean ± std over folds)",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {c} | {mean_sd(c)} |" for c in metric_cols],
        "",
        f"Temperatures per fold: {', '.join(f'{t:.2f}' for t in pf['temperature'])}",
        "",
        "## Blast class(es)",
        "",
        blast_rows.round(3).to_markdown(index=False),
        "",
        "Missed blasts (low recall) push a patient's blast % down, toward a milder category;",
        "false blasts (low precision) push it up.",
        "",
        "## Per-class report (pooled out-of-fold)",
        "",
        report.round(3).to_markdown(index=False),
        "",
        "## Top confusions",
        "",
        confusions.round(3).to_markdown(index=False),
        "",
        "## Virtual patients",
        "",
        f"{args.n_virtual} nucleated held-out cells each, calibrated probabilities, "
        "1,000 bootstraps.",
        "These are synthetic mixtures, not real patients.",
        "",
        vp[
            [
                "target_pct",
                "true_category",
                "hard_pct",
                "hard_ci_low",
                "hard_ci_high",
                "hard_category",
                "hard_borderline",
                "soft_pct",
                "soft_category",
            ]
        ]
        .round(2)
        .to_markdown(index=False),  # fmt: skip
        "",
        f"Figures: `figures/confusion_{name}.png`, `figures/reliability_{name}.png`, "
        f"`figures/virtual_patients_{name}.png`",
        "",
    ]
    (args.out / f"analysis_{name}.md").write_text("\n".join(md))
    print("\n".join(md[:20]))
    print(f"... wrote {args.out / f'analysis_{name}.md'}")


if __name__ == "__main__":
    main()
