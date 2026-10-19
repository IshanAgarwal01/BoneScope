"""Collect runs/*/<split_type>/fold*/metrics.json into mean ± std tables.

Usage:
    python scripts/aggregate_results.py [--runs runs] [--out reports/results.md]
        [--lineage-map configs/lineage_map.yaml]

With --lineage-map (class -> lineage), lineage-level accuracy is computed from
each fold's saved test predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mds.evaluate import SPLIT_CAVEAT, lineage_accuracy

METRICS = ["macro_f1", "balanced_acc", "top3_acc", "accuracy", "lineage_acc", "ece"]


def collect(runs: Path, lineage_of: dict[str, str] | None) -> pd.DataFrame:
    rows = []
    for mfile in sorted(runs.glob("*/*/fold*/metrics.json")):
        m = json.loads(mfile.read_text())
        m["run"] = mfile.parents[2].name
        if lineage_of is not None:
            classes = json.loads((mfile.parent / "classes.json").read_text())
            preds = np.load(mfile.parent / "test_preds.npz", allow_pickle=True)
            m["lineage_acc"] = lineage_accuracy(
                preds["labels"], preds["logits"].argmax(1), classes, lineage_of
            )
        rows.append(m)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (run, split_type), g in df.groupby(["run", "split_type"]):
        row = {
            "run": run,
            "model": g["model"].iloc[0],
            "loss": g["loss"].iloc[0],
            "sampler": g["sampler"].iloc[0],
            "split": SPLIT_CAVEAT.get(split_type, split_type),
            "folds": len(g),
        }
        for m in METRICS:
            if m in g and g[m].notna().any():
                vals = g[m].astype(float)
                sd = vals.std(ddof=1) if len(vals) > 1 else float("nan")
                row[m] = f"{vals.mean():.3f} ± {sd:.3f}" if len(vals) > 1 else f"{vals.mean():.3f}"
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs", type=Path)
    parser.add_argument("--out", default="reports/results.md", type=Path)
    parser.add_argument("--lineage-map", type=Path)
    args = parser.parse_args()

    lineage_of = yaml.safe_load(args.lineage_map.read_text()) if args.lineage_map else None
    df = collect(args.runs, lineage_of)
    if df.empty:
        raise SystemExit(f"No metrics.json under {args.runs}/*/*/fold*/")
    table = summarise(df)
    md = [
        "# Results",
        "",
        "> Educational project, not a medical device. Mean ± sample std over outer folds.",
        "> Every row states its split. Proxy-group and image-level splits **may be optimistic**:",
        "> cells from one patient can still sit in both train and test.",
        "> Plain accuracy is shown only to compare with published baselines; "
        "**macro-F1 is the primary metric.**",
        "",
        table.to_markdown(index=False),
        "",
        "Per-fold rows:",
        "",
        df[["run", "split_type", "fold", *[m for m in METRICS if m in df]]]
        .round(4)
        .to_markdown(index=False),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
