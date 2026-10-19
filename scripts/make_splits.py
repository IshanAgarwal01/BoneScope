"""Create the frozen 5-fold splits ONCE and write them to splits/.

Group source, in the priority order from the project brief:
  --group-regex  extract a patient/slide ID from each relative path (if IDs exist)
  --groups-csv   a CSV with columns path,group (e.g. from compute_proxy_groups.py)
A plain image-level stratified split is always written alongside, so results
can be reported both ways.

Examples:
    # proxy groups (no IDs in the release)
    python scripts/make_splits.py --root data/raw --groups-csv splits/proxy_groups.csv \
        --split-type proxy
    # IDs found in filenames such as "P012_xxx.jpg"
    python scripts/make_splits.py --root data/raw --group-regex "^.*/(P\\d+)_" \
        --split-type patient

Outputs:
    splits/splits.csv            path,label_name,label,group,fold,split_type
    splits/splits_inner_val.csv  split_type,outer_fold,path (validation per outer fold)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from mds.data import index_images
from mds.splits import (
    classes_missing_from_folds,
    encode_labels,
    fold_class_table,
    make_folds,
    make_inner_val,
)


def apply_class_merges(df: pd.DataFrame, merge_yaml: Path | None) -> pd.DataFrame:
    """Optionally merge rare classes (e.g. into lineage "other" buckets).

    The YAML maps original class name -> new class name. Off unless you pass a
    file: merging changes what the model is asked to learn, so it is a team
    decision made at Checkpoint 1, not a default.
    """
    if merge_yaml is None:
        return df
    mapping = yaml.safe_load(merge_yaml.read_text()) or {}
    unknown = set(mapping) - set(df["label_name"])
    if unknown:
        raise SystemExit(f"Merge map names classes not in the data: {sorted(unknown)}")
    out = df.copy()
    out["label_name"] = out["label_name"].replace(mapping)
    return out


def groups_from_regex(paths: pd.Series, pattern: str) -> pd.Series:
    rx = re.compile(pattern)
    groups = paths.map(lambda p: m.group(1) if (m := rx.search(p)) else None)
    if groups.isna().any():
        bad = paths[groups.isna()].head(5).tolist()
        raise SystemExit(f"Regex did not match {groups.isna().sum()} paths, e.g. {bad}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/raw", type=Path)
    parser.add_argument("--out", default="splits/splits.csv", type=Path)
    parser.add_argument("--split-type", choices=["patient", "slide", "proxy", "stratified"])
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--group-regex")
    parser.add_argument("--merge-map", type=Path, help="optional YAML: old class -> new class")
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--force", action="store_true", help="overwrite existing frozen splits")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(
            f"{args.out} already exists. Splits are frozen on purpose; "
            "pass --force only if the team agreed to regenerate them."
        )

    df = apply_class_merges(index_images(args.root), args.merge_map)
    df["label"], classes = encode_labels(df["label_name"])

    frames = []
    if args.split_type and args.split_type != "stratified":
        if args.group_regex:
            df["group"] = groups_from_regex(df["path"], args.group_regex)
        elif args.groups_csv:
            g = pd.read_csv(args.groups_csv, dtype={"group": str})[["path", "group"]]
            df = df.merge(g, on="path", how="left", validate="one_to_one")
            if df["group"].isna().any():
                raise SystemExit(f"{df['group'].isna().sum()} images have no group in CSV")
        else:
            raise SystemExit("--split-type patient/slide/proxy needs --group-regex or --groups-csv")
        frames.append(make_folds(df, args.split_type, args.n_splits, args.seed))
    frames.append(
        make_folds(
            df.drop(columns="group", errors="ignore"), "stratified", args.n_splits, args.seed
        )
    )
    splits = pd.concat(frames, ignore_index=True)
    inner = make_inner_val(splits, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.out, index=False)
    inner.to_csv(args.out.with_name(args.out.stem + "_inner_val.csv"), index=False)

    print(f"{len(classes)} classes, {len(df):,} images -> {args.out}")
    for split_type, sdf in splits.groupby("split_type"):
        print(f"\n[{split_type}] groups: {sdf['group'].nunique():,}")
        print(fold_class_table(sdf).to_string())
        missing = classes_missing_from_folds(sdf)
        print("classes missing from a fold (despite enough groups):", missing or "none")


if __name__ == "__main__":
    main()
