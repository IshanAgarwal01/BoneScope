"""Fold creation and leakage checks.

The single most common way medical-imaging projects fool themselves is letting
cells from the same patient (or slide) appear in both training and test sets:
the model learns that patient's stain/lighting and the score inflates. Every
function here exists to make that impossible, or at least loud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SPLIT_COLUMNS = ["path", "label_name", "label", "group", "fold", "split_type"]
GROUPED_SPLIT_TYPES = {"patient", "slide", "proxy"}


class LeakageError(AssertionError):
    """Raised when a group appears on both sides of a split."""


def encode_labels(label_names: pd.Series) -> tuple[pd.Series, list[str]]:
    """Map class names to integers in sorted order (stable across runs)."""
    classes = sorted(label_names.unique())
    lookup = {name: i for i, name in enumerate(classes)}
    return label_names.map(lookup).astype(int), classes


def make_folds(
    df: pd.DataFrame,
    split_type: str,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a `fold` (0..n_splits-1) to every image.

    Grouped split types use StratifiedGroupKFold: whole groups go to one fold
    while class proportions stay as even as group sizes allow. "stratified"
    ignores groups (image-level, the optimistic comparison the 2026 paper used);
    its `group` column is set to the row index so the leakage checks still hold
    trivially.
    """
    out = df.copy()
    if "label" not in out:
        out["label"], _ = encode_labels(out["label_name"])
    if split_type in GROUPED_SPLIT_TYPES:
        if "group" not in out:
            raise ValueError(f"split_type={split_type!r} needs a 'group' column")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(out, out["label"], groups=out["group"])
    elif split_type == "stratified":
        out["group"] = [f"img{i}" for i in range(len(out))]
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(out, out["label"])
    else:
        raise ValueError(f"Unknown split_type {split_type!r}")
    out["fold"] = -1
    for k, (_, test_idx) in enumerate(folds):
        out.iloc[test_idx, out.columns.get_loc("fold")] = k
    out["group"] = out["group"].astype(str)
    out["split_type"] = split_type
    assert (out["fold"] >= 0).all(), "every image must get a fold"
    assert_folds_group_disjoint(out)
    return out[SPLIT_COLUMNS]


def make_inner_val(
    splits: pd.DataFrame,
    n_inner: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Pick a group-disjoint validation set inside each outer training portion.

    For outer test fold k, the other 4 folds are split once more with
    StratifiedGroupKFold(n_inner) and one part (~1/n_inner, ~10% of all data)
    becomes validation. Checkpoints and early stopping use validation only, so
    the test fold stays untouched until the final evaluation.

    Returns long-format rows: (split_type, outer_fold, path).
    """
    rows = []
    for split_type, sdf in splits.groupby("split_type"):
        for k in sorted(sdf["fold"].unique()):
            pool = sdf[sdf["fold"] != k].reset_index(drop=True)
            if split_type == "stratified":
                splitter = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + k)
                _, val_idx = next(splitter.split(pool, pool["label"]))
            else:
                splitter = StratifiedGroupKFold(
                    n_splits=n_inner, shuffle=True, random_state=seed + k
                )
                _, val_idx = next(splitter.split(pool, pool["label"], groups=pool["group"]))
            for path in pool.loc[val_idx, "path"]:
                rows.append({"split_type": split_type, "outer_fold": int(k), "path": path})
    return pd.DataFrame(rows)


def partition(
    splits: pd.DataFrame,
    inner_val: pd.DataFrame,
    split_type: str,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) for one outer fold, with leakage checks."""
    sdf = splits[splits["split_type"] == split_type]
    if sdf.empty:
        raise ValueError(f"No rows with split_type={split_type!r} in splits file")
    val_paths = set(
        inner_val.loc[
            (inner_val["split_type"] == split_type) & (inner_val["outer_fold"] == fold), "path"
        ]
    )
    if not val_paths:
        raise ValueError(f"No inner validation rows for {split_type!r} fold {fold}")
    test = sdf[sdf["fold"] == fold]
    rest = sdf[sdf["fold"] != fold]
    val = rest[rest["path"].isin(val_paths)]
    train = rest[~rest["path"].isin(val_paths)]
    assert_disjoint({"train": train, "val": val, "test": test})
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def assert_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    """No image and no group may appear in two parts."""
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared_paths = set(parts[a]["path"]) & set(parts[b]["path"])
            if shared_paths:
                raise LeakageError(f"{len(shared_paths)} images in both {a} and {b}")
            shared_groups = set(parts[a]["group"]) & set(parts[b]["group"])
            if shared_groups:
                example = sorted(shared_groups)[:5]
                raise LeakageError(
                    f"{len(shared_groups)} groups in both {a} and {b}, e.g. {example}"
                )


def assert_folds_group_disjoint(splits: pd.DataFrame) -> None:
    """Every group must live in exactly one fold (per split type)."""
    for split_type, sdf in splits.groupby("split_type"):
        folds_per_group = sdf.groupby("group")["fold"].nunique()
        bad = folds_per_group[folds_per_group > 1]
        if len(bad):
            raise LeakageError(
                f"[{split_type}] {len(bad)} groups span several folds, e.g. {list(bad.index[:5])}"
            )


def fold_class_table(splits: pd.DataFrame) -> pd.DataFrame:
    """Images per (class, fold): shows whether every fold can test every class."""
    return splits.pivot_table(
        index="label_name", columns="fold", values="path", aggfunc="count", fill_value=0
    )


def classes_missing_from_folds(splits: pd.DataFrame) -> dict[str, list[int]]:
    """Classes absent from some fold, restricted to classes spread over >= n folds' worth
    of groups (a class with 3 groups can never appear in 5 folds; that is expected)."""
    n_folds = splits["fold"].nunique()
    table = fold_class_table(splits)
    groups_per_class = splits.groupby("label_name")["group"].nunique()
    missing = {}
    for cls, row in table.iterrows():
        if groups_per_class[cls] >= n_folds:
            absent = [int(f) for f, n in row.items() if n == 0]
            if absent:
                missing[cls] = absent
    return missing


def load_splits(
    path: str | Path, split_type: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load splits.csv and its inner-validation companion, re-checking leakage."""
    path = Path(path)
    splits = pd.read_csv(path, dtype={"group": str})
    inner = pd.read_csv(path.with_name(path.stem + "_inner_val.csv"))
    if split_type is not None:
        splits = splits[splits["split_type"] == split_type]
        inner = inner[inner["split_type"] == split_type]
    missing_cols = set(SPLIT_COLUMNS) - set(splits.columns)
    if missing_cols:
        raise ValueError(f"{path} lacks columns {sorted(missing_cols)}")
    assert_folds_group_disjoint(splits)
    return splits.reset_index(drop=True), inner.reset_index(drop=True)


def class_list(splits: pd.DataFrame) -> list[str]:
    """Class names ordered by integer label (as written by make_splits)."""
    pairs = splits[["label", "label_name"]].drop_duplicates().sort_values("label")
    if pairs["label"].duplicated().any():
        raise ValueError("label -> label_name mapping is not one-to-one")
    names = pairs["label_name"].tolist()
    if pairs["label"].tolist() != list(range(len(names))):
        raise ValueError("labels must be 0..C-1 with no gaps")
    return names


def label_counts(labels: np.ndarray | pd.Series, num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
