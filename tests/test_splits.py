import numpy as np
import pandas as pd
import pytest

from mds.splits import (
    LeakageError,
    assert_disjoint,
    assert_folds_group_disjoint,
    class_list,
    classes_missing_from_folds,
    make_folds,
    make_inner_val,
    partition,
)


@pytest.fixture
def grouped_df():
    """60 patients x 20 cells, 5 classes, with patient-specific class mixes."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(60):
        mix = rng.dirichlet(np.ones(5))
        for i, c in enumerate(rng.choice(5, size=20, p=mix)):
            rows.append({"path": f"c{c}/p{p}_{i}.jpg", "label_name": f"class{c}", "group": f"P{p}"})
    return pd.DataFrame(rows)


@pytest.fixture
def splits(grouped_df):
    proxy = make_folds(grouped_df, "proxy")
    strat = make_folds(grouped_df.drop(columns="group"), "stratified")
    return pd.concat([proxy, strat], ignore_index=True)


def test_every_group_in_exactly_one_fold(splits):
    assert_folds_group_disjoint(splits)
    proxy = splits[splits.split_type == "proxy"]
    assert (proxy.groupby("group")["fold"].nunique() == 1).all()
    assert sorted(proxy["fold"].unique()) == [0, 1, 2, 3, 4]


def test_every_fold_has_every_class_where_possible(splits):
    for _, sdf in splits.groupby("split_type"):
        assert classes_missing_from_folds(sdf) == {}


def test_folds_are_reproducible(grouped_df):
    a = make_folds(grouped_df, "proxy", seed=7)
    b = make_folds(grouped_df, "proxy", seed=7)
    assert a["fold"].tolist() == b["fold"].tolist()


@pytest.mark.parametrize("split_type", ["proxy", "stratified"])
@pytest.mark.parametrize("fold", range(5))
def test_partition_train_val_test_disjoint(splits, split_type, fold):
    inner = make_inner_val(splits)
    train, val, test = partition(splits, inner, split_type, fold)
    sdf = splits[splits.split_type == split_type]
    assert len(train) + len(val) + len(test) == len(sdf)
    assert set(test["fold"]) == {fold}
    assert 0.05 < len(val) / len(sdf) < 0.2
    # no shared groups between any two parts (partition already asserts; double-check)
    assert not set(train.group) & set(test.group)
    assert not set(val.group) & set(test.group)
    assert not set(train.group) & set(val.group)


def test_leakage_is_detected():
    a = pd.DataFrame({"path": ["x.jpg"], "group": ["P1"]})
    b = pd.DataFrame({"path": ["y.jpg"], "group": ["P1"]})
    with pytest.raises(LeakageError):
        assert_disjoint({"train": a, "test": b})


def test_group_spanning_folds_is_detected(splits):
    broken = splits.copy()
    idx = broken.index[broken.split_type == "proxy"][0]
    broken.loc[idx, "fold"] = (broken.loc[idx, "fold"] + 1) % 5
    with pytest.raises(LeakageError):
        assert_folds_group_disjoint(broken)


def test_class_list_matches_labels(splits):
    names = class_list(splits)
    assert names == sorted(names)
    assert len(names) == 5
