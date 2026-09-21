import numpy as np

from neurokinematics.data.factory import assign_splits


def test_group_intersections_and_q_duplicates_are_zero():
    groups = [f"root-{i}" for i in range(1000)]
    split = assign_splits(groups, 20260921, {"train":.7,"validation":.15,"test":.15})
    sets = {name: {groups[i] for i in np.flatnonzero(split == name.encode())} for name in ("train","validation","test")}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]


def test_split_seed_change_changes_assignment():
    groups = [f"g-{i}" for i in range(100)]
    ratios = {"train":.7,"validation":.15,"test":.15}
    assert not np.array_equal(assign_splits(groups, 1, ratios), assign_splits(groups, 2, ratios))
