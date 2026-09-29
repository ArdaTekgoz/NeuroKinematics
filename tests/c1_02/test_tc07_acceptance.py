"""T-C07 production acceptance, using one full read-only verification pass."""

import pytest

from neurokinematics.data.pair_validation import verify
from neurokinematics.data.pairs import ROOT


@pytest.fixture(scope="module")
def result():
    return verify(ROOT / "data/generated/C1-02/v1")


def test_manifest_schema_and_shards(result):
    assert result["status"] == "PASS"
    assert result["records"] == 24000
    assert result["shards"] > 0
    assert len(result["dataset_content_sha256"]) == 64


def test_local_wide_split_counts(result):
    counts = result["leakage"]["counts"]
    for split, planned in (("train", 8400), ("validation", 1800), ("test", 1800)):
        for mode in ("local", "wide"):
            assert sum(counts[f"{family}/{split}/{mode}"] for family in ("main", "boundary", "singularity")) == planned


def test_local_difference_and_limits(result):
    assert result["leakage"]["local_exact_equality"] == 0
    assert result["leakage"]["local_abs_delta_quantiles_rad"]["1"] <= .1


def test_split_and_teacher_family_isolation(result):
    for field in ("cross_split_group", "cross_split_candidate_family", "cross_split_exact_q", "cross_split_exact_pose"):
        assert result["leakage"][field] == 0
    assert result["candidate"]["cross_split_restart_or_valid_candidate_q"] == 0
    assert result["ancestry"]["cross_split_source_root"] == 0
    assert result["ancestry"]["cross_split_resampling_seed"] == 0
    assert result["near_pose"]["cross_split_near_pose"] == 0


def test_benchmark_isolation(result):
    for field in ("benchmark_exact_q_overlap", "benchmark_exact_pose_overlap", "benchmark_group_overlap"):
        assert result["leakage"][field] == 0
    assert result["benchmark_near_pose"]["same_target_pose"] == 0


def test_teacher_inventory_and_branch_choice(result):
    assert result["candidate"]["wide_rows"] == 12000
    assert result["candidate"]["candidate_rows"] == 48000


def test_train_only_normalization(result):
    assert result["normalization_train_only"] == "PASS"
