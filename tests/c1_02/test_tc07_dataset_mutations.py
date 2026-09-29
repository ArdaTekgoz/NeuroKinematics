"""Acceptance mutations against generated rows; no production file is changed."""

from copy import deepcopy
import json

import pytest

from neurokinematics.data.pair_validation import load_rows, validate_normalization
from neurokinematics.data.pairs import ROOT, audit_rows, load_contract, roots
from neurokinematics.kinematics.model import load_robot


@pytest.fixture(scope="module")
def production():
    output = ROOT / "data/generated/C1-02/v1"
    manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
    config, schema = load_contract()
    rows = load_rows(output, manifest, schema)
    source, _ = roots()
    query = ROOT / "data/generated/F0-05/acceptance/run-a/query-list.jsonl"
    return rows, source, query, config, load_robot()


def test_teacher_filtered_test_row_rejected(production):
    rows, source, query, config, inputs = production
    target = next(i for i, row in enumerate(rows) if row["split"] == "test" and row["pair_mode"] == "wide" and not row["label_present"])
    with pytest.raises(ValueError, match="count mismatch"):
        audit_rows(rows[:target] + rows[target+1:], source, query, config, inputs)


def test_cross_split_group_rejected(production):
    rows, source, query, config, inputs = production
    broken = list(rows)
    item = deepcopy(broken[0])
    item["split"] = "test" if item["split"] != "test" else "train"
    broken[0] = item
    with pytest.raises(ValueError, match="lineage mismatch"):
        audit_rows(broken, source, query, config, inputs)


def test_cross_split_teacher_family_rejected(production):
    rows, source, query, config, inputs = production
    broken = list(rows)
    first = broken[0]
    other = next(row for row in broken if row["split"] != first["split"])
    item = deepcopy(first)
    item["teacher_candidate_family"] = other["teacher_candidate_family"]
    broken[0] = item
    with pytest.raises(ValueError, match="cross-split leakage: candidate_family"):
        audit_rows(broken, source, query, config, inputs)


def test_validation_statistics_cannot_enter_train_normalization(production):
    rows, _, _, _, _ = production
    output = ROOT / "data/generated/C1-02/v1"
    norm = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    all_positions = [row["position_m"] for row in rows]
    import numpy as np
    norm["position_mean_m"] = np.asarray(all_positions).mean(axis=0).tolist()
    with pytest.raises(ValueError, match="train-only"):
        validate_normalization(norm, rows)


def test_corrupt_shard_hash_rejected(production, monkeypatch):
    rows, _, _, _, _ = production
    assert rows
    import neurokinematics.data.pair_validation as validation
    output = ROOT / "data/generated/C1-02/v1"
    manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
    schema = load_contract()[1]
    monkeypatch.setattr(validation, "sha", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="shard SHA mismatch"):
        validation.load_rows(output, manifest, schema)
