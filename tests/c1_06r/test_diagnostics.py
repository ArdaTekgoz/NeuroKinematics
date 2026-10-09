"""Failure-oriented checks for the new diagnostics, never final-test data."""
from dataclasses import replace
import numpy as np
import pytest
from neurokinematics.neural import c106r
from neurokinematics.neural.c104 import load_data, read_json, validate_rows, reject_shifted_labels
from neurokinematics.kinematics.model import load_robot


@pytest.fixture(scope="module")
def rows():
    train, validation = load_data(label_fk=False)
    return train, validation


def test_invalid_predictions_stay_in_denominator(rows):
    selected = c106r.subsets(rows[0], read_json(c106r.CONFIG))["local64"]
    q = selected.q_target.copy()
    q[0] = np.nan
    q[1, 0] = load_robot().limits[0][1] + 1
    result = c106r.geometric_metrics(q, selected, details=True)
    assert result["n"] == 64 and result["profile_a"] == 62
    assert result["nonfinite"] == 1 and result["out_of_limits"] == 1
    assert result["position_m"]["p99"] is None
    assert len(result["rows"]) == 64


def test_validation_cannot_be_overfit_training(rows):
    with pytest.raises(ValueError, match="train only"):
        c106r.subsets(rows[1], read_json(c106r.CONFIG))


def test_sealed_split_rejected(rows):
    with pytest.raises(ValueError, match="sealed split"):
        validate_rows(replace(rows[1], split="test"))


def test_missing_teacher_label_cannot_enter_overfit(rows):
    mixed = c106r.subsets(rows[0], read_json(c106r.CONFIG))["mixed64"]
    assert mixed.label_present.all()
    assert len(mixed.pair_id) == 64
    assert len(set(zip(mixed.family, mixed.mode))) == 6
    with pytest.raises(ValueError, match="insufficient"):
        c106r.subsets(replace(rows[0], label_present=np.zeros_like(rows[0].label_present)), read_json(c106r.CONFIG))


def test_wrong_joint_scaling_rejected(rows):
    selected = c106r.subsets(rows[0], read_json(c106r.CONFIG))["local64"]
    with pytest.raises(ValueError, match="scaling"):
        validate_rows(replace(selected, target_normalized=selected.target_normalized * 180 / np.pi))


def test_shifted_label_negative_is_effective(rows):
    selected = c106r.subsets(rows[0], read_json(c106r.CONFIG))["local64"]
    assert reject_shifted_labels(selected)["rejected_rows"] == 64


def test_hash_drift_blocks_execution(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        c106r.guard_hashes({str(source): "0" * 64})


def test_shape_cannot_drop_failed_rows(rows):
    selected = c106r.subsets(rows[0], read_json(c106r.CONFIG))["local64"]
    with pytest.raises(ValueError, match="shape mismatch"):
        c106r.geometric_metrics(selected.q_target[:-1], selected)
