"""C1-04 real-data contracts and deliberate failure paths."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from neurokinematics.neural.c104 import (MLP, _read_split, _save_checkpoint, contracts,
    controlled_subset, load_checkpoint, load_data, predict, reject_shifted_labels,
    validate_rows, validate_split_pair)


@pytest.fixture(scope="module")
def rows():
    return load_data(label_fk=False)


def test_real_pair_budget_and_sealed_test(rows):
    train, validation = rows
    assert len(train.pair_id) == 16800 and len(validation.pair_id) == 3600
    assert train.label_present.sum() == 15204 and validation.label_present.sum() == 3249
    assert ((train.mode == "wide") & ~train.label_present).sum() == 1596
    assert ((validation.mode == "wide") & ~validation.label_present).sum() == 351
    assert train.pose_only.shape == (16800, 7)
    assert train.conditioned.shape == (16800, 13)
    assert not set(train.group_id.tolist()) & set(validation.group_id.tolist())


def test_pilot_subset_and_shifted_label_rejection(rows):
    pilot, monitor = controlled_subset(*rows)
    assert len(pilot.pair_id) == 64 and len(monitor.pair_id) == 32
    assert all(pilot.mode == "local") and all(monitor.mode == "local")
    result = reject_shifted_labels(pilot)
    assert result["status"] == "PASS_REJECTED"
    assert result["rejected_rows"] > 0


def test_mlp_parameter_budget_and_input_width():
    assert sum(p.numel() for p in MLP("pose_only").parameters()) == 135174
    assert sum(p.numel() for p in MLP("conditioned").parameters()) == 136710
    assert MLP("pose_only")(torch.zeros(2, 7)).shape == (2, 6)
    assert MLP("conditioned")(torch.zeros(2, 13)).shape == (2, 6)
    with pytest.raises(RuntimeError):
        MLP("pose_only")(torch.zeros(2, 13))


def test_checkpoint_identity_and_feature_order(tmp_path: Path, rows):
    model = MLP("pose_only")
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    record = _save_checkpoint(path, model, optimizer, "pose_only", 2026100201, 1, 1.0)
    assert record["bytes"] > 0
    loaded, metadata = load_checkpoint(path, expected_variant="pose_only")
    sample = rows[1]
    result = predict(loaded, metadata, sample.position[0], sample.quaternion[0], None)
    assert len(result["q_raw_rad"]) == 6
    with pytest.raises(ValueError, match="feature order"):
        predict(loaded, {**metadata, "input_order": list(reversed(metadata["input_order"]))},
                sample.position[0], sample.quaternion[0], None)
    with pytest.raises(ValueError, match="noncanonical quaternion"):
        predict(loaded, metadata, sample.position[0], -sample.quaternion[0], None)
    payload = torch.load(path, weights_only=True)
    payload["robot_urdf_sha256"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="robot_urdf_sha256"):
        load_checkpoint(path)


def test_missing_label_and_nonfinite_are_not_supervised(rows):
    train, _ = rows
    missing = np.flatnonzero(~train.label_present)
    assert len(missing) == 1596
    assert np.isnan(train.q_target[missing]).all()
    assert np.isfinite(train.target_normalized).all()
    assert np.all(train.target_normalized[missing] == 0)


@pytest.mark.parametrize("field,index,value,message", [
    ("target_normalized", 0, np.array([np.nan] * 6, dtype=np.float32), "nonfinite"),
    ("q_target", 0, np.array([np.nan] * 6), "label sentinel"),
    ("q_current", 0, np.array([100.] * 6), "joint limit"),
    ("position", 0, np.array([np.nan] * 3), "nonfinite"),
])
def test_invalid_training_row_fails_fast(rows, field, index, value, message):
    train, _ = rows
    changed = getattr(train, field).copy()
    changed[index] = value
    with pytest.raises(ValueError, match=message):
        validate_rows(replace(train, **{field: changed}))


def test_feature_swap_and_missing_mask_fail_fast(rows):
    train, _ = rows
    features = train.pose_only.copy()
    features[:, [0, 1]] = features[:, [1, 0]]
    with pytest.raises(ValueError, match="feature order"):
        validate_rows(replace(train, pose_only=features))
    missing = int(np.flatnonzero(~train.label_present)[0])
    target = train.target_normalized.copy()
    target[missing, 0] = 0.5
    with pytest.raises(ValueError, match="missing label"):
        validate_rows(replace(train, target_normalized=target))


def test_split_leakage_and_test_seal(rows):
    train, validation = rows
    groups = validation.group_id.copy()
    groups[0] = train.group_id[0]
    with pytest.raises(ValueError, match="group leakage"):
        validate_split_pair(train, replace(validation, group_id=groups))
    config, _, norm = contracts()
    with pytest.raises(ValueError, match="sealed"):
        _read_split("test", config, norm, label_fk=False)


def test_raw_out_of_limit_is_visible(rows):
    model = MLP("pose_only")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[-1].bias.fill_(10)
    train, _ = rows
    result = predict(model, {"model_variant": "pose_only", "input_order": contracts()[0]["features"]["pose_only"]},
                     train.position[0], train.quaternion[0], None)
    assert result["finite"] and not result["in_limits"]
    assert max(result["q_raw_rad"]) > 10
