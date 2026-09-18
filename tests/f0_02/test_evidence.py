import json

import numpy as np
import pytest

from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.transforms import axis_angle
from neurokinematics.kinematics.validation import (
    SEED, load_config, sample_configurations, sample_hash, run_validation, write_json,
)


def test_deterministic_sampling_and_hash(robot):
    a = sample_configurations(robot)
    b = sample_configurations(robot)
    assert a.shape == (10000, 6) and a.dtype == np.float64
    assert np.array_equal(a, b)
    assert sample_hash(a) == sample_hash(b)
    assert sample_hash(a) == "8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101"
    assert sample_hash(a) != sample_hash(sample_configurations(robot, seed=SEED + 1))
    assert sample_hash(a) == sample_hash(np.asfortranarray(a))
    assert sample_hash(a) == sample_hash(a.astype(">f8"))
    assert np.isfinite(a).all()
    assert (a >= np.array(robot.limits)[:, 0]).all()
    assert (a <= np.array(robot.limits)[:, 1]).all()


def test_1000_sample_smoke(robot):
    summary, diagnostics = run_validation(robot, 1000)
    assert summary["status"] == "INCONCLUSIVE"  # insufficient N for final acceptance
    assert summary["valid_comparison_count"] == 1000
    assert summary["position_error_m"]["max"] <= 1e-9
    assert summary["rotation_frobenius_error"]["max"] <= 1e-9
    assert not diagnostics["invalid_results"]


def test_structured_report_and_failure_detection(robot, tmp_path):
    class WrongFK(IndependentFK):
        def forward_kinematics(self, q):
            t = super().forward_kinematics(q)
            t[0, 3] += 1e-4
            return t
    summary, diagnostics = run_validation(robot, 3, custom=WrongFK(robot))
    assert summary["status"] == "FAIL" and summary["threshold_exceeded_count"] == 3
    assert len(diagnostics["failed_or_near_threshold"]) == 3
    assert summary["worst_position"]["q"]
    path = tmp_path / "result.json"
    write_json(path, summary)
    assert json.loads(path.read_text()) == summary


def test_rotation_only_failure_is_not_masked_by_position(robot):
    class WrongRotation(IndependentFK):
        def forward_kinematics(self, q):
            t = super().forward_kinematics(q)
            t[:3, :3] = t[:3, :3] @ axis_angle([1, 0, 0], 1e-4)
            return t
    summary, diagnostics = run_validation(robot, 2, custom=WrongRotation(robot))
    assert summary["position_error_m"]["max"] <= 1e-9
    assert summary["rotation_frobenius_error"]["max"] > 1e-9
    assert summary["threshold_exceeded_count"] == 2
    assert summary["status"] == "FAIL"


def test_near_threshold_diagnostics_preserve_samples(robot):
    class NearThreshold(IndependentFK):
        def forward_kinematics(self, q):
            t = super().forward_kinematics(q)
            t[0, 3] += 0.95e-9
            return t
    summary, diagnostics = run_validation(robot, 2, custom=NearThreshold(robot))
    assert summary["threshold_exceeded_count"] == 0
    assert len(diagnostics["failed_or_near_threshold"]) == 2


@pytest.mark.parametrize("fault", ["nan", "float32", "bad_rotation", "bad_last_row", "bad_shape"])
def test_invalid_results_never_pass(robot, fault):
    class InvalidFK(IndependentFK):
        def forward_kinematics(self, q):
            t = super().forward_kinematics(q)
            if fault == "nan":
                t[0, 0] = np.nan
            elif fault == "float32":
                t = t.astype(np.float32)
            elif fault == "bad_rotation":
                t[:3, :3] = 0
            elif fault == "bad_last_row":
                t[3, 0] = 1
            else:
                t = t[:3]
            return t
    summary, diagnostics = run_validation(robot, 2, custom=InvalidFK(robot))
    assert summary["status"] == "FAIL"
    assert summary["invalid_result_count"] == 2
    assert summary["nonfinite_result_count"] == (2 if fault == "nan" else 0)
    assert len(diagnostics["invalid_results"]) == 2
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("key,value", [("sample_count", 1000), ("seed", 1), ("bit_generator", "MT19937"), ("position_threshold_m", 1e-8), ("rotation_frobenius_threshold", 1e-8)])
def test_frozen_config_rejects_changes(tmp_path, key, value):
    config = load_config()
    config[key] = value
    path = tmp_path / "config.json"
    write_json(path, config)
    with pytest.raises(ValueError, match="frozen T-F02 config"):
        load_config(path)
