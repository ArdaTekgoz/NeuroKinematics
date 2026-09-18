"""Always included by test-f02; no slow skip/marker excludes the 10,000-q gate."""

from neurokinematics.kinematics.validation import run_validation


def test_tf02_final_10000_float64(robot, record_property):
    summary, diagnostics = run_validation(robot, 10000)
    for key in ("sample_count", "seed", "sample_sha256", "status"):
        record_property(key, summary[key])
    record_property("max_position_error_m", summary["position_error_m"]["max"])
    record_property("max_rotation_frobenius_error", summary["rotation_frobenius_error"]["max"])
    assert summary["sample_shape"] == [10000, 6]
    assert summary["sample_sha256"] == "8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101"
    assert summary["dtype"] == "float64"
    assert summary["valid_comparison_count"] == 10000
    assert summary["position_error_m"]["max"] <= 1e-9
    assert summary["rotation_frobenius_error"]["max"] <= 1e-9
    assert summary["threshold_exceeded_count"] == 0
    assert summary["nonfinite_result_count"] == 0
    assert summary["invalid_result_count"] == 0
    assert summary["status"] == "PASS"
    assert diagnostics["invalid_results"] == []
