"""Acceptance tests for the T-F00 environment check."""

from __future__ import annotations

import json

import pytest

from neurokinematics.environment_check import build_report, main


def test_t_f00_environment_contract() -> None:
    report = build_report()

    assert report["test_id"] == "T-F00"
    assert report["status"] == "PASS", report
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["python_version"]["passed"]
    assert checks["python_architecture_bits"]["passed"]
    assert checks["numpy_import"]["passed"]
    assert checks["numpy_float64_bits"]["passed"]
    assert checks["pinocchio_version"]["actual"] == "4.1.0"
    assert checks["pinocchio_version"]["passed"]
    assert checks["pinocchio_se3_identity"]["passed"]


def test_t_f00_cli_writes_the_same_json_it_prints(tmp_path, capsys) -> None:
    output = tmp_path / "environment.json"

    exit_code = main(["--output", str(output)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["status"] == "PASS"
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(stdout)


def test_t_f00_cli_requires_an_existing_output_parent(tmp_path) -> None:
    output = tmp_path / "missing" / "environment.json"

    with pytest.raises(SystemExit) as exc_info:
        main(["--output", str(output)])

    assert exc_info.value.code == 2
    assert not output.exists()

