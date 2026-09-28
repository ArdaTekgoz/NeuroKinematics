"""C1-01 contract and failure-path tests; external plugin smoke is Linux-only."""

import json
import sys
import time

import pytest

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.core.contract import (ROOT, SolveRequest, load_contract, solver_config_hash,
                                            validate_wire_request, wxyz_to_xyzw, xyzw_to_wxyz)
from neurokinematics.core.runner import FROZEN_QUERY_PATH, load_queries, smoke_gate, smoke_selection
from neurokinematics.core.results import evaluate_attempt, validate_result_record
from neurokinematics.core import results
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.core.worker import LocalWorker, WorkerError, WorkerReply


@pytest.fixture(scope="module")
def frozen():
    config = load_contract()
    rows, _ = load_queries(FROZEN_QUERY_PATH, config)
    return config, rows


def test_frozen_request_omits_offline_target_and_rejects_joint_mutation(frozen):
    config, rows = frozen
    solver = config["solvers"][0]
    request = SolveRequest.from_query(rows[0], solver, config, 50).with_deadline(time.monotonic_ns())
    wire = request.wire()
    assert "q_target" not in wire
    assert validate_wire_request(wire, config) == request
    changed = {**wire, "joint_order": list(reversed(wire["joint_order"]))}
    with pytest.raises(ValueError, match="joint order"):
        validate_wire_request(changed, config)
    changed = {**wire, "q_current": [float("nan"), *wire["q_current"][1:]]}
    with pytest.raises(ValueError, match="finite"):
        validate_wire_request(changed, config)
    changed = {**wire, "q_target": rows[0]["q_target"]}
    with pytest.raises(ValueError, match="fields"):
        validate_wire_request(changed, config)


def test_pose_order_round_trip():
    quaternion = (0.5, -0.5, 0.5, -0.5)
    assert wxyz_to_xyzw(quaternion) == (-0.5, 0.5, -0.5, 0.5)
    assert xyzw_to_wxyz(wxyz_to_xyzw(quaternion)) == quaternion


def test_worker_rejects_false_success_and_identity_mismatch(frozen):
    config, rows = frozen
    solver = config["solvers"][0]
    request = SolveRequest.from_query(rows[0], solver, config, 50)
    reply = {"query_id": request.query_id, "solver_id": request.solver_id,
             "solver_config_sha256": request.solver_config_sha256,
             "native_status": "SUCCESS", "termination_reason": "OK", "q_candidate": None,
             "iterations": None, "iteration_availability": "NOT_AVAILABLE",
             "solver_internal_elapsed_ns": None, "error_class": None}
    with pytest.raises(WorkerError, match="without a candidate"):
        WorkerReply.parse(json.dumps(reply), request)
    reply["q_candidate"] = list(request.q_current)
    reply["solver_id"] = "wrong"
    with pytest.raises(WorkerError, match="solver_id mismatch"):
        WorkerReply.parse(json.dumps(reply), request)


def test_process_exit_and_timeout_classes(frozen, tmp_path):
    config, rows = frozen
    solver = config["solvers"][0]
    request = SolveRequest.from_query(rows[0], solver, config, 10)
    ready = json.dumps({"ready": True, "solver_id": request.solver_id,
                        "solver_config_sha256": request.solver_config_sha256})
    for body, expected in (("sys.exit(3)", "PROCESS_FAILURE"),
                           ("time.sleep(1)", "TIMEOUT")):
        source = f"import sys,time,json\nprint({ready!r}, flush=True)\nfor line in sys.stdin:\n {body}\n"
        worker = LocalWorker([sys.executable, "-c", source], request.solver_id,
                             request.solver_config_sha256, tmp_path / f"{expected}.log")
        try:
            worker.start()
            with pytest.raises(WorkerError) as error:
                worker.call(request)
            assert error.value.kind == expected
        finally:
            worker.stop()


def test_smoke_selection_and_gate_reject_missing_solver(frozen, tmp_path):
    config, rows = frozen
    chosen = smoke_selection(rows)
    assert [(row["subset"], row["start_class"]) for row in chosen] == (
        [("main", "local")] * 4 + [("boundary", "local")] * 2
        + [("singularity", "local")] * 2)
    gate = {"status": "PASS", "query_list_sha256": config["queries"]["list_sha256"],
            "solvers": {"dls/default": {"pass": True}}}
    path = tmp_path / "smoke-gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="mandatory solver"):
        smoke_gate(path, config, FROZEN_QUERY_PATH)


def test_stage1_config_hash_still_frozen(frozen):
    config, _ = frozen
    manifest = strict_json((ROOT / "experiments/C1-01/frozen-hashes.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 17
    assert len({solver_config_hash(solver) for solver in config["solvers"]}) == 5


def test_independent_fk_validation_catches_tampered_geometry(frozen, tmp_path):
    config, rows = frozen
    query, solver = rows[0], config["solvers"][0]

    class KnownCandidateWorker:
        stderr_path = tmp_path / "unused.log"

        def call(self, request):
            return WorkerReply(request.query_id, request.solver_id, request.solver_config_sha256,
                               "SUCCESS", "FIXTURE_CANDIDATE", tuple(query["q_target"]),
                               1, "AVAILABLE", 0, None), 0

    validator = CandidateValidator()
    dataset_hash = strict_json((ROOT / "experiments/F0-05/query-manifest.json").read_text(
        encoding="utf-8"))["dataset_manifest_sha256"]
    record = evaluate_attempt(query, solver, config, 50, 0, KnownCandidateWorker(), validator,
                              config["queries"]["list_sha256"], dataset_hash)
    assert record["profile_b_geometry"]
    validate_result_record(record, query, solver, config, config["queries"]["list_sha256"],
                           dataset_hash, validator)
    changed = {**record, "position_error_m": 0.0005}
    with pytest.raises(ValueError, match="independent position_error_m"):
        validate_result_record(changed, query, solver, config, config["queries"]["list_sha256"],
                               dataset_hash, validator)


def test_late_finite_candidate_is_validated_but_timeout(frozen, tmp_path, monkeypatch):
    config, rows = frozen
    query, solver = rows[0], config["solvers"][0]
    # Exercise a known late call without depending on the host scheduler or
    # Windows Python 3.12's coarse GetTickCount64 monotonic clock.
    ticks = iter((100_000_000, 160_000_000, 161_000_000))
    monkeypatch.setattr(results, "perf_counter_ns", lambda: next(ticks))

    class LateWorker:
        stderr_path = tmp_path / "unused.log"

        def call(self, request):
            return WorkerReply(request.query_id, request.solver_id, request.solver_config_sha256,
                               "SUCCESS", "FIXTURE_LATE", tuple(query["q_target"]),
                               1, "AVAILABLE", 0, None), 0

    dataset_hash = strict_json((ROOT / "experiments/F0-05/query-manifest.json").read_text(
        encoding="utf-8"))["dataset_manifest_sha256"]
    validator = CandidateValidator()
    record = evaluate_attempt(query, solver, config, 50, 0, LateWorker(), validator,
                              config["queries"]["list_sha256"], dataset_hash)
    assert record["profile_b_geometry"] is True
    assert record["common_status"] == "TIMEOUT"
    assert record["profile_b_deadline"] is False
    assert record["q_candidate"] is not None
    validate_result_record(record, query, solver, config, config["queries"]["list_sha256"],
                           dataset_hash, validator)
