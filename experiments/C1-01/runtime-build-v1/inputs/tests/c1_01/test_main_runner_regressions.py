"""Fault evidence and terminal restart regressions before the common main run."""

import json
import sys
import time
from types import SimpleNamespace

import pytest

from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.core import contract, dls_worker, results, runner
from neurokinematics.core.contract import SolveRequest, load_contract, solver_config_hash, validate_wire_request
from neurokinematics.solvers.dls import SolverStatus
from neurokinematics.core.worker import LocalWorker, WorkerError, WorkerReply


@pytest.fixture(scope="module")
def frozen():
    config = load_contract()
    rows, manifest = runner.load_queries(runner.FROZEN_QUERY_PATH, config)
    return config, rows, manifest


def test_terminal_timeout_does_not_start_unused_worker(monkeypatch, tmp_path, frozen):
    config, rows, manifest = frozen
    plan = {**config, "benchmark": {**config["benchmark"], "measurement_passes": 1}}
    events = []

    class Worker:
        process = None

        def __init__(self, *args):
            pass

        def start(self):
            events.append("start")
            if events.count("start") > 2:
                raise WorkerError("INVALID_OUTPUT", "unused replacement failed")
            self.process = object()
            return 1

        def call(self, request, *, late_reply_window_s):
            events.append("warm")

        def stop(self):
            self.process = None

    def measured(*args):
        events.append("measure")
        args[5].process = None
        return {"common_status": "TIMEOUT", "profile_a_geometry": False,
                "profile_b_geometry": False, "profile_a_deadline": False,
                "profile_b_deadline": False}

    monkeypatch.setattr(runner, "LocalWorker", Worker)
    monkeypatch.setattr(runner, "evaluate_attempt", measured)
    monkeypatch.setattr(runner, "validate_result_record", lambda *args: None)
    result = runner.run_method(config["solvers"][0], plan, rows[:1], manifest,
                               tmp_path, None, rows[:20], mode="benchmark")
    assert events == (["start"] + ["warm"] * 20 + ["measure"]) * 2
    assert result["record_count"] == 2
    assert result["worker_start_error"] is None
    assert result["status"] == "MEASURED_UNVERIFIED"
    assert result["restart_warmups"] == 0
    assert result["warmup_calls"] == 40


def test_progress_observes_flushed_measured_rows(monkeypatch, tmp_path, frozen):
    config, rows, manifest = frozen
    plan = {**config, "benchmark": {**config["benchmark"], "measurement_passes": 1,
                                   "deadline_profiles_ms": [10]}}

    class Worker:
        process = object()

        def __init__(self, *args):
            pass

        def start(self):
            return 1

        def call(self, request, *, late_reply_window_s):
            pass

        def stop(self):
            self.process = None

    record = {"common_status": "SUCCESS", "profile_a_geometry": True,
              "profile_b_geometry": True, "profile_a_deadline": True,
              "profile_b_deadline": True}
    observed = []

    def progress(event):
        persisted = (tmp_path / "dls-default-benchmark.jsonl").read_bytes().splitlines()
        assert len(persisted) == event["record_count"]
        observed.append(event)

    monkeypatch.setattr(runner, "LocalWorker", Worker)
    monkeypatch.setattr(runner, "evaluate_attempt", lambda *args: dict(record))
    monkeypatch.setattr(runner, "validate_result_record", lambda *args: None)
    runner.run_method(config["solvers"][0], plan, rows[:1001], manifest, tmp_path,
                      None, rows[:20], mode="benchmark", progress_callback=progress)
    assert [event["record_count"] for event in observed] == [1000, 1001]
    assert all(event["expected_record_count"] == 1001 and event["solver_id"] == "dls/default"
               and event["deadline_profile_ms"] == 10 and event["measurement_pass_index"] == 0
               for event in observed)


def candidate_worker(query, native, tmp_path, candidate=None):
    class Worker:
        stderr_path = tmp_path / "unused.log"

        def call(self, request):
            return WorkerReply(request.query_id, request.solver_id, request.solver_config_sha256,
                               native, "FAULT_FIXTURE", tuple(query["q_target"] if candidate is None else candidate),
                               1, "AVAILABLE", 0, None), 0

    return Worker()


@pytest.mark.parametrize("native,transport_ns", [("SUCCESS", 1), ("SUCCESS", 11_000_000),
                                                ("TIMEOUT", 11_000_000), ("NUMERICAL_FAILURE", 1)])
def test_independent_fk_failure_is_explicit_even_after_deadline(
        monkeypatch, tmp_path, frozen, native, transport_ns):
    config, rows, manifest = frozen
    query, solver = rows[0], config["solvers"][0]
    validator = CandidateValidator()

    def broken_fk(q):
        raise RuntimeError("injected independent FK failure")

    monkeypatch.setattr(validator.reference, "reference_forward_kinematics", broken_fk)
    ticks = iter((0, transport_ns, transport_ns + 1))
    monkeypatch.setattr(results, "perf_counter_ns", lambda: next(ticks))
    record = results.evaluate_attempt(query, solver, config, 10, 0,
                                      candidate_worker(query, native, tmp_path), validator,
                                      config["queries"]["list_sha256"], manifest["dataset_manifest_sha256"])
    assert record["common_status"] == "VALIDATION_ERROR"
    assert record["error_class"] == "VALIDATION_ERROR"
    assert record["validation_status"] == "NUMERICAL_FAILURE"
    assert record["q_candidate"] == query["q_target"]
    assert not record["profile_a_deadline"] and not record["profile_b_deadline"]
    results.validate_result_record(record, query, solver, config, config["queries"]["list_sha256"],
                                   manifest["dataset_manifest_sha256"], validator)
    hidden = {**record, "common_status": "UNRESOLVED", "error_class": None}
    with pytest.raises(ValueError, match="common status"):
        results.validate_result_record(hidden, query, solver, config, config["queries"]["list_sha256"],
                                       manifest["dataset_manifest_sha256"], validator)


@pytest.mark.parametrize("failure", ["NUMERICAL_FAILURE", "JOINT_LIMIT_FAILURE"])
def test_solver_numeric_and_limit_failures_remain_distinct(tmp_path, frozen, failure):
    config, rows, manifest = frozen
    query, solver = rows[0], config["solvers"][0]
    candidate = list(query["q_target"])
    native = failure
    if failure == "JOINT_LIMIT_FAILURE":
        candidate[0] += 100
        native = "SUCCESS"
    validator = CandidateValidator()
    record = results.evaluate_attempt(query, solver, config, 50, 0,
                                      candidate_worker(query, native, tmp_path, candidate), validator,
                                      config["queries"]["list_sha256"], manifest["dataset_manifest_sha256"])
    assert record["common_status"] == failure
    if failure == "NUMERICAL_FAILURE":
        assert record["position_error_m"] is not None
    results.validate_result_record(record, query, solver, config, config["queries"]["list_sha256"],
                                   manifest["dataset_manifest_sha256"], validator)


@pytest.mark.parametrize("phase", ["ready", "reply"])
def test_invalid_protocol_line_is_preserved_and_rejected(tmp_path, frozen, phase):
    config, rows, _ = frozen
    solver = config["solvers"][0]
    request = SolveRequest.from_query(rows[0], solver, config, 50)
    ready = json.dumps({"ready": True, "solver_id": solver["id"],
                        "solver_config_sha256": solver_config_hash(solver)})
    bad_line = "\x1b[31mRTPS diagnostic: segment failure\x1b[m\n"
    source = ("import sys,time\n"
              f"sys.stdout.write({(bad_line if phase == 'ready' else ready + chr(10))!r});sys.stdout.flush()\n"
              f"for line in sys.stdin:\n sys.stdout.write({bad_line!r});sys.stdout.flush();time.sleep(1)\n")
    log = tmp_path / "protocol.log"
    client = LocalWorker([sys.executable, "-c", source], solver["id"], solver_config_hash(solver), log)
    try:
        with pytest.raises(WorkerError) as failure:
            client.start()
            client.call(request)
        assert failure.value.kind == "INVALID_OUTPUT"
        assert client.process is None
    finally:
        client.stop()
    evidence = json.loads(log.read_text(encoding="utf-8"))
    assert evidence["event"] == "C101_PROTOCOL_ERROR"
    assert evidence["phase"] == phase
    assert evidence["raw_text"] == bad_line
    assert evidence["raw_repr"] == repr(bad_line)
    assert evidence["utf8_hex"] == bad_line.encode("utf-8").hex()


def test_wire_requires_bounded_monotonic_expiry(monkeypatch, frozen):
    config, rows, _ = frozen
    monkeypatch.setattr(contract, "monotonic_ns", lambda: 100_000_000)
    request = SolveRequest.from_query(rows[0], config["solvers"][0], config, 10)
    with pytest.raises(ValueError, match="expiry"):
        validate_wire_request(request.wire(), config)
    with pytest.raises(ValueError, match="exceeds frozen budget"):
        validate_wire_request(request.with_deadline(100_000_001).wire(), config)
    expired = request.with_deadline(80_000_000)
    assert validate_wire_request(expired.wire(), config) == expired
    missing = expired.wire()
    del missing["expires_at_monotonic_ns"]
    with pytest.raises(ValueError, match="fields"):
        validate_wire_request(missing, config)


@pytest.mark.parametrize("worker_now,expected_budget", [(104_000_000, 6_000_000),
                                                        (110_000_000, None),
                                                        (120_000_000, None)])
def test_dls_deducts_ipc_and_never_dispatches_expired_request(
        monkeypatch, frozen, worker_now, expected_budget):
    config, rows, _ = frozen
    request = SolveRequest.from_query(rows[0], config["solvers"][0], config, 10).with_deadline(100_000_000)
    budgets = []

    class DLS:
        def solve(self, *args, deadline_ns):
            budgets.append(deadline_ns)
            return SimpleNamespace(status=SolverStatus.TIMEOUT, termination_reason="FIXTURE",
                                   q_candidate=None, iterations=1, elapsed_ns=deadline_ns)

    monkeypatch.setattr(dls_worker, "monotonic_ns", lambda: worker_now)
    reply = dls_worker.solve_request(request, DLS())
    assert budgets == ([] if expected_budget is None else [expected_budget])
    assert reply["native_status"] == "TIMEOUT"
    if expected_budget is None:
        assert reply["termination_reason"] == "DEADLINE_EXPIRED_BEFORE_SOLVE"
        assert reply["solver_internal_elapsed_ns"] == 0


def test_input_preparation_consumes_measured_budget(monkeypatch, tmp_path, frozen):
    config, rows, manifest = frozen
    ticks = iter((100_000_000, 112_000_000, 113_000_000))
    monkeypatch.setattr(results, "perf_counter_ns", lambda: next(ticks))
    original = SolveRequest.from_query
    events = []

    def convert(*args):
        events.append("convert")
        return original(*args)

    class Worker:
        stderr_path = tmp_path / "unused.log"

        def call(self, request):
            assert events == ["convert"]
            assert request.expires_at_monotonic_ns == 110_000_000
            return WorkerReply(request.query_id, request.solver_id, request.solver_config_sha256,
                               "TIMEOUT", "EXPIRED", None, 0, "AVAILABLE", 0, None), 0

    monkeypatch.setattr(SolveRequest, "from_query", convert)
    record = results.evaluate_attempt(rows[0], config["solvers"][0], config, 10, 0, Worker(),
                                      CandidateValidator(), config["queries"]["list_sha256"],
                                      manifest["dataset_manifest_sha256"])
    assert record["transport_elapsed_ns"] == 12_000_000
    assert record["total_elapsed_ns"] == 13_000_000
    assert record["common_status"] == "TIMEOUT"


def test_real_dls_worker_does_not_renew_expired_ipc_budget(tmp_path, frozen):
    config, rows, _ = frozen
    solver = config["solvers"][0]
    request = SolveRequest.from_query(rows[0], solver, config, 10)
    client = LocalWorker([sys.executable, "-m", "neurokinematics.core.dls_worker"],
                         solver["id"], solver_config_hash(solver), tmp_path / "expired.log")
    try:
        client.start()
        request = request.with_deadline(time.monotonic_ns() - 20_000_000)
        reply, _ = client.call(request, late_reply_window_s=3.0)
        assert reply.native_status == "TIMEOUT"
        assert reply.termination_reason == "DEADLINE_EXPIRED_BEFORE_SOLVE"
        assert reply.solver_internal_elapsed_ns == 0
        assert reply.q_candidate is None
    finally:
        client.stop()
