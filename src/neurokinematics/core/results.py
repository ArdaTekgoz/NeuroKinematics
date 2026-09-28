"""Independent C1-01 candidate validation and common result semantics."""

from __future__ import annotations

import math
from time import monotonic_ns as perf_counter_ns

from neurokinematics.benchmark.contract import strict_json, validate_structure
from neurokinematics.benchmark.validation import CandidateValidator
from neurokinematics.kinematics.model import load_robot
from neurokinematics.solvers.dls import SolverResult, SolverStatus

from .contract import ROOT, SolveRequest, solver_config_hash
from .worker import LocalWorker, WorkerError, WorkerReply


SCHEMA_PATH = ROOT / "experiments/C1-01/result-schema.json"
FATAL_DEADLINE = {"TIMEOUT", "INVALID_INPUT", "INVALID_OUTPUT", "JOINT_LIMIT_FAILURE",
                  "NUMERICAL_FAILURE", "PROCESS_FAILURE", "INSTALLATION_FAILURE",
                  "ADAPTER_ERROR", "VALIDATION_ERROR", "PROVEN_UNREACHABLE"}


def _validator_status(native: str) -> SolverStatus:
    if native == "SUCCESS":
        return SolverStatus.SUCCESS
    if native == "TIMEOUT":
        return SolverStatus.TIMEOUT
    if native == "INVALID_INPUT":
        return SolverStatus.INVALID_INPUT
    if native == "NUMERICAL_FAILURE":
        return SolverStatus.NUMERICAL_FAILURE
    return SolverStatus.UNRESOLVED


def _common_status(native: str, candidate: tuple[float, ...] | None, verdict,
                   failure: WorkerError | None, error_class: str | None) -> str:
    # CandidateValidator returns this verdict (rather than raising) when its
    # independent FK cannot be evaluated. Preserve that infrastructure defect,
    # including when the solver itself was late or reported a numerical error.
    if (candidate is not None and verdict.joint_limits == "PASS"
            and verdict.status == SolverStatus.NUMERICAL_FAILURE
            and verdict.position_error_m is None and verdict.orientation_error_rad is None):
        return "VALIDATION_ERROR"
    if failure is not None:
        return failure.kind if failure.kind in FATAL_DEADLINE else "ADAPTER_ERROR"
    if error_class in FATAL_DEADLINE:
        return error_class
    if native == "TIMEOUT":
        return "TIMEOUT"
    if native == "INVALID_INPUT":
        return "INVALID_INPUT"
    if native == "NUMERICAL_FAILURE":
        return "NUMERICAL_FAILURE"
    if native in FATAL_DEADLINE:
        return native
    if candidate is not None and verdict.joint_limits == "FAIL":
        return "JOINT_LIMIT_FAILURE"
    if native == "SUCCESS" and verdict.profile_b_geometry:
        return "SUCCESS"
    return "UNRESOLVED"


def evaluate_attempt(query: dict, solver: dict, config: dict, deadline_ms: int, pass_index: int,
                     worker: LocalWorker, validator: CandidateValidator, query_list_sha256: str,
                     dataset_manifest_sha256: str, *, seed: int | None = None) -> dict:
    """Time call/IPC plus independent Pinocchio validation on one frozen query."""
    if pass_index < 0 or pass_index >= config["benchmark"]["measurement_passes"]:
        raise ValueError("measurement pass outside frozen range")
    started = perf_counter_ns()
    request = SolveRequest.from_query(query, solver, config, deadline_ms, seed)
    reply: WorkerReply | None = None
    failure: WorkerError | None = None
    request = request.with_deadline(started)
    try:
        reply, _ = worker.call(request)
    except WorkerError as exc:
        failure = exc
    called = perf_counter_ns()
    native = reply.native_status if reply is not None else "NOT_AVAILABLE"
    candidate = reply.q_candidate if reply is not None else None
    proxy = SolverResult(_validator_status(native if failure is None else failure.kind),
                         reply.termination_reason if reply is not None else str(failure),
                         candidate, reply.iterations if reply is not None else None,
                         None, None, 0)
    try:
        verdict = validator.validate(proxy, request.target_position_m, request.target_quaternion_wxyz)
    except Exception as exc:
        # Validator defects cannot be turned into solver mathematical failure.
        raise WorkerError("VALIDATION_ERROR", f"independent validator failed: {exc}") from exc
    validated = perf_counter_ns()
    transport_ns = called - started
    validation_ns = validated - called
    total_ns = validated - started
    inner = reply.solver_internal_elapsed_ns if reply is not None else None
    if inner is not None and inner > transport_ns:
        raise WorkerError("INVALID_OUTPUT", "worker internal time exceeds enclosing call")
    adapter_ns = transport_ns if inner is None else transport_ns - inner
    common = _common_status(native, candidate, verdict, failure,
                            reply.error_class if reply is not None else None)
    if common != "VALIDATION_ERROR" and reply is not None and transport_ns > request.deadline_ns:
        common = "TIMEOUT"
    admissible = common not in FATAL_DEADLINE and total_ns <= request.deadline_ns
    record = {
        "schema_version": "1.0.0", "query_id": request.query_id,
        "query_group_id": query["query_group_id"], "query_list_sha256": query_list_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "subset": query["subset"], "start_class": query["start_class"],
        "q_current": list(request.q_current), "target_position_m": list(request.target_position_m),
        "target_quaternion_wxyz": list(request.target_quaternion_wxyz),
        "solver_id": solver["id"], "solver_name": solver["name"],
        "solver_variant": solver["variant"], "solver_version": solver["version"],
        "solver_config_sha256": request.solver_config_sha256,
        "q_candidate": None if candidate is None else list(candidate),
        "native_status": native, "common_status": common,
        "termination_reason": reply.termination_reason if reply is not None else str(failure),
        "timed_out": common == "TIMEOUT", "iterations": reply.iterations if reply is not None else None,
        "iteration_availability": reply.iteration_availability if reply is not None else "NOT_AVAILABLE",
        "solver_internal_elapsed_ns": inner, "adapter_ipc_elapsed_ns": adapter_ns,
        "transport_elapsed_ns": transport_ns, "validation_elapsed_ns": validation_ns,
        "total_elapsed_ns": total_ns, "position_error_m": verdict.position_error_m,
        "orientation_error_rad": verdict.orientation_error_rad,
        "orientation_error_deg": verdict.orientation_error_deg,
        "profile_a_geometry": verdict.profile_a_geometry,
        "profile_b_geometry": verdict.profile_b_geometry,
        "profile_a_deadline": bool(verdict.profile_a_geometry and admissible),
        "profile_b_deadline": bool(verdict.profile_b_geometry and admissible),
        "joint_limits": verdict.joint_limits, "collision": "NOT_CHECKED",
        "validation_status": verdict.status.value,
        "reachability": "KNOWN_REACHABLE",
        "error_class": ("VALIDATION_ERROR" if common == "VALIDATION_ERROR" else
                        failure.kind if failure is not None else
                        "TIMEOUT" if common == "TIMEOUT" and native != "TIMEOUT" else reply.error_class),
        "stderr_log_ref": str(worker.stderr_path) if failure is not None else None,
        "deadline_profile_ms": deadline_ms, "measurement_pass_index": pass_index,
        "frame": request.base_frame, "tcp": request.tcp_frame,
        "quaternion_order": "wxyz", "joint_order": list(request.joint_order),
    }
    return record


def validate_result_record(record: dict, query: dict, solver: dict, config: dict,
                           query_hash: str, dataset_hash: str,
                           validator: CandidateValidator | None = None) -> None:
    """Offline structural, binding and independent FK verification."""
    schema = strict_json(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_structure(record, schema)
    if (record["query_id"] != query["query_id"] or record["query_group_id"] != query["query_group_id"]
            or record["subset"] != query["subset"] or record["start_class"] != query["start_class"]
            or record["q_current"] != query["q_current"]
            or record["target_position_m"] != query["target_position_m"]
            or record["target_quaternion_wxyz"] != query["target_quaternion_wxyz"]):
        raise ValueError("result/query binding mismatch")
    if record["query_list_sha256"] != query_hash or record["dataset_manifest_sha256"] != dataset_hash:
        raise ValueError("result frozen manifest binding mismatch")
    if (record["solver_id"] != solver["id"] or record["solver_name"] != solver["name"]
            or record["solver_variant"] != solver["variant"] or record["solver_version"] != solver["version"]
            or record["solver_config_sha256"] != solver_config_hash(solver)):
        raise ValueError("solver identity/config binding mismatch")
    if (record["frame"], record["tcp"], record["quaternion_order"], record["joint_order"]) != (
            config["robot"]["base_frame"], config["robot"]["tcp_frame"], "wxyz", config["robot"]["joint_order"]):
        raise ValueError("frame/TCP/quaternion/joint binding mismatch")
    if record["collision"] != "NOT_CHECKED" or record["reachability"] != "KNOWN_REACHABLE":
        raise ValueError("unsupported collision or reachability claim")
    if record["deadline_profile_ms"] not in config["benchmark"]["deadline_profiles_ms"]:
        raise ValueError("unfrozen deadline")
    if record["total_elapsed_ns"] != record["transport_elapsed_ns"] + record["validation_elapsed_ns"]:
        raise ValueError("total timing mismatch")
    inner = record["solver_internal_elapsed_ns"]
    if record["adapter_ipc_elapsed_ns"] != record["transport_elapsed_ns"] - (inner or 0):
        raise ValueError("adapter timing mismatch")
    if (record["iterations"] is None) != (record["iteration_availability"] == "NOT_AVAILABLE"):
        raise ValueError("iteration availability mismatch")
    if record["timed_out"] != (record["common_status"] == "TIMEOUT"):
        raise ValueError("timeout status mismatch")
    if record["common_status"] not in config["benchmark"]["status_codes"]:
        raise ValueError("unknown common status")
    source_status = record["common_status"] if record["native_status"] == "NOT_AVAILABLE" else record["native_status"]
    proxy = SolverResult(_validator_status(source_status), record["termination_reason"],
                         record["q_candidate"], record["iterations"], None, None, 0)
    validator = validator or CandidateValidator(load_robot())
    verdict = validator.validate(proxy, record["target_position_m"], record["target_quaternion_wxyz"])
    if record["native_status"] == "SUCCESS" and record["q_candidate"] is None:
        raise ValueError("native success lacks candidate")
    recorded_failure = (WorkerError(record["error_class"], record["termination_reason"])
                        if record["native_status"] == "NOT_AVAILABLE" and record["error_class"] else None)
    expected_common = _common_status(record["native_status"], record["q_candidate"],
                                     verdict, recorded_failure, record["error_class"])
    if (expected_common != "VALIDATION_ERROR" and record["native_status"] != "NOT_AVAILABLE"
            and record["transport_elapsed_ns"] > record["deadline_profile_ms"] * 1_000_000):
        expected_common = "TIMEOUT"
    if record["common_status"] != expected_common:
        raise ValueError("common status does not match native/independent verdict")
    for key in ("position_error_m", "orientation_error_rad", "orientation_error_deg"):
        actual, recorded = getattr(verdict, key), record[key]
        if (actual is None) != (recorded is None) or (actual is not None and not math.isclose(actual, recorded, rel_tol=0, abs_tol=1e-10)):
            raise ValueError(f"independent {key} mismatch")
    for key in ("profile_a_geometry", "profile_b_geometry", "joint_limits", "validation_status"):
        expected = verdict.status.value if key == "validation_status" else getattr(verdict, key)
        if record[key] != expected:
            raise ValueError(f"independent {key} mismatch")
    admissible = (record["common_status"] not in FATAL_DEADLINE
                  and record["total_elapsed_ns"] <= record["deadline_profile_ms"] * 1_000_000)
    for profile in ("a", "b"):
        if record[f"profile_{profile}_deadline"] != bool(record[f"profile_{profile}_geometry"] and admissible):
            raise ValueError("independent deadline mismatch")
