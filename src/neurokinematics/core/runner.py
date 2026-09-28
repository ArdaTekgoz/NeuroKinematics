"""C1-01 gated, serial five-solver execution over the frozen F0-05 queries."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
from typing import Callable

import numpy as np

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.benchmark.queries import encode_query
from neurokinematics.benchmark.runner import percentiles
from neurokinematics.benchmark.validation import CandidateValidator

from .contract import ROOT, load_contract, sha256, solver_config_hash, SolveRequest
from .results import evaluate_attempt, validate_result_record
from .worker import LocalWorker, WorkerError


QUERY_MANIFEST = ROOT / "experiments/F0-05/query-manifest.json"
FROZEN_QUERY_PATH = ROOT / "data/generated/F0-05/acceptance/run-a/query-list.jsonl"
SOLVER_IDS = ("dls/default", "kdl/default", "trac_ik/speed", "pick_ik/local", "pick_ik/global")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def load_queries(path: Path, config: dict) -> tuple[list[dict], dict]:
    manifest = strict_json(QUERY_MANIFEST.read_text(encoding="utf-8"))
    if sha256(QUERY_MANIFEST) != config["queries"]["manifest_sha256"]:
        raise ValueError("F0-05 query manifest hash mismatch")
    if sha256(path) != config["queries"]["list_sha256"]:
        raise ValueError("F0-05 query list hash mismatch")
    rows = []
    counts = Counter()
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n") or b"\r" in raw:
                raise ValueError("query JSONL newline violation")
            row = strict_json(raw.decode("utf-8"))
            if encode_query(row) != raw:
                raise ValueError("query JSONL canonical encoding violation")
            rows.append(row)
            counts[row["subset"]] += 1
    if len(rows) != config["queries"]["count"] or dict(counts) != config["queries"]["subsets"]:
        raise ValueError("query count/subsets mismatch")
    if manifest["dataset_manifest_sha256"] != sha256(ROOT / "experiments/F0-04/dataset-manifest.json"):
        raise ValueError("frozen dataset manifest mismatch")
    return rows, manifest


def smoke_selection(rows: list[dict]) -> list[dict]:
    """Fixed, disclosed selection: four easy and two each boundary/singularity local starts."""
    limit = {"main": 4, "boundary": 2, "singularity": 2}
    chosen: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        subset = row["subset"]
        if row["start_class"] == "local" and len(chosen[subset]) < limit[subset]:
            chosen[subset].append(row)
    if any(len(chosen[subset]) != count for subset, count in limit.items()):
        raise ValueError("frozen query list lacks smoke rows")
    return [*chosen["main"], *chosen["boundary"], *chosen["singularity"]]


def command_for(solver: dict, external_worker: Path | None, config: dict) -> list[str]:
    solver_id = solver["id"]
    if solver_id == "dls/default":
        return [sys.executable, "-m", "neurokinematics.core.dls_worker"]
    if external_worker is None:
        raise ValueError("external MoveIt worker executable is required")
    return [str(external_worker), "--solver", solver_id,
            "--config-hash", solver_config_hash(solver),
            "--urdf", str(ROOT / config["robot"]["urdf_path"]),
            "--robot-spec", str(ROOT / config["robot"]["joint_limits_source"]),
            "--base", config["robot"]["base_frame"], "--tcp", config["robot"]["tcp_frame"]]


def run_method(solver: dict, config: dict, rows: list[dict], manifest: dict, output: Path,
               external_worker: Path | None, warmup_rows: list[dict], *, mode: str,
               progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Append every measured attempt; process launch and warmup are outside attempts."""
    solver_id = solver["id"]
    safe_id = solver_id.replace("/", "-")
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"{safe_id}-{mode}.jsonl"
    log_path = output / f"{safe_id}-{mode}.stderr.log"
    worker = LocalWorker(command_for(solver, external_worker, config), solver_id,
                         solver_config_hash(solver), log_path)
    validator = CandidateValidator()
    deadlines = (50,) if mode == "smoke" else tuple(config["benchmark"]["deadline_profiles_ms"])
    passes = 1 if mode == "smoke" else config["benchmark"]["measurement_passes"]
    expected = len(rows) * len(deadlines) * passes
    if len(warmup_rows) != config["execution"]["warmup_queries_per_deadline"]:
        raise ValueError("warmup must use the first 20 frozen F0-05 queries")
    status = Counter()
    subset_status = defaultdict(Counter)
    geometry = Counter()
    deadline_success = Counter()
    count = 0
    launches = []
    warmup_ns = 0
    warmup_calls = 0
    restart_warmups = 0
    start_error = None
    started = utc_now()

    def start_warm_worker(deadline_ms):
        nonlocal warmup_ns, warmup_calls
        launches.append(worker.start())
        warm_start = perf_counter_ns()
        try:
            for row in warmup_rows:
                request = SolveRequest.from_query(row, solver, config, deadline_ms)
                # Warmup is excluded from measurement. Let a late reply drain
                # without replacing the process that is being warmed.
                worker.call(request, late_reply_window_s=3.0)
                warmup_calls += 1
        finally:
            warmup_ns += perf_counter_ns() - warm_start

    with raw_path.open("wb") as stream:
        for deadline_ms in deadlines:
            try:
                start_warm_worker(deadline_ms)
            except WorkerError as exc:
                start_error = {"class": exc.kind, "detail": str(exc), "deadline_ms": deadline_ms}
                worker.stop()
                break
            for pass_index in range(passes):
                for query_index, row in enumerate(rows):
                    record = evaluate_attempt(row, solver, config, deadline_ms, pass_index, worker,
                                              validator, config["queries"]["list_sha256"],
                                              manifest["dataset_manifest_sha256"])
                    validate_result_record(record, row, solver, config,
                                           config["queries"]["list_sha256"],
                                           manifest["dataset_manifest_sha256"], validator)
                    stream.write(encode_query(record))
                    count += 1
                    status[record["common_status"]] += 1
                    subset_status[row["subset"]][record["common_status"]] += 1
                    for profile in ("a", "b"):
                        geometry[profile] += record[f"profile_{profile}_geometry"]
                        deadline_success[profile] += record[f"profile_{profile}_deadline"]
                    if progress_callback is not None and (count % 1000 == 0 or count == expected):
                        stream.flush()
                        progress_callback({"solver_id": solver_id, "record_count": count,
                                           "expected_record_count": expected,
                                           "deadline_profile_ms": deadline_ms,
                                           "measurement_pass_index": pass_index})
                    more_in_deadline = query_index + 1 < len(rows) or pass_index + 1 < passes
                    if worker.process is None and more_in_deadline:
                        try:
                            start_warm_worker(deadline_ms)
                            restart_warmups += 1
                        except WorkerError as exc:
                            start_error = {"class": exc.kind, "detail": str(exc), "deadline_ms": deadline_ms}
                            break
                if start_error:
                    break
            worker.stop()
            if start_error:
                break
        if progress_callback is not None and count and count != expected and count % 1000:
            stream.flush()
            progress_callback({"solver_id": solver_id, "record_count": count,
                               "expected_record_count": expected,
                               "deadline_profile_ms": deadline_ms,
                               "measurement_pass_index": pass_index})
    summary = {
        "schema_version": "1.0.0", "task": "C1-01", "mode": mode, "solver_id": solver_id,
        "started_utc": started, "finished_utc": utc_now(), "query_list_sha256": config["queries"]["list_sha256"],
        "solver_config_sha256": solver_config_hash(solver), "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path), "raw_bytes": raw_path.stat().st_size,
        "stderr_path": str(log_path), "stderr_sha256": sha256(log_path) if log_path.exists() else None,
        "record_count": count, "expected_record_count": expected,
        "status_counts": dict(status), "subset_status_counts": {k: dict(v) for k, v in subset_status.items()},
        "profile_a_geometry_count": geometry["a"], "profile_b_geometry_count": geometry["b"],
        "profile_a_deadline_count": deadline_success["a"], "profile_b_deadline_count": deadline_success["b"],
        "worker_launch_elapsed_ns": launches, "warmup_elapsed_ns": warmup_ns,
        "warmup_calls": warmup_calls, "restart_warmups": restart_warmups,
        "worker_start_error": start_error, "environment": {
            "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "NOT_AVAILABLE",
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "numpy": np.__version__, "pid": os.getpid(),
        },
    }
    if mode == "smoke":
        easy_success = any(
            strict_json(line.decode("utf-8"))["profile_b_deadline"]
            for line in raw_path.read_bytes().splitlines(keepends=True)
            if strict_json(line.decode("utf-8"))["subset"] == "main"
        )
        fatal = sum(status[k] for k in ("PROCESS_FAILURE", "INSTALLATION_FAILURE", "ADAPTER_ERROR",
                                        "INVALID_OUTPUT", "VALIDATION_ERROR"))
        summary["pass"] = bool(count == expected and start_error is None and fatal == 0 and easy_success)
    else:
        summary["status"] = "MEASURED_UNVERIFIED" if count == expected and start_error is None else "INCOMPLETE"
    write_json(output / f"{safe_id}-{mode}-summary.json", summary)
    return summary


def smoke_gate(path: Path, config: dict, query_path: Path) -> dict:
    gate = strict_json(path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("query_list_sha256") != config["queries"]["list_sha256"]:
        raise ValueError("five-solver smoke gate is not PASS on frozen queries")
    if set(gate.get("solvers", {})) != set(SOLVER_IDS) or not all(
            gate["solvers"][solver_id]["pass"] for solver_id in SOLVER_IDS):
        raise ValueError("a mandatory solver did not pass easy smoke")
    for solver_id in SOLVER_IDS:
        raw_path = path.parent / f"{solver_id.replace('/', '-')}-smoke.jsonl"
        if not raw_path.is_file() or sha256(raw_path) != gate["solvers"][solver_id]["raw_sha256"]:
            raise ValueError(f"smoke raw evidence missing or changed: {solver_id}")
        checked = verify_file(raw_path, solver_id, "smoke", query_path)
        if checked["record_count"] != gate["solvers"][solver_id]["record_count"]:
            raise ValueError(f"smoke row count changed: {solver_id}")
    return gate


def run(mode: str, query_path: Path, output: Path, external_worker: Path | None,
        gate_path: Path | None = None, *,
        progress_callback: Callable[[dict], None] | None = None) -> dict:
    config = load_contract()
    if mode not in ("smoke", "benchmark"):
        raise ValueError("invalid mode")
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise ValueError("C1-01 execution requires the frozen Ubuntu x86_64 Linux environment")
    os_release = Path("/etc/os-release").read_text(encoding="utf-8")
    if 'ID=ubuntu' not in os_release or 'VERSION_ID="24.04"' not in os_release:
        raise ValueError("C1-01 execution requires Ubuntu 24.04")
    if os.environ.get("ROS_DISTRO") != "jazzy":
        raise ValueError("C1-01 execution requires ROS 2 Jazzy")
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) != config["execution"]["cpu_affinity_logical_count"]:
        raise ValueError("exactly two logical CPUs must be assigned to the benchmark process")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(variable)
        if value is None or not value.isdecimal() or not 1 <= int(value) <= config["execution"]["max_solver_threads"]:
            raise ValueError(f"{variable} must be set to one or two threads")
    if external_worker is None or not external_worker.is_file() or not os.access(external_worker, os.X_OK):
        raise ValueError("built external MoveIt worker executable required")
    if mode == "benchmark":
        if gate_path is None:
            raise ValueError("benchmark requires smoke gate path")
        smoke_gate(gate_path, config, query_path)
    rows, manifest = load_queries(query_path, config)
    warmup_rows = rows[:config["execution"]["warmup_queries_per_deadline"]]
    if mode == "smoke":
        rows = smoke_selection(rows)
    solvers = {solver["id"]: solver for solver in config["solvers"]}
    result = {"task": "C1-01", "mode": mode, "started_utc": utc_now(),
              "query_list_sha256": config["queries"]["list_sha256"], "solvers": {}}
    for solver_id in SOLVER_IDS:
        item = run_method(solvers[solver_id], config, rows, manifest, output, external_worker,
                          warmup_rows, mode=mode, progress_callback=progress_callback)
        result["solvers"][solver_id] = {"pass": item.get("pass", False), "record_count": item["record_count"],
                                         "raw_sha256": item["raw_sha256"],
                                         "status": item.get("status", "PASS" if item.get("pass") else "FAIL")}
    result["finished_utc"] = utc_now()
    if mode == "smoke":
        result["status"] = ("PASS" if set(result["solvers"]) == set(SOLVER_IDS)
                            and all(item["pass"] for item in result["solvers"].values()) else "FAIL")
    else:
        result["status"] = ("MEASURED_UNVERIFIED" if set(result["solvers"]) == set(SOLVER_IDS)
                            and all(item["status"] == "MEASURED_UNVERIFIED" for item in result["solvers"].values())
                            else "INCOMPLETE")
    write_json(output / f"{mode}-gate.json" if mode == "smoke" else output / "benchmark-manifest.json", result)
    return result


def verify_file(result_path: Path, solver_id: str, mode: str, query_path: Path) -> dict:
    config = load_contract()
    rows, manifest = load_queries(query_path, config)
    if mode == "smoke":
        rows = smoke_selection(rows)
    elif mode != "benchmark":
        raise ValueError("invalid mode")
    solver = next((item for item in config["solvers"] if item["id"] == solver_id), None)
    if solver is None:
        raise ValueError("unknown solver ID")
    validator = CandidateValidator()
    deadlines = (50,) if mode == "smoke" else tuple(config["benchmark"]["deadline_profiles_ms"])
    passes = 1 if mode == "smoke" else config["benchmark"]["measurement_passes"]
    digest = hashlib.sha256()
    count = 0
    with result_path.open("rb") as stream:
        for deadline in deadlines:
            for pass_index in range(passes):
                for query in rows:
                    raw = stream.readline()
                    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
                        raise ValueError("missing/truncated result row")
                    digest.update(raw)
                    record = strict_json(raw.decode("utf-8"))
                    if encode_query(record) != raw or record["deadline_profile_ms"] != deadline or record["measurement_pass_index"] != pass_index:
                        raise ValueError("result order/canonical encoding mismatch")
                    validate_result_record(record, query, solver, config, config["queries"]["list_sha256"],
                                           manifest["dataset_manifest_sha256"], validator)
                    count += 1
        if stream.readline():
            raise ValueError("extra result rows")
    return {"status": "PASS", "solver_id": solver_id, "mode": mode,
            "record_count": count, "raw_sha256": digest.hexdigest()}


def summarize_file(result_path: Path, solver_id: str, mode: str, query_path: Path) -> dict:
    """Verify every row, then report all-attempt and independently successful distributions."""
    checked = verify_file(result_path, solver_id, mode, query_path)
    groups = defaultdict(lambda: {"queries": set(), "status": Counter(), "native_status": Counter(),
                                  "profile_a_geometry": 0, "profile_b_geometry": 0,
                                  "profile_a_deadline": 0, "profile_b_deadline": 0,
                                  "joint_limit_violations": 0, "timeouts": 0,
                                  "invalid_outputs": 0, "process_adapter_errors": 0,
                                  "position_error_m": [], "orientation_error_deg": [],
                                  "all_elapsed_ns": [], "profile_a_success_elapsed_ns": [],
                                  "profile_b_success_elapsed_ns": [], "solver_internal_elapsed_ns": [],
                                  "adapter_ipc_elapsed_ns": [], "validation_elapsed_ns": [],
                                  "iterations": []})
    with result_path.open("rb") as stream:
        for raw in stream:
            row = strict_json(raw.decode("utf-8"))
            keys = ("all", row["subset"], row["start_class"],
                    f"{row['subset']}/{row['start_class']}",
                    f"{row['deadline_profile_ms']}ms",
                    f"pass/{row['measurement_pass_index']}",
                    f"{row['deadline_profile_ms']}ms/pass/{row['measurement_pass_index']}",
                    f"{row['subset']}/{row['start_class']}/{row['deadline_profile_ms']}ms")
            for key in keys:
                group = groups[key]
                group["queries"].add(row["query_id"])
                group["status"][row["common_status"]] += 1
                group["native_status"][row["native_status"]] += 1
                group["joint_limit_violations"] += row["joint_limits"] == "FAIL"
                group["timeouts"] += row["common_status"] == "TIMEOUT"
                group["invalid_outputs"] += row["common_status"] == "INVALID_OUTPUT"
                group["process_adapter_errors"] += row["common_status"] in (
                    "PROCESS_FAILURE", "INSTALLATION_FAILURE", "ADAPTER_ERROR", "VALIDATION_ERROR")
                for profile in ("a", "b"):
                    group[f"profile_{profile}_geometry"] += row[f"profile_{profile}_geometry"]
                    group[f"profile_{profile}_deadline"] += row[f"profile_{profile}_deadline"]
                    if row[f"profile_{profile}_deadline"]:
                        group[f"profile_{profile}_success_elapsed_ns"].append(row["total_elapsed_ns"])
                for field in ("position_error_m", "orientation_error_deg", "solver_internal_elapsed_ns",
                              "iterations"):
                    if row[field] is not None:
                        group[field].append(row[field])
                for field in ("all_elapsed_ns", "adapter_ipc_elapsed_ns", "validation_elapsed_ns"):
                    group[field].append(row["total_elapsed_ns"] if field == "all_elapsed_ns" else row[field])
    summarized = {}
    series = ("position_error_m", "orientation_error_deg", "all_elapsed_ns",
              "profile_a_success_elapsed_ns", "profile_b_success_elapsed_ns",
              "solver_internal_elapsed_ns", "adapter_ipc_elapsed_ns", "validation_elapsed_ns", "iterations")
    for key, group in groups.items():
        attempts = len(group["all_elapsed_ns"])
        item = {"independent_queries": len(group["queries"]), "attempts": attempts,
                "common_status_counts": dict(group["status"]),
                "native_status_counts": dict(group["native_status"]),
                "joint_limit_violations": group["joint_limit_violations"],
                "timeout_rate": group["timeouts"] / attempts,
                "invalid_output_rate": group["invalid_outputs"] / attempts,
                "process_adapter_error_rate": group["process_adapter_errors"] / attempts}
        for profile in ("a", "b"):
            item[f"profile_{profile}_geometry_rate"] = group[f"profile_{profile}_geometry"] / attempts
            item[f"profile_{profile}_deadline_rate"] = group[f"profile_{profile}_deadline"] / attempts
        for field in series:
            item[field] = percentiles(group[field])
            item[f"{field}_missing"] = attempts - len(group[field])
        summarized[key] = item
    return {**checked, "groups": summarized}
