"""Persistent local newline-JSON worker protocol shared by DLS and MoveIt."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import threading
from time import monotonic_ns as perf_counter_ns

from neurokinematics.benchmark.contract import strict_json

from .contract import SolveRequest, require_numeric_vector

NATIVE_STATUSES = {"SUCCESS", "UNRESOLVED", "TIMEOUT", "MAX_ITERATIONS", "STALLED",
                   "INVALID_INPUT", "NUMERICAL_FAILURE", "JOINT_LIMIT_FAILURE",
                   "INVALID_OUTPUT", "PROCESS_FAILURE", "INSTALLATION_FAILURE",
                   "ADAPTER_ERROR", "VALIDATION_ERROR"}
ERROR_CLASSES = {"INVALID_INPUT", "INVALID_OUTPUT", "JOINT_LIMIT_FAILURE",
                 "NUMERICAL_FAILURE", "PROCESS_FAILURE", "INSTALLATION_FAILURE",
                 "ADAPTER_ERROR", "VALIDATION_ERROR", "TIMEOUT"}


class WorkerError(RuntimeError):
    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind


@dataclass(frozen=True)
class WorkerReply:
    query_id: str
    solver_id: str
    solver_config_sha256: str
    native_status: str
    termination_reason: str
    q_candidate: tuple[float, ...] | None
    iterations: int | None
    iteration_availability: str
    solver_internal_elapsed_ns: int | None
    error_class: str | None

    @classmethod
    def parse(cls, raw: str, request: SolveRequest):
        value = strict_json(raw)
        required = {"query_id", "solver_id", "solver_config_sha256", "native_status", "termination_reason", "q_candidate", "iterations", "iteration_availability", "solver_internal_elapsed_ns", "error_class"}
        if type(value) is not dict or set(value) != required:
            raise WorkerError("INVALID_OUTPUT", "worker reply fields mismatch")
        for key in ("query_id", "solver_id", "solver_config_sha256"):
            if value[key] != getattr(request, key):
                raise WorkerError("INVALID_OUTPUT", f"worker {key} mismatch")
        if type(value["native_status"]) is not str or value["native_status"] not in NATIVE_STATUSES:
            raise WorkerError("INVALID_OUTPUT", "unknown native status")
        if type(value["termination_reason"]) is not str or not value["termination_reason"]:
            raise WorkerError("INVALID_OUTPUT", "missing termination reason")
        candidate = value["q_candidate"]
        if candidate is not None:
            try:
                candidate = require_numeric_vector(candidate, len(request.joint_order), "q_candidate")
            except ValueError as exc:
                raise WorkerError("INVALID_OUTPUT", str(exc)) from exc
        if value["native_status"] == "SUCCESS" and candidate is None:
            raise WorkerError("INVALID_OUTPUT", "worker reported success without a candidate")
        iterations = value["iterations"]
        if iterations is None:
            if value["iteration_availability"] != "NOT_AVAILABLE":
                raise WorkerError("INVALID_OUTPUT", "missing iteration availability")
        elif type(iterations) is not int or iterations < 0 or value["iteration_availability"] != "AVAILABLE":
            raise WorkerError("INVALID_OUTPUT", "invalid iterations")
        inner = value["solver_internal_elapsed_ns"]
        if inner is not None and (type(inner) is not int or inner < 0):
            raise WorkerError("INVALID_OUTPUT", "invalid internal time")
        if value["error_class"] is not None and (type(value["error_class"]) is not str
                                                 or value["error_class"] not in ERROR_CLASSES):
            raise WorkerError("INVALID_OUTPUT", "invalid error class")
        return cls(value["query_id"], value["solver_id"], value["solver_config_sha256"],
                   value["native_status"], value["termination_reason"], candidate,
                   iterations, value["iteration_availability"], inner, value["error_class"])


class LocalWorker:
    """One serial process; failures are explicit and never inferred as unreachable."""

    def __init__(self, command: list[str], solver_id: str, config_sha256: str, stderr_path: Path):
        if not command:
            raise ValueError("empty worker command")
        self.command = command
        self.solver_id = solver_id
        self.config_sha256 = config_sha256
        self.stderr_path = stderr_path
        self.process: subprocess.Popen | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_file = None
        self._reader_thread = None

    def _record_protocol_failure(self, phase: str, raw: str, detail: str) -> None:
        """Preserve the rejected line before closing the process and its log."""
        evidence = {"event": "C101_PROTOCOL_ERROR", "phase": phase,
                    "solver_id": self.solver_id, "detail": detail,
                    "raw_text": raw, "raw_repr": repr(raw),
                    "utf8_hex": raw.encode("utf-8").hex()}
        assert self._stderr_file is not None
        self._stderr_file.write(json.dumps(evidence, ensure_ascii=True, allow_nan=False) + "\n")
        self._stderr_file.flush()

    def start(self, timeout_s: float = 30.0) -> int:
        if self.process is not None:
            raise WorkerError("ADAPTER_ERROR", "worker already running")
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_file = self.stderr_path.open("a", encoding="utf-8")
        started = perf_counter_ns()
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self._stderr_file, text=True, encoding="utf-8", bufsize=1)
        except OSError as exc:
            self._stderr_file.close()
            self._stderr_file = None
            raise WorkerError("INSTALLATION_FAILURE", str(exc)) from exc
        self.lines = queue.Queue()
        process = self.process
        process_lines = self.lines

        def read_lines():
            assert process.stdout is not None
            for line in process.stdout:
                process_lines.put(line)
            process_lines.put(None)

        self._reader_thread = threading.Thread(target=read_lines, daemon=True)
        self._reader_thread.start()
        try:
            ready = self.lines.get(timeout=timeout_s)
        except queue.Empty as exc:
            self.stop()
            raise WorkerError("PROCESS_FAILURE", "worker ready timeout") from exc
        if ready is None:
            code = process.poll()
            self.stop()
            raise WorkerError("PROCESS_FAILURE", f"worker exited before ready: {code}")
        try:
            value = strict_json(ready)
        except ValueError as exc:
            self._record_protocol_failure("ready", ready, str(exc))
            self.stop()
            raise WorkerError("INVALID_OUTPUT", f"invalid ready line: {exc}") from exc
        if value != {"ready": True, "solver_id": self.solver_id, "solver_config_sha256": self.config_sha256}:
            self._record_protocol_failure("ready", ready, "ready identity mismatch")
            self.stop()
            raise WorkerError("INVALID_OUTPUT", "ready identity mismatch")
        return perf_counter_ns() - started

    def call(self, request: SolveRequest, *, late_reply_window_s: float = 0.020) -> tuple[WorkerReply, int]:
        if request.solver_id != self.solver_id or request.solver_config_sha256 != self.config_sha256:
            raise WorkerError("ADAPTER_ERROR", "request worker identity mismatch")
        if self.process is None or self.process.poll() is not None:
            raise WorkerError("PROCESS_FAILURE", "worker not running")
        started = perf_counter_ns()
        if request.expires_at_monotonic_ns is None:
            request = request.with_deadline(started)
        payload = json.dumps(request.wire(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            raise WorkerError("PROCESS_FAILURE", f"worker write failed: {exc}") from exc
        remaining_s = max(0.0, (request.expires_at_monotonic_ns - perf_counter_ns()) / 1e9)
        try:
            raw = self.lines.get(timeout=remaining_s)
        except queue.Empty as exc:
            # A bounded late-reply window preserves finite candidates for
            # independent FK. The caller still marks the attempt TIMEOUT.
            try:
                raw = self.lines.get(timeout=late_reply_window_s)
            except queue.Empty:
                self.stop()
                raise WorkerError("TIMEOUT", "worker reply exceeded deadline and late-reply window") from exc
        if raw is None:
            code = self.process.poll()
            self.stop()
            raise WorkerError("PROCESS_FAILURE", f"worker exited during request: {code}")
        try:
            reply = WorkerReply.parse(raw, request)
        except (ValueError, WorkerError) as exc:
            self._record_protocol_failure("reply", raw, str(exc))
            self.stop()
            raise WorkerError("INVALID_OUTPUT", str(exc)) from exc
        return reply, perf_counter_ns() - started

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            if process.poll() is None:
                process.kill()
            try:
                # stdout belongs exclusively to read_lines; communicate()
                # would create a second reader and could split protocol lines.
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        self._reader_thread = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
