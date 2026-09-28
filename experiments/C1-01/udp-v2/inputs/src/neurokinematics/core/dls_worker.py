"""Local DLS worker implementing the same wire protocol as MoveIt plugins."""

from __future__ import annotations

import json
import sys
from time import monotonic_ns

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.solvers.dls import DLS

from .contract import load_contract, solver_config_hash, validate_wire_request


def emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def solve_request(request, dls) -> dict:
    """Deduct IPC and request validation before entering the numeric solver."""
    if request.expires_at_monotonic_ns is None:
        raise ValueError("missing monotonic expiry")
    remaining_ns = request.expires_at_monotonic_ns - monotonic_ns()
    if remaining_ns > request.deadline_ns:
        raise ValueError("monotonic expiry exceeds frozen budget")
    reply = {"query_id": request.query_id, "solver_id": request.solver_id,
             "solver_config_sha256": request.solver_config_sha256,
             "native_status": "TIMEOUT", "termination_reason": "DEADLINE_EXPIRED_BEFORE_SOLVE",
             "q_candidate": None, "iterations": 0, "iteration_availability": "AVAILABLE",
             "solver_internal_elapsed_ns": 0, "error_class": None}
    if remaining_ns <= 0:
        return reply
    result = dls.solve(request.q_current, request.target_position_m,
                       request.target_quaternion_wxyz, deadline_ns=remaining_ns)
    reply.update(native_status=result.status.value, termination_reason=result.termination_reason,
                 q_candidate=None if result.q_candidate is None else result.q_candidate.tolist(),
                 iterations=result.iterations,
                 iteration_availability="AVAILABLE" if result.iterations is not None else "NOT_AVAILABLE",
                 solver_internal_elapsed_ns=result.elapsed_ns)
    return reply


def main() -> int:
    config = load_contract()
    solver = next(s for s in config["solvers"] if s["id"] == "dls/default")
    hash_ = solver_config_hash(solver)
    dls = DLS()
    emit({"ready": True, "solver_id": solver["id"], "solver_config_sha256": hash_})
    for line in sys.stdin:
        try:
            request = validate_wire_request(strict_json(line), config)
            if request.solver_id != solver["id"]:
                raise ValueError("wrong worker solver")
            emit(solve_request(request, dls))
        except Exception as exc:
            # A malformed request cannot be bound to a trusted query id. Exit;
            # the caller records an adapter/process failure and the stderr log.
            print(f"DLS worker rejected request: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
