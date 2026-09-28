"""Unmeasured protocol probes for the exact workers in a C1-01 session."""

from pathlib import Path
from time import monotonic_ns

from .contract import SolveRequest, sha256, solver_config_hash
from .runner import SOLVER_IDS, command_for, write_json
from .worker import LocalWorker


def probe_expired_requests(config: dict, rows: list[dict], output: Path,
                           external_worker: Path) -> dict:
    """Require all five actual workers to reject expired work before dispatch.

    This is a protocol negative control, outside warmup and measured attempts.
    The expiry is intentionally in the past while the profile remains frozen.
    """
    output.mkdir(parents=True, exist_ok=True)
    result = {"task": "C1-01", "scope": "unmeasured expired-request protocol probe",
              "status": "PASS", "solvers": {}}
    solvers = {solver["id"]: solver for solver in config["solvers"]}
    for solver_id in SOLVER_IDS:
        solver = solvers[solver_id]
        safe_id = solver_id.replace("/", "-")
        log = output / f"{safe_id}-expired.stderr.log"
        worker = LocalWorker(command_for(solver, external_worker, config), solver_id,
                             solver_config_hash(solver), log)
        item = {"status": "FAIL"}
        try:
            worker.start()
            request = SolveRequest.from_query(rows[0], solver, config, 10)
            request = request.with_deadline(monotonic_ns() - 2 * request.deadline_ns)
            reply, call_ns = worker.call(request, late_reply_window_s=3.0)
            item.update(request=request.wire(), reply=vars(reply), call_elapsed_ns=call_ns)
            if (reply.native_status != "TIMEOUT" or reply.q_candidate is not None
                    or reply.solver_internal_elapsed_ns != 0
                    or reply.termination_reason != "DEADLINE_EXPIRED_BEFORE_SOLVE"
                    or reply.error_class is not None):
                raise ValueError("expired request was not rejected before solver dispatch")
            item["status"] = "PASS"
        except Exception as exc:
            item["error"] = {"class": type(exc).__name__, "detail": str(exc)}
            result["status"] = "FAIL"
        finally:
            worker.stop()
        item["stderr_sha256"] = sha256(log) if log.is_file() else None
        result["solvers"][solver_id] = item
    write_json(output / "expired-request-probe.json", result)
    return result
