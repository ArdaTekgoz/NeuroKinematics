"""User-facing C1-01 smoke, gated benchmark and offline verification."""

import argparse
import json
from pathlib import Path

from .runner import FROZEN_QUERY_PATH, run, summarize_file, verify_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("smoke", "benchmark"):
        command = sub.add_parser(action)
        command.add_argument("--queries", type=Path, default=FROZEN_QUERY_PATH)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--external-worker", type=Path, required=True)
        if action == "benchmark":
            command.add_argument("--smoke-gate", type=Path, required=True)
    for action in ("verify-results", "summarize-results"):
        check = sub.add_parser(action)
        check.add_argument("--queries", type=Path, default=FROZEN_QUERY_PATH)
        check.add_argument("--results", type=Path, required=True)
        check.add_argument("--solver", required=True)
        check.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    args = parser.parse_args()
    if args.action == "verify-results":
        result = verify_file(args.results, args.solver, args.mode, args.queries)
    elif args.action == "summarize-results":
        result = summarize_file(args.results, args.solver, args.mode, args.queries)
    else:
        result = run(args.action, args.queries, args.output, args.external_worker,
                     getattr(args, "smoke_gate", None))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") in ("PASS", "MEASURED_UNVERIFIED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
