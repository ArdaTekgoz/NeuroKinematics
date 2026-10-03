"""Run one C1-04 command and retain exact stdout/stderr bytes and exit code."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd", type=Path, default=ROOT,
                        help="working directory for an isolated checkout")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("name must be alphanumeric/dash/underscore")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command")
    cwd = args.cwd.resolve()
    if not cwd.is_dir():
        parser.error(f"working directory does not exist: {cwd}")
    output = ROOT / "experiments/C1-04/stage2/commands" / args.name
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    start = now()
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True)
    end = now()
    (output / "stdout.log").write_bytes(result.stdout)
    (output / "stderr.log").write_bytes(result.stderr)
    payload = {"argv": command, "cwd": str(cwd), "start_utc": start, "end_utc": end,
               "exit_code": result.returncode,
               "thread_environment": {key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
               "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
               "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
               "stdout_base64": base64.b64encode(result.stdout).decode("ascii"),
               "stderr_base64": base64.b64encode(result.stderr).decode("ascii")}
    (output / "command.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
