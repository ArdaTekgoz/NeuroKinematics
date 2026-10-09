"""Record an immutable C1-06R subprocess attempt, including failed commands."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("safe attempt name required")
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not argv:
        parser.error("command required")
    output = ROOT / "experiments/C1-06R/commands" / args.name
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    selected = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    selected.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUTF8="1")
    env.update(selected)
    start = datetime.now(timezone.utc).isoformat()
    with (output / "stdout.log").open("wb") as out, (output / "stderr.log").open("wb") as err:
        try:
            code = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err).returncode
        except OSError as exc:
            code = 127
            err.write(str(exc).encode("utf-8"))
    record = dict(argv=argv, cwd=str(ROOT), start_utc=start,
                  end_utc=datetime.now(timezone.utc).isoformat(), exit_code=code,
                  environment=selected)
    for name in ("stdout", "stderr"):
        record[name + "_sha256"] = hashlib.sha256((output / (name + ".log")).read_bytes()).hexdigest()
    (output / "command.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=True))
    if code:
        print((output / "stderr.log").read_text(encoding="utf-8", errors="replace")[-6000:])
    raise SystemExit(code)


if __name__ == "__main__":
    main()
