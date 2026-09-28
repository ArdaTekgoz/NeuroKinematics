"""Run the complete frozen F0-02 gate in order and keep actual exits/output.

Run with: pixi run --locked python scripts/run_f02_acceptance.py
Stops on the first failure. Does not build or change immutable robot assets.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/F0-02"
COMMANDS = [
    ["pixi", "lock", "--check"],
    ["pixi", "install", "--locked"],
    ["pixi", "run", "--locked", "test-f00", "--junitxml=experiments/F0-02/f00-junit.xml"],
    ["pixi", "run", "--locked", "verify-robot-a"],
    ["pixi", "run", "--locked", "test-f01", "--junitxml=experiments/F0-02/f01-junit.xml"],
    ["pixi", "run", "--locked", "test-f02-unit", "--junitxml=experiments/F0-02/unit-junit.xml"],
    ["pixi", "run", "--locked", "validate-fk"],
    ["pixi", "run", "--locked", "test-f02", "--junitxml=experiments/F0-02/pytest-junit.xml", "-o", "junit_family=legacy"],
]


def hashes():
    # Markdown is closure narrative, not implementation evidence. SHA256SUMS
    # deliberately does not hash itself. All other F0-02 evidence is covered.
    files = [p for p in OUTPUT.iterdir() if p.is_file() and p.suffix != ".md" and p.name != "SHA256SUMS"]
    for folder in (ROOT / "src/neurokinematics/kinematics", ROOT / "tests/f0_02"):
        files.extend(folder.glob("*.py"))
    files.extend([ROOT / "pixi.toml", ROOT / "pixi.lock", Path(__file__).resolve()])
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}" for p in sorted(files)]
    (OUTPUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    records = []
    for command in COMMANDS:
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", env=child_env)
        record = {"command": command, "exit_code": result.returncode, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(), "output": result.stdout}
        records.append(record)
        (OUTPUT / "commands.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("$ " + " ".join(command), flush=True)
        print(result.stdout, flush=True)
        if result.returncode:
            return result.returncode
    hashes()
    return 0


if __name__ == "__main__":
    sys.exit(main())
