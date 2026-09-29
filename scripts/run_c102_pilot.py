"""Run only the frozen C1-02 90-row teacher pilot after Stage 1 preflight."""

from pathlib import Path
import json
import subprocess
import sys

from neurokinematics.data.pairs import EVIDENCE, ROOT, pilot


def main():
    check = subprocess.run([sys.executable, str(ROOT / "scripts/check_c102_stage1.py"), "--check"],
                           cwd=ROOT, check=True, capture_output=True, text=True)
    output = ROOT / "data/generated/C1-02/pilot-20260928-v3"
    result = pilot(output)
    result["output_root"] = output.relative_to(ROOT).as_posix()
    result["stage1_check"] = json.loads(check.stdout)
    (EVIDENCE / "pilot-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "solver_calls": result["solver_calls"],
                      "teacher_status": result["teacher_status"], "elapsed_wall_s": result["elapsed_wall_s"]}))


if __name__ == "__main__":
    main()
