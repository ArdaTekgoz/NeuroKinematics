"""Generate frozen C1-02 pairs after the explicit Stage 2 pilot gate."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from neurokinematics.data.pairs import EVIDENCE, ROOT, generate, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    subprocess.run([sys.executable, str(ROOT / "scripts/check_c102_stage1.py"), "--check"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    pilot = json.loads((EVIDENCE / "pilot-summary.json").read_text(encoding="utf-8"))
    if (pilot["status"] != "PASS" or pilot["actual_rows"] != 90 or
            pilot["solver_calls"] != 360 or pilot["elapsed_wall_s"] > 1800 or
            pilot.get("peak_rss_bytes", 2**63) > 4 * 1024**3 or
            any(pilot.get("thread_environment", {}).get(name) != "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")) or
            pilot["config_sha256"] != sha(EVIDENCE / "config.json") or
            pilot["schema_sha256"] != sha(EVIDENCE / "schema.json")):
        raise ValueError("pilot gate failed or frozen contract drift")
    result = generate(output)
    print(json.dumps({"status": "GENERATED_UNVERIFIED", "rows": result["manifest"]["record_count"],
                      "dataset_content_sha256": result["manifest"]["dataset_content_sha256"],
                      "missing_labels": result["audit"]["missing_labels"]}, sort_keys=True))


if __name__ == "__main__":
    main()
