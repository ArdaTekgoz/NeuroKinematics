"""Verify the complete C1-02 dataset; never generate or relabel rows."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from neurokinematics.data.pair_validation import verify
from neurokinematics.data.pairs import EVIDENCE, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts/check_c102_stage1.py"), "--check"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    result = verify(input_path)
    output_path = Path(args.output) if args.output else EVIDENCE / "acceptance.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "records": result["records"],
                      "shards": result["shards"], "dataset_content_sha256": result["dataset_content_sha256"]}))


if __name__ == "__main__":
    main()
