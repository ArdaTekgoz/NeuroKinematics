"""Hash C1-04 Stage 2 small evidence with explicit LF portability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/C1-04/stage2"
MANIFEST = BASE / "evidence-manifest.json"
SUMS = BASE / "SHA256SUMS"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def records() -> list[dict]:
    output = []
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or path in (MANIFEST, SUMS):
            continue
        data = path.read_bytes()
        canonical = data.replace(b"\r\n", b"\n")
        output.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(data),
                       "raw_sha256": sha(data), "canonical_lf_sha256": sha(canonical)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = records()
    if args.write:
        if len(current) < 20:
            raise ValueError("Stage 2 evidence incomplete")
        payload = {"task": "C1-04", "status": "STAGE_2_EVIDENCE", "files": current,
                   "policy": "canonical-LF SHA for checkout verification; raw SHA and bytes are original local snapshot"}
        MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        SUMS.write_text("".join(f"{item['canonical_lf_sha256']}  {item['path']}\n" for item in current)
                        + f"{sha(MANIFEST.read_bytes())}  {MANIFEST.relative_to(ROOT).as_posix()}\n",
                        encoding="utf-8", newline="\n")
    else:
        frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
        if [item["path"] for item in frozen] != [item["path"] for item in current]:
            raise ValueError("Stage 2 file inventory drift")
        for expected, actual in zip(frozen, current):
            if expected["canonical_lf_sha256"] != actual["canonical_lf_sha256"]:
                raise ValueError(f"Stage 2 evidence drift: {actual['path']}")
        if SUMS.read_text(encoding="utf-8").splitlines()[-1].split("  ")[0] != sha(MANIFEST.read_bytes()):
            raise ValueError("Stage 2 manifest SHA drift")
    print(json.dumps({"status": "PASS", "files": len(current), "manifest_sha256": sha(MANIFEST.read_bytes())}))


if __name__ == "__main__":
    main()
