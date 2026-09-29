"""Copy small generated C1-02 evidence and write a non-self-referential SHA index."""

import hashlib
import json
from pathlib import Path
import shutil

from neurokinematics.data.pairs import EVIDENCE, ROOT


def main():
    generated = ROOT / "data/generated/C1-02/v1"
    acceptance = json.loads((EVIDENCE / "acceptance.json").read_text(encoding="utf-8"))
    determinism = json.loads((EVIDENCE / "determinism-summary.json").read_text(encoding="utf-8"))
    if acceptance["status"] != "PASS" or determinism["status"] != "PASS":
        raise ValueError("cannot collect evidence before acceptance and determinism pass")
    for name in ("dataset-manifest.json", "leakage-audit.json", "normalization.json", "teacher-summary.json"):
        shutil.copyfile(generated / name, EVIDENCE / name)
    paths = sorted(path for path in EVIDENCE.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths]
    (EVIDENCE / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "files": len(paths)}))


if __name__ == "__main__":
    main()
