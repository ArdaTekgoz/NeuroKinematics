"""Build a clean source snapshot, overlay exact frozen additions, replay results.

Uses the existing locked runtime; deliberately does not claim a fresh install.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/C1-06S"
CHECKOUT = ROOT / "temp/c106s-clean-source"
ARCHIVE = ROOT / "temp/c106s-clean-source.zip"

if CHECKOUT.exists() or ARCHIVE.exists():
    raise FileExistsError("preserve prior clean reproduction")
manifest = json.loads((BASE/"final-freeze.json").read_text(encoding="utf-8"))
CHECKOUT.mkdir(parents=True)
subprocess.run(["git", "archive", "--format=zip", "--output="+str(ARCHIVE), manifest["head"]], cwd=ROOT, check=True)
with zipfile.ZipFile(ARCHIVE) as archive:
    for member in archive.infolist():
        if not (CHECKOUT/member.filename).resolve().is_relative_to(CHECKOUT.resolve()):
            raise ValueError("archive path escape")
    archive.extractall(CHECKOUT)
for name, digest in manifest["files"].items():
    source, destination = ROOT/name, CHECKOUT/name
    if not destination.resolve().is_relative_to(CHECKOUT.resolve()):
        raise ValueError("manifest path escape")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("source SHA mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
shutil.copyfile(ROOT/"scripts/replay_c106s.py", CHECKOUT/"scripts/replay_c106s.py")
env = os.environ.copy()
env["PYTHONPATH"] = str(CHECKOUT/"src")
argv = [sys.executable, str(CHECKOUT/"scripts/replay_c106s.py"), "--artifacts", str(ROOT),
        "--output", str(BASE/"clean-replay.json")]
subprocess.run(argv, cwd=CHECKOUT, env=env, check=True)
