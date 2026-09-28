"""Reject dependency drift while rebuilding only the local C1-01 worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, encoding="utf-8").strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(base: dict, *, recorded: dict | None = None) -> dict:
    packages = command("dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\n")
    actual = {
        "dpkg_closure_sha256": hashlib.sha256(packages.encode("utf-8")).hexdigest(),
        "dpkg_package_count": len(packages.splitlines()),
        "pixi_lock_sha256": digest(Path("/work/pixi.lock")),
        "pixi": command("pixi", "--version"),
    }
    for field, value in actual.items():
        if value != base[field]:
            raise ValueError(f"inherited dependency changed: {field}: {value!r}")
        if recorded is not None and recorded[field] != value:
            raise ValueError(f"recorded dependency differs from built image: {field}")
    for name, source in base["sources"].items():
        directory = str(Path("/opt/c101/external") / name)
        if command("git", "-C", directory, "rev-parse", "HEAD") != source["commit"]:
            raise ValueError(f"inherited external source commit changed: {name}")
        if command("git", "-C", directory, "status", "--porcelain"):
            raise ValueError(f"inherited external source is modified: {name}")
    if recorded is not None and recorded["sources"] != base["sources"]:
        raise ValueError("recorded external sources differ from base lock")
    return {"status": "PASS", "scope": "inherited dependency closure", **actual}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--recorded-lock", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args()
    base = json.loads(args.base_lock.read_text(encoding="utf-8-sig"))
    recorded = (json.loads(args.recorded_lock.read_text(encoding="utf-8-sig"))
                if args.recorded_lock else None)
    result = audit(base, recorded=recorded)
    if args.input_manifest:
        manifest = json.loads(args.input_manifest.read_text(encoding="utf-8-sig"))
        runtime_files = {"Dockerfile", ".dockerignore", "base-environment-lock.json"}
        for relative, expected in manifest["files"].items():
            root = Path("/opt/c101/runtime") if relative in runtime_files else Path("/work")
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
                raise ValueError(f"image input differs from build snapshot: {relative}")
        result["input_manifest_sha256"] = digest(args.input_manifest)
        result["worker_binary_sha256"] = digest(Path(
            "/opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker"))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
