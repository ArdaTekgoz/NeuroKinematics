"""Capture the exact built Ubuntu/Jazzy dependency closure before C1-01 smoke.

This audit writes no new dependency versions. It only records what the real
container installed, and rejects source commits that differ from Stage 1.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from neurokinematics.core.contract import ROOT, load_contract, sha256


def output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, encoding="utf-8", stderr=subprocess.STDOUT).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_contract()
    if platform.system() != "Linux" or platform.machine().lower() != "x86_64":
        raise ValueError("lock audit requires Linux x86_64")
    release = Path("/etc/os-release").read_text(encoding="utf-8")
    if 'ID=ubuntu' not in release or 'VERSION_ID="24.04"' not in release:
        raise ValueError("lock audit requires Ubuntu 24.04")
    if os.environ.get("ROS_DISTRO") != "jazzy":
        raise ValueError("lock audit requires ROS 2 Jazzy")
    if len(os.sched_getaffinity(0)) != 2:
        raise ValueError("lock audit requires two logical CPUs")
    sources = {}
    mapping = {"moveit2": "kdl/default", "trac_ik": "trac_ik/speed", "pick_ik": "pick_ik/global"}
    solvers = {item["id"]: item for item in config["solvers"]}
    for directory, solver_id in mapping.items():
        path = args.external_root / directory
        commit = output(["git", "-C", str(path), "rev-parse", "HEAD"])
        if commit != solvers[solver_id]["source_commit"]:
            raise ValueError(f"source commit mismatch: {directory}: {commit}")
        if output(["git", "-C", str(path), "status", "--porcelain"]):
            raise ValueError(f"source tree is modified: {directory}")
        sources[directory] = {"commit": commit, "url": solvers[solver_id]["upstream_url"]}
    packages = output(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\n"])
    package_rows = sorted(line.split("\t") for line in packages.splitlines())
    if any(len(row) != 3 for row in package_rows):
        raise ValueError("invalid dpkg closure")
    pixi_version = output(["pixi", "--version"])
    if pixi_version != "pixi 0.81.0":
        raise ValueError(f"unexpected Pixi version: {pixi_version}")
    result = {
        "schema_version": "1.0.0", "task": "C1-01", "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "image_id": args.image_id, "os_release": release, "architecture": platform.machine(),
        "ros_distro": os.environ["ROS_DISTRO"], "compiler": output(["c++", "--version"]).splitlines()[0],
        "libc": output(["ldd", "--version"]).splitlines()[0], "pixi": pixi_version,
        "pixi_lock_sha256": sha256(ROOT / "pixi.lock"),
        "sources": sources, "dpkg_package_count": len(package_rows),
        "dpkg_closure_sha256": hashlib.sha256(packages.encode("utf-8")).hexdigest(),
        "dpkg_packages": package_rows,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_governors": {path.parent.name: path.read_text(encoding="utf-8").strip()
                          for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")},
        "meminfo": {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
                    if line.startswith(("MemTotal:", "MemAvailable:"))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("image_id", "ros_distro", "dpkg_package_count",
                                                   "dpkg_closure_sha256", "pixi_lock_sha256", "cpu_affinity")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
