"""Resolve an apt transaction, record exact versions, then install those versions."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def capture(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args()
    pinned = {}
    for package in sorted(set(args.packages)):
        policy = capture(["apt-cache", "policy", package])
        match = re.search(r"^\s*Candidate:\s*(\S+)", policy, re.MULTILINE)
        if not match or match[1] == "(none)":
            raise RuntimeError(f"no apt candidate: {package}")
        pinned[package] = match[1]
    simulation = capture(["apt-get", "--simulate", "install", "--no-install-recommends",
                          *[f"{key}={value}" for key, value in pinned.items()]])
    for line in simulation.splitlines():
        if line.startswith("Inst "):
            match = re.match(r"Inst (\S+) (?:\[[^\]]+\] )?\((\S+)", line)
            if not match:
                raise RuntimeError(f"unrecognized apt transaction line: {line}")
            pinned[match[1]] = match[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"packages": pinned, "simulation": simulation}, indent=2) + "\n",
                           encoding="utf-8")
    subprocess.run(["apt-get", "install", "--yes", "--no-install-recommends",
                    *[f"{key}={value}" for key, value in sorted(pinned.items())]], check=True)


if __name__ == "__main__":
    main()
