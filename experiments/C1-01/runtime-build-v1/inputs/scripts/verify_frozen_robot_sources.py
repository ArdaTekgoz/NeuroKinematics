"""Verify the remote bytes pinned by the F0-00 robot source contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import urllib.request
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/robots/kuka_kr6_r900_sixx.source.toml"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NeuroKinematics-F0-00"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _result(label: str, url: str, expected: str) -> dict[str, Any]:
    try:
        payload = _fetch(url)
    except Exception as exc:  # pragma: no cover - depends on external network state
        return {
            "label": label,
            "url": url,
            "expected_sha256": expected,
            "actual_sha256": None,
            "bytes": None,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    actual = hashlib.sha256(payload).hexdigest()
    return {
        "label": label,
        "url": url,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "bytes": len(payload),
        "passed": actual == expected,
        "error": None,
    }


def build_report() -> dict[str, Any]:
    contract = tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source = contract["source"]
    repository = source["repository"].removesuffix("/")
    owner_repo = repository.removeprefix("https://github.com/")
    commit = source["commit"]
    raw_root = f"https://raw.githubusercontent.com/{owner_repo}/{commit}"

    results = [
        _result(entry["path"], f"{raw_root}/{entry['path']}", entry["sha256"])
        for entry in source["files"]
    ]
    passed = all(result["passed"] for result in results)
    return {
        "schema_version": 1,
        "check_id": "F0-00-ROBOT-SOURCE",
        "status": "PASS" if passed else "FAIL",
        "source_release": source["release"],
        "source_commit": commit,
        "manufacturer_reference": contract["manufacturer_reference"],
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report()
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if not args.output.parent.is_dir():
            raise SystemExit(f"output parent does not exist: {args.output.parent}")
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
