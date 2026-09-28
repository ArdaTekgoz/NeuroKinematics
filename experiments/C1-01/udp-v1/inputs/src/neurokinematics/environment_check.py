"""T-F00 reproducible Python environment check."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

EXPECTED_PYTHON = (3, 12)
EXPECTED_PINOCCHIO = "4.1.0"


def _check(name: str, expected: Any, actual: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": bool(passed),
    }


def build_report() -> dict[str, Any]:
    """Collect the deterministic checks required by T-F00."""

    python_version = platform.python_version()
    architecture_bits = struct.calcsize("P") * 8
    checks = [
        _check(
            "python_version",
            "3.12.*",
            python_version,
            sys.version_info[:2] == EXPECTED_PYTHON,
        ),
        _check("python_architecture_bits", 64, architecture_bits, architecture_bits == 64),
    ]
    packages: dict[str, str | None] = {"numpy": None, "pinocchio": None}

    try:
        numpy = importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - exercised only in a broken environment
        checks.append(_check("numpy_import", "installed", type(exc).__name__, False))
        checks.append(_check("numpy_float64_bits", 64, None, False))
    else:
        packages["numpy"] = str(numpy.__version__)
        checks.append(_check("numpy_import", "installed", packages["numpy"], True))
        float64_bits = int(numpy.finfo(numpy.float64).bits)
        checks.append(_check("numpy_float64_bits", 64, float64_bits, float64_bits == 64))

    try:
        pinocchio = importlib.import_module("pinocchio")
    except Exception as exc:  # pragma: no cover - exercised only in a broken environment
        checks.append(
            _check("pinocchio_version", EXPECTED_PINOCCHIO, type(exc).__name__, False)
        )
        checks.append(_check("pinocchio_se3_identity", True, False, False))
    else:
        packages["pinocchio"] = str(pinocchio.__version__)
        checks.append(
            _check(
                "pinocchio_version",
                EXPECTED_PINOCCHIO,
                packages["pinocchio"],
                packages["pinocchio"] == EXPECTED_PINOCCHIO,
            )
        )
        try:
            identity = pinocchio.SE3.Identity()
            homogeneous = identity.homogeneous
            smoke_passed = bool(
                homogeneous.shape == (4, 4)
                and numpy.isfinite(homogeneous).all()
                and numpy.allclose(homogeneous, numpy.eye(4), rtol=0.0, atol=0.0)
            )
        except Exception:  # pragma: no cover - exercised only in a broken environment
            smoke_passed = False
        checks.append(_check("pinocchio_se3_identity", True, smoke_passed, smoke_passed))

    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "test_id": "T-F00",
        "status": "PASS" if passed else "FAIL",
        "environment": {
            "implementation": platform.python_implementation(),
            "python": python_version,
            "architecture_bits": architecture_bits,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the JSON report to this file; its parent must already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run T-F00, emit JSON, and return zero only when every check passes."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.output is not None and not args.output.parent.is_dir():
        parser.error(f"output parent does not exist: {args.output.parent}")

    report = build_report()
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "PASS" else 1


def cli() -> None:
    """Console-script entry point."""

    raise SystemExit(main())


if __name__ == "__main__":
    cli()

