"""Freeze and audit C1-04 Stage 1 inputs without running neural training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/C1-04"
MANIFEST = EVIDENCE / "input-hashes.json"
FROZEN_SUMS = EVIDENCE / "SHA256SUMS"
UPSTREAM = [
    "AGENTS.md",
    "docs/raporlar/02_Core_v1_0_r1.md",
    "docs/TEST_PROTOCOL.md",
    "docs/adr/ADR-011-c103-torch-fk.md",
    "docs/adr/ADR-012-c103-runtime-overlay.md",
    "experiments/F0-06/G0_DECISION.md",
    "experiments/F0-06/handoff-inputs.json",
    "experiments/C1-02/config.json",
    "experiments/C1-02/schema.json",
    "experiments/C1-02/dataset-manifest.json",
    "experiments/C1-02/normalization.json",
    "experiments/C1-02/acceptance.json",
    "experiments/C1-02/input-hashes.json",
    "experiments/C1-03/stage2/acceptance.json",
    "experiments/C1-03/requirements-win-cpu.lock",
    "experiments/C1-03/stage2/runtime-supplement.lock",
    "assets/robots/robot_a/robot.urdf",
    "assets/robots/robot_a/robot_spec.json",
    "assets/robots/robot_a/manifest.json",
    "config/robots/tcp_tool0.json",
    "src/neurokinematics/data/pairs.py",
    "src/neurokinematics/data/pair_validation.py",
    "src/neurokinematics/kinematics/torch_fk.py",
    "src/neurokinematics/kinematics/pinocchio_fk.py",
    "src/neurokinematics/kinematics/custom_fk.py",
    "pixi.lock",
]
DATA_ROOT = Path("data/generated/C1-02/v1")
LOCAL_SMALL = ["dataset-manifest.json", "normalization.json"]
ROBOT_HASHES = {
    "assets/robots/robot_a/robot.urdf": "83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96",
    "assets/robots/robot_a/robot_spec.json": "4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d",
    "assets/robots/robot_a/manifest.json": "aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1",
    "config/robots/tcp_tool0.json": "52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18",
}


def digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
            size += len(block)
    return sha.hexdigest(), size


def record(relative: str, storage: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"missing input: {relative}")
    value, size = digest(path)
    return {"path": relative.replace("\\", "/"), "sha256": value, "bytes": size, "storage": storage}


def expected_paths() -> list[tuple[str, str]]:
    c102 = json.loads((ROOT / "experiments/C1-02/dataset-manifest.json").read_text(encoding="utf-8"))
    if c102["record_count"] != 24000 or len(c102["shards"]) != 34:
        raise ValueError("C1-02 manifest row/shard count drift")
    if c102["dataset_content_sha256"] != "2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2":
        raise ValueError("C1-02 canonical data identity drift")
    return [(path, "GIT_TRACKED") for path in UPSTREAM] + [
        ((DATA_ROOT / name).as_posix(), "LOCAL_ONLY") for name in LOCAL_SMALL
    ] + [
        ((DATA_ROOT / shard["path"]).as_posix(), "LOCAL_ONLY") for shard in c102["shards"]
    ]


def audit_rows(rows: list[dict]) -> dict:
    c102 = json.loads((ROOT / "experiments/C1-02/dataset-manifest.json").read_text(encoding="utf-8"))
    by_path = {item["path"]: item for item in rows}
    if len(by_path) != len(rows):
        raise ValueError("duplicate input path")
    for path, expected in ROBOT_HASHES.items():
        if by_path[path]["sha256"] != expected:
            raise ValueError(f"frozen robot identity drift: {path}")
    if by_path["experiments/C1-02/config.json"]["sha256"] != c102["config_sha256"]:
        raise ValueError("accepted C1-02 config drift")
    if by_path["experiments/C1-02/schema.json"]["sha256"] != c102["schema_sha256"]:
        raise ValueError("accepted C1-02 schema drift")
    for shard in c102["shards"]:
        relative = (DATA_ROOT / shard["path"]).as_posix()
        if by_path[relative]["sha256"] != shard["file_sha256"]:
            raise ValueError(f"C1-02 shard manifest SHA mismatch: {relative}")
    local_manifest = (DATA_ROOT / "dataset-manifest.json").as_posix()
    tracked_manifest = "experiments/C1-02/dataset-manifest.json"
    local_normalization = (DATA_ROOT / "normalization.json").as_posix()
    tracked_normalization = "experiments/C1-02/normalization.json"
    if by_path[local_manifest]["sha256"] != by_path[tracked_manifest]["sha256"]:
        raise ValueError("local C1-02 manifest differs from accepted copy")
    if by_path[local_normalization]["sha256"] != by_path[tracked_normalization]["sha256"]:
        raise ValueError("local C1-02 normalization differs from accepted copy")
    accepted = json.loads((ROOT / "experiments/C1-02/acceptance.json").read_text(encoding="utf-8"))
    if accepted["status"] != "PASS" or accepted["records"] != 24000 or accepted["shards"] != 34 or accepted["dataset_content_sha256"] != c102["dataset_content_sha256"]:
        raise ValueError("C1-02 acceptance drift")
    c103 = json.loads((ROOT / "experiments/C1-03/stage2/acceptance.json").read_text(encoding="utf-8"))
    if c103["status"] != "PASS / ACCEPTED" or any(c103["tests"][key] != "PASS" for key in ("T-C01", "T-C02")):
        raise ValueError("C1-03 acceptance drift")
    if by_path["src/neurokinematics/kinematics/torch_fk.py"]["sha256"] != c103["source_hashes"]["src/neurokinematics/kinematics/torch_fk.py"]:
        raise ValueError("accepted Torch FK source drift")
    return {
        "status": "PASS",
        "input_files": len(rows),
        "local_shards": 34,
        "local_shard_bytes": sum(by_path[(DATA_ROOT / s["path"]).as_posix()]["bytes"] for s in c102["shards"]),
        "c102_records": 24000,
        "c102_dataset_content_sha256": c102["dataset_content_sha256"],
        "c102_acceptance": "PASS",
        "c103_acceptance": "PASS / ACCEPTED",
        "training": "NOT_RUN",
    }


def audit_frozen_outputs() -> int:
    if not FROZEN_SUMS.is_file():
        raise ValueError("Stage 1 SHA256SUMS missing")
    count = 0
    for line in FROZEN_SUMS.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64 or len(relative) == 0:
            raise ValueError("invalid Stage 1 SHA256SUMS row")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"missing/outside frozen output: {relative}")
        actual, _ = digest(path)
        if actual != expected:
            raise ValueError(f"Stage 1 frozen output drift: {relative}: {actual} != {expected}")
        count += 1
    if count < 8:
        raise ValueError("Stage 1 SHA256SUMS incomplete")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-manifest", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = expected_paths()
    rows = [record(path, storage) for path, storage in paths]
    result = audit_rows(rows)
    payload = {"schema_version": "1.0.0", "task": "C1-04", "stage": "STAGE_1_INPUTS", "files": rows}
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    else:
        frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if frozen != payload:
            raise ValueError("Stage 1 input hash manifest drift; do not rewrite to bypass")
        result["frozen_outputs"] = audit_frozen_outputs()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
