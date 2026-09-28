"""Build and verify the immutable F0-01 robot asset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
ASSET_REL = Path("assets/robots/robot_a")
ASSET = ROOT / ASSET_REL
SNAPSHOT_REL = ASSET_REL / "source_snapshot"
SNAPSHOT = ROOT / SNAPSHOT_REL
SOURCE_CONTRACT_REL = Path("config/robots/kuka_kr6_r900_sixx.source.toml")
TCP_CONTRACT_REL = Path("config/robots/tcp_tool0.json")
WRAPPER_REL = ASSET_REL / "robot.urdf.xacro"
URDF_REL = ASSET_REL / "robot.urdf"
ROBOT_SPEC_REL = ASSET_REL / "robot_spec.json"
MANIFEST_REL = ASSET_REL / "manifest.json"
UPSTREAM_MACRO_REL = Path(
    "kuka_agilus_support/urdf/kr6_r900_sixx_macro.xacro"
)
COMMON_CONSTANTS_REL = Path("kuka_resources/urdf/common_constants.xacro")
COMMON_MATERIALS_REL = Path("kuka_resources/urdf/common_materials.xacro")
COMMON_COLOURS_REL = Path("kuka_resources/urdf/common_colours.xacro")
EXPECTED_COMMIT = "fbda927964caa1eb4e408fb0c25fe46b5a0bde3c"
EXPECTED_RELEASE = "2.0.2"
EXPECTED_VARIANT = "standard"
EXPECTED_JOINTS = [f"joint_{index}" for index in range(1, 7)]


class RobotAssetError(RuntimeError):
    """Raised when an F0-01 invariant is violated."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RobotAssetError(f"missing required file: {path}")
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize canonical robot-spec JSON as UTF-8, sorted compact JSON plus LF."""

    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_source_contract(root: Path = ROOT) -> dict[str, Any]:
    path = root / SOURCE_CONTRACT_REL
    if not path.is_file():
        raise RobotAssetError(f"missing source contract: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def validate_source_identity(contract: dict[str, Any]) -> None:
    source = contract.get("source", {})
    checks = {
        "release": (source.get("release"), EXPECTED_RELEASE),
        "commit": (source.get("commit"), EXPECTED_COMMIT),
        "variant": (contract.get("variant"), EXPECTED_VARIANT),
        "package": (source.get("package"), "kuka_agilus_support"),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise RobotAssetError("unexpected source identity: " + "; ".join(mismatches))


def verify_frozen_sources(root: Path = ROOT) -> list[dict[str, str]]:
    contract = load_source_contract(root)
    validate_source_identity(contract)
    snapshot = root / SNAPSHOT_REL
    results: list[dict[str, str]] = []
    for entry in contract["source"]["files"]:
        source_path = str(entry["path"])
        path = snapshot / source_path
        actual = sha256_file(path)
        expected = str(entry["sha256"])
        if actual != expected:
            raise RobotAssetError(
                f"source hash mismatch: {source_path}: {actual} != {expected}"
            )
        results.append({"path": source_path, "sha256": actual})
    return results


def _adapt_xacro_inputs(temp_dir: Path, root: Path = ROOT) -> Path:
    """Replace ROS package lookup only in temporary Xacro copies.

    The immutable snapshot stays byte-identical to upstream. Absolute include paths
    make Xacro 2.1.1 usable on the locked native-Windows environment without ROS.
    """

    snapshot = root / SNAPSHOT_REL
    adapted = temp_dir / "adapted"
    adapted.mkdir()
    constants = adapted / "common_constants.xacro"
    materials = adapted / "common_materials.xacro"
    colours = adapted / "common_colours.xacro"
    macro = adapted / "kr6_r900_sixx_macro.xacro"
    wrapper = adapted / "robot.urdf.xacro"

    constants.write_bytes((snapshot / COMMON_CONSTANTS_REL).read_bytes())
    colours.write_bytes((snapshot / COMMON_COLOURS_REL).read_bytes())
    materials_text = (snapshot / COMMON_MATERIALS_REL).read_text(encoding="utf-8")
    materials_text = materials_text.replace(
        "$(find kuka_resources)/urdf/common_colours.xacro", colours.as_posix()
    )
    materials.write_text(materials_text, encoding="utf-8", newline="\n")

    macro_text = (snapshot / UPSTREAM_MACRO_REL).read_text(encoding="utf-8")
    macro_text = macro_text.replace(
        "$(find kuka_resources)/urdf/common_constants.xacro", constants.as_posix()
    ).replace(
        "$(find kuka_resources)/urdf/common_materials.xacro", materials.as_posix()
    )
    macro.write_text(macro_text, encoding="utf-8", newline="\n")

    wrapper_text = (root / WRAPPER_REL).read_text(encoding="utf-8")
    wrapper_text = wrapper_text.replace(
        "source_snapshot/kuka_agilus_support/urdf/kr6_r900_sixx_macro.xacro",
        macro.as_posix(),
    )
    wrapper.write_text(wrapper_text, encoding="utf-8", newline="\n")
    return wrapper


def resolve_xacro(root: Path = ROOT) -> bytes:
    try:
        import xacro
    except ImportError as exc:  # pragma: no cover - broken environment only
        raise RobotAssetError("locked xacro dependency is unavailable") from exc

    with tempfile.TemporaryDirectory(prefix="neuro-f001-xacro-") as directory:
        wrapper = _adapt_xacro_inputs(Path(directory), root)
        try:
            document = xacro.process_file(str(wrapper), in_order=True)
        except Exception as exc:
            raise RobotAssetError(f"xacro resolution failed: {exc}") from exc
        # process_file adds a banner containing the random temporary path before
        # the document element. Serialize only the robot element and supply a
        # fixed declaration so identical inputs produce identical bytes.
        xml = b'<?xml version="1.0" encoding="utf-8"?>\n' + document.documentElement.toxml(
            encoding="utf-8"
        )
    return xml.replace(b"\r\n", b"\n") + (b"" if xml.endswith(b"\n") else b"\n")


def _vector(text: str | None, *, field: str, joint: str) -> list[float]:
    if text is None:
        raise RobotAssetError(f"missing {field} for {joint}")
    values = [float(value) for value in text.split()]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RobotAssetError(f"invalid {field} for {joint}: {text!r}")
    return values


def extract_active_joints(urdf_bytes: bytes) -> list[dict[str, Any]]:
    try:
        robot = ET.fromstring(urdf_bytes)
    except ET.ParseError as exc:
        raise RobotAssetError(f"invalid resolved URDF XML: {exc}") from exc
    by_name = {joint.get("name"): joint for joint in robot.findall("joint")}
    active: list[dict[str, Any]] = []
    for name in EXPECTED_JOINTS:
        joint = by_name.get(name)
        if joint is None:
            raise RobotAssetError(f"missing active joint: {name}")
        if joint.get("type") != "revolute":
            raise RobotAssetError(f"unsupported joint type for {name}: {joint.get('type')}")
        parent = joint.find("parent")
        child = joint.find("child")
        limit = joint.find("limit")
        if parent is None or not parent.get("link"):
            raise RobotAssetError(f"missing parent for {name}")
        if child is None or not child.get("link"):
            raise RobotAssetError(f"missing child for {name}")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise RobotAssetError(f"missing lower/upper limit for {name}")
        lower = float(limit.get("lower", "nan"))
        upper = float(limit.get("upper", "nan"))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise RobotAssetError(f"invalid finite ordered limits for {name}")
        active.append(
            {
                "name": name,
                "type": "revolute",
                "parent": parent.get("link"),
                "child": child.get("link"),
                "axis": _vector(
                    None if joint.find("axis") is None else joint.find("axis").get("xyz"),
                    field="axis",
                    joint=name,
                ),
                "limit": {"lower_rad": lower, "upper_rad": upper},
                "source_path": str(SNAPSHOT_REL / UPSTREAM_MACRO_REL).replace("\\", "/"),
                "limit_cross_check_path": str(
                    SNAPSHOT_REL
                    / "kuka_agilus_support/config/kr6_r900_sixx_joint_limits.yaml"
                ).replace("\\", "/"),
            }
        )
    extras = [
        joint.get("name")
        for joint in robot.findall("joint")
        if joint.get("type") in {"revolute", "continuous", "prismatic"}
        and joint.get("name") not in EXPECTED_JOINTS
    ]
    if extras:
        raise RobotAssetError(f"unexpected active joints: {extras}")
    return active


def _validate_joint_contract(
    joints: list[dict[str, Any]], contract: dict[str, Any]
) -> None:
    kinematics = contract["kinematics"]
    if [joint["name"] for joint in joints] != kinematics["joint_names"]:
        raise RobotAssetError("active joint order differs from frozen contract")
    if [joint["axis"] for joint in joints] != kinematics["joint_axes"]:
        raise RobotAssetError("joint axes differ from frozen source contract")
    expected_links = [
        ("base_link", "link_1"),
        ("link_1", "link_2"),
        ("link_2", "link_3"),
        ("link_3", "link_4"),
        ("link_4", "link_5"),
        ("link_5", "link_6"),
    ]
    actual_links = [(joint["parent"], joint["child"]) for joint in joints]
    if actual_links != expected_links:
        raise RobotAssetError(f"robot is not the expected open serial chain: {actual_links}")


def _robot_spec(
    contract: dict[str, Any], joints: list[dict[str, Any]], urdf_sha256: str, tcp_sha256: str
) -> dict[str, Any]:
    source = contract["source"]
    kinematics = contract["kinematics"]
    return {
        "schema_version": "1.0",
        "robot": {
            "id": contract["robot_id"],
            "manufacturer": contract["manufacturer"],
            "model": contract["model"],
            "variant": contract["variant"],
        },
        "source": {
            "repository": source["repository"],
            "release": source["release"],
            "commit": source["commit"],
            "package": source["package"],
            "license_spdx": source["license_spdx"],
            "license_path": str(SNAPSHOT_REL / "LICENSE").replace("\\", "/"),
        },
        "mechanism": {
            "type": contract["mechanism"],
            "dof": contract["dof"],
            "active_joint_order": EXPECTED_JOINTS,
            "active_joints": joints,
        },
        "frames": {
            "base": kinematics["base_link"],
            "tip": kinematics["tip_link"],
            "tcp": kinematics["tcp_link"],
        },
        "characteristic_length": {
            "value": kinematics["characteristic_length_m"],
            "unit": "m",
        },
        "resolved_urdf": {
            "path": str(URDF_REL).replace("\\", "/"),
            "sha256": urdf_sha256,
        },
        "tcp_contract": {
            "path": str(TCP_CONTRACT_REL).replace("\\", "/"),
            "sha256": tcp_sha256,
        },
    }


def _category(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".stl", ".dae"}:
        return "mesh"
    if suffix == ".xacro":
        return "xacro"
    if suffix == ".urdf":
        return "resolved-urdf"
    if path.name == "LICENSE":
        return "license"
    if path == ROBOT_SPEC_REL:
        return "canonical-robot-spec"
    return "source-or-contract"


def _file_record(root: Path, relative: Path, payload: bytes | None = None) -> dict[str, Any]:
    data = (root / relative).read_bytes() if payload is None else payload
    return {
        "path": str(relative).replace("\\", "/"),
        "category": _category(relative),
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
    }


def build_payloads(root: Path = ROOT) -> tuple[bytes, bytes, bytes]:
    contract = load_source_contract(root)
    validate_source_identity(contract)
    verify_frozen_sources(root)
    tcp_path = root / TCP_CONTRACT_REL
    tcp_sha = sha256_file(tcp_path)
    expected_tcp_sha = contract["kinematics"]["tcp_contract_sha256"]
    if tcp_sha != expected_tcp_sha:
        raise RobotAssetError(f"TCP contract hash mismatch: {tcp_sha} != {expected_tcp_sha}")

    urdf_bytes = resolve_xacro(root)
    joints = extract_active_joints(urdf_bytes)
    _validate_joint_contract(joints, contract)
    urdf_sha = sha256_bytes(urdf_bytes)
    spec = _robot_spec(contract, joints, urdf_sha, tcp_sha)
    spec_bytes = canonical_json_bytes(spec)
    spec_sha = sha256_bytes(spec_bytes)

    source_files = sorted(
        path.relative_to(root)
        for path in (root / SNAPSHOT_REL).rglob("*")
        if path.is_file()
    )
    files = [_file_record(root, path) for path in source_files]
    files.extend(
        [
            _file_record(root, SOURCE_CONTRACT_REL),
            _file_record(root, TCP_CONTRACT_REL),
            _file_record(root, WRAPPER_REL),
            _file_record(root, URDF_REL, urdf_bytes),
            _file_record(root, ROBOT_SPEC_REL, spec_bytes),
        ]
    )
    files.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": "1.0",
        "robot": spec["robot"],
        "source": spec["source"],
        "frames": spec["frames"],
        "mechanism": spec["mechanism"],
        "characteristic_length": spec["characteristic_length"],
        "resolved_urdf_sha256": urdf_sha,
        "tcp_contract": spec["tcp_contract"],
        "robot_spec": {
            "path": str(ROBOT_SPEC_REL).replace("\\", "/"),
            "sha256": spec_sha,
            "canonicalization": {
                "scope": "entire robot_spec.json payload; no self-hash field",
                "json_keys": "lexicographically sorted",
                "json_separators": [",", ":"],
                "encoding": "UTF-8",
                "line_endings": "LF",
                "final_newline": True,
                "hash_algorithm": "SHA-256 over raw bytes",
            },
        },
        "files": files,
        "generation": {
            "tool": "xacro",
            "version": importlib.metadata.version("xacro"),
            "adapter": str(WRAPPER_REL).replace("\\", "/"),
            "command": "pixi run --locked build-robot-a",
            "notes": "Temporary absolute include adaptation replaces ROS $(find) only; the source snapshot remains byte-identical.",
        },
        "validation": {
            "status": "PASS",
            "test_id": "T-F01",
            "limitations": [
                "Model-internal kinematics only; physical accuracy is not established.",
                "Collision, dynamics, payload and robot safety are not validated.",
                "Manufacturer PDF bytes were not retrievable in F0-00; its document hash remains unavailable.",
                "Linux execution was not run; only the linux-64 dependency graph is locked.",
            ],
        },
    }
    return urdf_bytes, spec_bytes, pretty_json_bytes(manifest)


def build(root: Path = ROOT) -> dict[str, str]:
    urdf_bytes, spec_bytes, manifest_bytes = build_payloads(root)
    (root / URDF_REL).write_bytes(urdf_bytes)
    (root / ROBOT_SPEC_REL).write_bytes(spec_bytes)
    (root / MANIFEST_REL).write_bytes(manifest_bytes)
    return {
        "resolved_urdf_sha256": sha256_bytes(urdf_bytes),
        "robot_spec_sha256": sha256_bytes(spec_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "tcp_sha256": sha256_file(root / TCP_CONTRACT_REL),
    }


def _verify_mesh_references(root: Path, urdf_bytes: bytes, manifest: dict[str, Any]) -> int:
    robot = ET.fromstring(urdf_bytes)
    records = {entry["path"] for entry in manifest["files"]}
    count = 0
    for mesh in robot.findall(".//mesh"):
        uri = mesh.get("filename", "")
        prefix = "package://kuka_agilus_support/"
        if not uri.startswith(prefix):
            raise RobotAssetError(f"unsupported mesh URI: {uri}")
        relative = SNAPSHOT_REL / "kuka_agilus_support" / uri[len(prefix) :]
        if not (root / relative).is_file():
            raise RobotAssetError(f"unresolved mesh URI: {uri}")
        normalized = str(relative).replace("\\", "/")
        if normalized not in records:
            raise RobotAssetError(f"mesh omitted from manifest: {normalized}")
        count += 1
    return count


def verify_file_records(root: Path, records: Any) -> int:
    if not isinstance(records, list) or not records:
        raise RobotAssetError("manifest has no file records")
    for entry in records:
        relative = Path(entry["path"])
        actual = sha256_file(root / relative)
        if actual != entry.get("sha256"):
            raise RobotAssetError(f"manifest hash mismatch: {relative}")
    return len(records)


def verify(root: Path = ROOT) -> dict[str, Any]:
    manifest_path = root / MANIFEST_REL
    if not manifest_path.is_file():
        raise RobotAssetError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source_identity(
        {
            "variant": manifest.get("robot", {}).get("variant"),
            "source": manifest.get("source", {}),
        }
    )
    expected_urdf, expected_spec, expected_manifest = build_payloads(root)
    comparisons = {
        URDF_REL: expected_urdf,
        ROBOT_SPEC_REL: expected_spec,
        MANIFEST_REL: expected_manifest,
    }
    for relative, expected in comparisons.items():
        path = root / relative
        if not path.is_file():
            raise RobotAssetError(f"missing generated file: {relative}")
        if path.read_bytes() != expected:
            raise RobotAssetError(f"generated file is not reproducible: {relative}")

    records = manifest.get("files")
    record_count = verify_file_records(root, records)

    joints = extract_active_joints(expected_urdf)
    contract = load_source_contract(root)
    _validate_joint_contract(joints, contract)
    mesh_count = _verify_mesh_references(root, expected_urdf, manifest)

    try:
        import pinocchio

        # The Windows urdfdom path bridge cannot represent the Turkish character
        # in the user's checkout path. Parsing the same persisted UTF-8 bytes via
        # Pinocchio's XML API avoids that filesystem-encoding defect.
        model = pinocchio.buildModelFromXML(expected_urdf.decode("utf-8"))
    except Exception as exc:
        raise RobotAssetError(f"Pinocchio URDF parse failed: {exc}") from exc
    if model.nq != 6 or model.nv != 6:
        raise RobotAssetError(f"Pinocchio model DOF mismatch: nq={model.nq}, nv={model.nv}")

    return {
        "schema_version": 1,
        "test_id": "T-F01",
        "status": "PASS",
        "source_commit": EXPECTED_COMMIT,
        "source_release": EXPECTED_RELEASE,
        "critical_source_files_verified": len(verify_frozen_sources(root)),
        "distributed_files_verified": record_count,
        "mesh_references_verified": mesh_count,
        "active_joints_verified": len(joints),
        "pinocchio_version": str(pinocchio.__version__),
        "pinocchio_nq": model.nq,
        "pinocchio_nv": model.nv,
        "resolved_urdf_sha256": sha256_bytes(expected_urdf),
        "robot_spec_sha256": sha256_bytes(expected_spec),
        "manifest_sha256": sha256_file(manifest_path),
        "tcp_sha256": sha256_file(root / TCP_CONTRACT_REL),
    }


def source_report(root: Path = ROOT) -> dict[str, Any]:
    contract = load_source_contract(root)
    validate_source_identity(contract)
    files = verify_frozen_sources(root)
    return {
        "schema_version": 1,
        "test_id": "T-F01-source-snapshot",
        "status": "PASS",
        "repository": contract["source"]["repository"],
        "release": contract["source"]["release"],
        "commit": contract["source"]["commit"],
        "package": contract["source"]["package"],
        "license_spdx": contract["source"]["license_spdx"],
        "snapshot_root": str(SNAPSHOT_REL).replace("\\", "/"),
        "critical_files": files,
        "critical_file_count": len(files),
        "note": "Snapshot exported from the exact Git tag/commit with core.autocrlf=false; hashes are SHA-256 over raw bytes.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Regenerate the resolved URDF and manifest.")
    verify_parser = subparsers.add_parser("verify", help="Verify all F0-01 invariants.")
    verify_parser.add_argument("--output", type=Path)
    source_parser = subparsers.add_parser(
        "verify-sources", help="Verify and report the frozen upstream snapshot."
    )
    source_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            report = build()
        elif args.command == "verify":
            report = verify()
        else:
            report = source_report()
    except RobotAssetError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1
    payload = pretty_json_bytes(report).decode("utf-8")
    if args.command != "build" and args.output is not None:
        if not args.output.parent.is_dir():
            raise SystemExit(f"output parent does not exist: {args.output.parent}")
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
