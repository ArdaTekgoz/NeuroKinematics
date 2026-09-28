"""T-F01 acceptance and explicit-error tests for robot A."""

from __future__ import annotations

import copy
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from neurokinematics.robot_asset import (
    ASSET,
    EXPECTED_COMMIT,
    EXPECTED_JOINTS,
    EXPECTED_RELEASE,
    MANIFEST_REL,
    ROBOT_SPEC_REL,
    ROOT,
    SNAPSHOT_REL,
    SOURCE_CONTRACT_REL,
    TCP_CONTRACT_REL,
    URDF_REL,
    RobotAssetError,
    build_payloads,
    canonical_json_bytes,
    extract_active_joints,
    load_source_contract,
    resolve_xacro,
    sha256_bytes,
    sha256_file,
    validate_source_identity,
    verify,
    verify_file_records,
    verify_frozen_sources,
)


def _manifest() -> dict:
    return json.loads((ROOT / MANIFEST_REL).read_text(encoding="utf-8"))


def _urdf_root() -> ET.Element:
    return ET.fromstring((ROOT / URDF_REL).read_bytes())


def _minimal_source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / SOURCE_CONTRACT_REL).parent.mkdir(parents=True)
    shutil.copy2(ROOT / SOURCE_CONTRACT_REL, root / SOURCE_CONTRACT_REL)
    contract = load_source_contract()
    for entry in contract["source"]["files"]:
        relative = SNAPSHOT_REL / entry["path"]
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, root / relative)
    return root


def test_exact_upstream_release_commit_variant_and_license() -> None:
    manifest = _manifest()
    assert manifest["source"]["release"] == EXPECTED_RELEASE
    assert manifest["source"]["commit"] == EXPECTED_COMMIT
    assert manifest["source"]["license_spdx"] == "Apache-2.0"
    assert manifest["robot"]["variant"] == "standard"
    license_path = ROOT / manifest["source"]["license_path"]
    assert license_path.is_file()
    assert "Apache License" in license_path.read_text(encoding="utf-8")


def test_all_frozen_critical_source_hashes_match() -> None:
    assert len(verify_frozen_sources()) == 7


def test_xacro_resolution_is_byte_deterministic_and_persisted() -> None:
    first = resolve_xacro()
    second = resolve_xacro()
    assert first == second
    assert first == (ROOT / URDF_REL).read_bytes()


def test_resolved_urdf_and_tcp_hashes_match_manifest() -> None:
    manifest = _manifest()
    assert sha256_file(ROOT / URDF_REL) == manifest["resolved_urdf_sha256"]
    assert sha256_file(ROOT / TCP_CONTRACT_REL) == manifest["tcp_contract"]["sha256"]


def test_pinocchio_410_parses_six_dof_fixed_model() -> None:
    import pinocchio

    model = pinocchio.buildModelFromXML((ROOT / URDF_REL).read_text(encoding="utf-8"))
    assert pinocchio.__version__ == "4.1.0"
    assert model.nq == 6
    assert model.nv == 6


def test_fixed_base_open_serial_chain_and_joint_order() -> None:
    robot = _urdf_root()
    links = {link.get("name") for link in robot.findall("link")}
    child_links = {
        joint.find("child").get("link")
        for joint in robot.findall("joint")
        if joint.find("child") is not None
    }
    assert links - child_links == {"base_link"}
    active = extract_active_joints((ROOT / URDF_REL).read_bytes())
    assert [joint["name"] for joint in active] == EXPECTED_JOINTS
    assert all(joint["type"] == "revolute" for joint in active)


def test_frames_axes_and_finite_ordered_limits_match_contract() -> None:
    manifest = _manifest()
    contract = load_source_contract()
    assert manifest["frames"] == {"base": "base_link", "tip": "flange", "tcp": "tool0"}
    joints = manifest["mechanism"]["active_joints"]
    assert [joint["axis"] for joint in joints] == contract["kinematics"]["joint_axes"]
    for joint in joints:
        lower = joint["limit"]["lower_rad"]
        upper = joint["limit"]["upper_rad"]
        assert math.isfinite(lower) and math.isfinite(upper) and lower < upper
        assert (ROOT / joint["source_path"]).is_file()
        assert (ROOT / joint["limit_cross_check_path"]).is_file()


def test_every_distributed_file_exists_and_matches_recorded_hash() -> None:
    manifest = _manifest()
    assert verify_file_records(ROOT, manifest["files"]) == 29


def test_all_mesh_uris_resolve_to_manifested_snapshot_files() -> None:
    records = {entry["path"] for entry in _manifest()["files"]}
    meshes = _urdf_root().findall(".//mesh")
    assert len(meshes) == 14
    for mesh in meshes:
        uri = mesh.get("filename")
        assert uri is not None and uri.startswith("package://kuka_agilus_support/")
        relative = SNAPSHOT_REL / "kuka_agilus_support" / uri.split("/", 3)[3]
        assert (ROOT / relative).is_file()
        assert str(relative).replace("\\", "/") in records


def test_canonical_robot_spec_hash_is_reproducible() -> None:
    manifest = _manifest()
    spec_bytes = (ROOT / ROBOT_SPEC_REL).read_bytes()
    spec = json.loads(spec_bytes)
    assert canonical_json_bytes(spec) == spec_bytes
    assert sha256_bytes(spec_bytes) == manifest["robot_spec"]["sha256"]
    assert "self-hash" in manifest["robot_spec"]["canonicalization"]["scope"]


def test_full_verifier_passes_and_rebuild_payloads_match() -> None:
    report = verify()
    assert report["status"] == "PASS"
    assert report["active_joints_verified"] == 6
    urdf, spec, manifest = build_payloads()
    assert urdf == (ROOT / URDF_REL).read_bytes()
    assert spec == (ROOT / ROBOT_SPEC_REL).read_bytes()
    assert manifest == (ROOT / MANIFEST_REL).read_bytes()


def test_missing_limit_is_an_explicit_error() -> None:
    robot = _urdf_root()
    joint = next(item for item in robot.findall("joint") if item.get("name") == "joint_3")
    joint.remove(joint.find("limit"))
    with pytest.raises(RobotAssetError, match="missing lower/upper limit for joint_3"):
        extract_active_joints(ET.tostring(robot, encoding="utf-8"))


def test_missing_source_is_an_explicit_error(tmp_path: Path) -> None:
    root = _minimal_source_tree(tmp_path)
    missing = root / SNAPSHOT_REL / "LICENSE"
    missing.unlink()
    with pytest.raises(RobotAssetError, match="missing required file"):
        verify_frozen_sources(root)


def test_modified_source_hash_is_an_explicit_error(tmp_path: Path) -> None:
    root = _minimal_source_tree(tmp_path)
    changed = root / SNAPSHOT_REL / "LICENSE"
    changed.write_bytes(changed.read_bytes() + b"changed")
    with pytest.raises(RobotAssetError, match="source hash mismatch"):
        verify_frozen_sources(root)


def test_manifest_hash_mismatch_is_an_explicit_error(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"actual")
    records = [{"path": "payload.bin", "sha256": sha256_bytes(b"expected")}]
    with pytest.raises(RobotAssetError, match="manifest hash mismatch"):
        verify_file_records(tmp_path, records)


def test_unexpected_variant_is_an_explicit_error() -> None:
    contract = copy.deepcopy(load_source_contract())
    contract["variant"] = "KR 6 R900-2"
    with pytest.raises(RobotAssetError, match="unexpected source identity.*variant"):
        validate_source_identity(contract)
