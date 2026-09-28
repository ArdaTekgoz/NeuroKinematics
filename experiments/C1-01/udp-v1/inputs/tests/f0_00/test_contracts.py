"""T-F00 checks for the frozen scope and robot-input contracts."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONTRACT = ROOT / "config/robots/kuka_kr6_r900_sixx.source.toml"
TCP_CONTRACT = ROOT / "config/robots/tcp_tool0.json"
EXPECTED_TCP_SHA256 = "52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_robot_source_contract_is_exact_and_complete_for_f0_00() -> None:
    contract = tomllib.loads(ROBOT_CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "frozen-f0-00-input"
    assert contract["robot_id"] == "kuka_kr6_r900_sixx"
    assert contract["model"] == "KR 6 R900 sixx"
    assert contract["variant"] == "standard"
    assert contract["dof"] == 6
    assert contract["source"]["release"] == "2.0.2"
    assert (
        contract["source"]["commit"]
        == "fbda927964caa1eb4e408fb0c25fe46b5a0bde3c"
    )
    assert contract["source"]["license_spdx"] == "Apache-2.0"
    assert len(contract["source"]["files"]) == 7

    manufacturer = contract["manufacturer_reference"]
    assert manufacturer["document"] == "KUKA 0000-205-456 / V6.1 / 23.09.2022 / en"
    assert manufacturer["maximum_reach_m"] == 0.9015
    assert manufacturer["document_sha256_status"] == "NOT_AVAILABLE"

    kinematics = contract["kinematics"]
    assert kinematics["base_link"] == "base_link"
    assert kinematics["tip_link"] == "flange"
    assert kinematics["tcp_link"] == "tool0"
    assert kinematics["joint_names"] == [f"joint_{index}" for index in range(1, 7)]
    assert math.isclose(kinematics["characteristic_length_m"], 0.9015)
    assert kinematics["tcp_contract_sha256"] == _sha256(TCP_CONTRACT)

    assert contract["hash_contract"]["algorithm"] == "SHA-256"
    assert contract["hash_contract"]["resolved_urdf_sha256_status"] == "PENDING_F0-01"
    assert contract["hash_contract"]["robot_spec_sha256_status"] == "PENDING_F0-01"


def test_tcp_contract_matches_the_frozen_flange_to_tool0_transform() -> None:
    assert _sha256(TCP_CONTRACT) == EXPECTED_TCP_SHA256
    tcp = json.loads(TCP_CONTRACT.read_text(encoding="utf-8"))

    assert tcp["parent_frame"] == "flange"
    assert tcp["child_frame"] == "tool0"
    assert tcp["translation_m"] == [0.0, 0.0, 0.0]
    assert tcp["rotation"]["representation"] == "quaternion_wxyz"
    assert tcp["rotation"]["source_rpy_rad"] == [0.0, math.pi / 2.0, 0.0]

    quaternion = tcp["rotation"]["value"]
    expected = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
    assert all(math.isclose(actual, target, abs_tol=1e-15) for actual, target in zip(quaternion, expected, strict=True))
    assert math.isclose(sum(value * value for value in quaternion), 1.0, abs_tol=1e-15)


def test_spec_contains_the_required_t_f00_contract_sections() -> None:
    spec = (ROOT / "docs/SPEC.md").read_text(encoding="utf-8")

    for heading in (
        "## 1. Kapsam",
        "## 2. Robot ve model girdisi",
        "## 3. Frame, dönüşüm ve TCP",
        "## 4. Birim, sayı ve poz temsili",
        "## 5. Tolerans ve bütçe profilleri",
        "## 6. Çalışma ortamı",
        "## 8. Kapsam dışı ve iddia sınırları",
    ):
        assert heading in spec
