"""Read the F0-01 byte-locked inputs; no kinematics backend is used here."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FROZEN_HASHES = {
    "assets/robots/robot_a/robot.urdf": "83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96",
    "assets/robots/robot_a/robot_spec.json": "4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d",
    "assets/robots/robot_a/manifest.json": "aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1",
    "config/robots/tcp_tool0.json": "52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18",
}
SOURCE_COMMIT = "fbda927964caa1eb4e408fb0c25fe46b5a0bde3c"


@dataclass(frozen=True)
class RobotInputs:
    urdf: bytes
    joint_names: tuple[str, ...]
    limits: tuple[tuple[float, float], ...]
    base: str
    tip: str
    tcp: str
    robot_id: str
    hashes: dict[str, str]


def load_robot(root: Path = ROOT) -> RobotInputs:
    """Verify raw bytes before parsing; parse the very same verified byte buffers."""
    payloads = {}
    for relative, expected in FROZEN_HASHES.items():
        payload = (root / relative).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"immutable hash mismatch: {relative}: {actual} != {expected}")
        payloads[relative] = payload
    manifest = json.loads(payloads["assets/robots/robot_a/manifest.json"])
    spec = json.loads(payloads["assets/robots/robot_a/robot_spec.json"])
    if manifest["source"]["commit"] != SOURCE_COMMIT:
        raise ValueError("source commit mismatch")
    for key in ("mechanism", "frames", "source", "robot"):
        if manifest[key] != spec[key]:
            raise ValueError(f"manifest/RobotSpec mismatch: {key}")
    names = tuple(manifest["mechanism"]["active_joint_order"])
    joints = {j["name"]: j for j in spec["mechanism"]["active_joints"]}
    limits = tuple((joints[n]["limit"]["lower_rad"], joints[n]["limit"]["upper_rad"])
                   for n in names)
    return RobotInputs(payloads["assets/robots/robot_a/robot.urdf"], names, limits,
                       spec["frames"]["base"], spec["frames"]["tip"],
                       spec["frames"]["tcp"], spec["robot"]["id"], dict(FROZEN_HASHES))


def validate_q(q, joint_names, limits) -> np.ndarray:
    """Numeric real radians, manifest order, inclusive limits; never clip or convert degrees."""
    raw = np.asarray(q)
    if raw.dtype.kind not in "fiu":
        raise ValueError("q must contain real numeric radians")
    values = np.asarray(raw, dtype=np.float64)
    bounds = np.asarray(limits, dtype=np.float64)
    if values.shape != (len(joint_names),):
        raise ValueError(f"q shape must be {(len(joint_names),)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("q must be finite")
    if bounds.shape != (len(joint_names), 2) or not np.isfinite(bounds).all():
        raise ValueError("invalid joint limits")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("joint limits must be ordered")
    if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
        raise ValueError("q outside joint limits (radians)")
    return values
