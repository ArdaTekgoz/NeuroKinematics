"""Frozen C1-01 solver identities and strict query/worker contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from time import monotonic_ns

import numpy as np

from neurokinematics.benchmark.contract import strict_json
from neurokinematics.kinematics.model import ROOT, load_robot, validate_q


CONFIG_PATH = ROOT / "experiments/C1-01/baseline-config.json"
HASHES_PATH = ROOT / "experiments/C1-01/frozen-hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(root: Path = ROOT) -> dict:
    """Read exactly the Stage 1 config; never replace a mismatch with a new hash."""
    manifest = strict_json((root / "experiments/C1-01/frozen-hashes.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = root / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: expected {expected}, found {actual}")
    config = strict_json((root / "experiments/C1-01/baseline-config.json").read_text(encoding="utf-8"))
    if config["stage"] != "STAGE_1_CONTRACT_ONLY":
        raise ValueError("unexpected frozen contract stage")
    ids = [solver["id"] for solver in config["solvers"]]
    if len(ids) != len(set(ids)) or set(ids) != {"dls/default", "kdl/default", "trac_ik/speed", "pick_ik/local", "pick_ik/global"}:
        raise ValueError("solver registry mismatch")
    return config


def solver_config_hash(solver: dict) -> str:
    """Bind the *frozen* solver entry, including variant, version and parameters."""
    payload = json.dumps(solver, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_numeric_vector(value, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{field} must contain {size} values")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{field} must be finite real values")
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class SolveRequest:
    query_id: str
    q_current: tuple[float, ...]
    target_position_m: tuple[float, float, float]
    target_quaternion_wxyz: tuple[float, float, float, float]
    base_frame: str
    tcp_frame: str
    joint_order: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    deadline_ns: int
    solver_id: str
    solver_config_sha256: str
    seed: int | None = None
    expires_at_monotonic_ns: int | None = None

    def with_deadline(self, started_ns: int) -> SolveRequest:
        """Bind the fixed budget to the local monotonic clock before IPC begins."""
        return replace(self, expires_at_monotonic_ns=started_ns + self.deadline_ns)

    @classmethod
    def from_query(cls, query: dict, solver: dict, config: dict, deadline_ms: int, seed: int | None = None):
        """Select an allowlist; q_target is intentionally inaccessible to workers."""
        robot = load_robot()
        if type(query.get("query_id")) is not str or not query["query_id"]:
            raise ValueError("missing query_id")
        if deadline_ms not in config["benchmark"]["deadline_profiles_ms"]:
            raise ValueError("unfrozen deadline")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("invalid seed")
        q = require_numeric_vector(query.get("q_current"), len(robot.joint_names), "q_current")
        validate_q(q, robot.joint_names, robot.limits)
        p = require_numeric_vector(query.get("target_position_m"), 3, "target_position_m")
        quat = require_numeric_vector(query.get("target_quaternion_wxyz"), 4, "target_quaternion_wxyz")
        if not np.isclose(np.linalg.norm(quat), 1.0, rtol=0, atol=1e-8):
            raise ValueError("target quaternion must be unit length")
        return cls(query["query_id"], q, p, quat, robot.base, robot.tcp,
                   tuple(robot.joint_names), tuple(tuple(pair) for pair in robot.limits),
                   deadline_ms * 1_000_000, solver["id"], solver_config_hash(solver), seed)

    def wire(self) -> dict:
        return {
            "query_id": self.query_id,
            "q_current": list(self.q_current),
            "target_position_m": list(self.target_position_m),
            "target_quaternion_wxyz": list(self.target_quaternion_wxyz),
            "base_frame": self.base_frame,
            "tcp_frame": self.tcp_frame,
            "joint_order": list(self.joint_order),
            "joint_limits": [list(pair) for pair in self.joint_limits],
            "deadline_ns": self.deadline_ns,
            "solver_id": self.solver_id,
            "solver_config_sha256": self.solver_config_sha256,
            "seed": self.seed,
            "expires_at_monotonic_ns": self.expires_at_monotonic_ns,
        }


def validate_wire_request(raw: dict, config: dict) -> SolveRequest:
    """Worker-side validation independent of the caller's Python dataclass."""
    keys = {"query_id", "q_current", "target_position_m", "target_quaternion_wxyz", "base_frame", "tcp_frame", "joint_order", "joint_limits", "deadline_ns", "solver_id", "solver_config_sha256", "seed", "expires_at_monotonic_ns"}
    if type(raw) is not dict or set(raw) != keys:
        raise ValueError("request fields mismatch")
    robot = load_robot()
    if raw["base_frame"] != robot.base or raw["tcp_frame"] != robot.tcp:
        raise ValueError("base/TCP mismatch")
    if raw["joint_order"] != list(robot.joint_names) or raw["joint_limits"] != [list(x) for x in robot.limits]:
        raise ValueError("joint order/limits mismatch")
    if type(raw["deadline_ns"]) is not int or raw["deadline_ns"] <= 0:
        raise ValueError("invalid deadline")
    expires = raw["expires_at_monotonic_ns"]
    if type(expires) is not int or not 0 < expires < 2**63:
        raise ValueError("invalid monotonic expiry")
    if expires - monotonic_ns() > raw["deadline_ns"]:
        raise ValueError("monotonic expiry exceeds frozen budget")
    solver = next((s for s in config["solvers"] if s["id"] == raw["solver_id"]), None)
    if solver is None or solver_config_hash(solver) != raw["solver_config_sha256"]:
        raise ValueError("solver/config hash mismatch")
    deadline_ms = raw["deadline_ns"] // 1_000_000
    if raw["deadline_ns"] != deadline_ms * 1_000_000:
        raise ValueError("deadline must be a frozen millisecond profile")
    query = {k: raw[k] for k in ("query_id", "q_current", "target_position_m", "target_quaternion_wxyz")}
    expected = SolveRequest.from_query(query, solver, config, deadline_ms, raw["seed"])
    expected = replace(expected, expires_at_monotonic_ns=expires)
    if expected.wire() != raw:
        raise ValueError("request canonicalization mismatch")
    return expected


def wxyz_to_xyzw(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    q = require_numeric_vector(quaternion, 4, "wxyz")
    return q[1], q[2], q[3], q[0]


def xyzw_to_wxyz(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    q = require_numeric_vector(quaternion, 4, "xyzw")
    return q[3], q[0], q[1], q[2]
