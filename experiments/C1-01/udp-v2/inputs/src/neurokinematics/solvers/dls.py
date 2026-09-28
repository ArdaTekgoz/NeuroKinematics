"""Fixed-damping, projected DLS. No target joint vector enters this API."""

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from neurokinematics.kinematics.finite_difference import log_so3
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.metrics import finite_array, quaternion_rotation, rotation_error
from neurokinematics.kinematics.model import ROOT, load_robot, validate_q
from neurokinematics.kinematics.transforms import check_transform

CONFIG_PATH = ROOT / "experiments/F0-05/solver-config.json"


class SolverStatus(StrEnum):
    SUCCESS = "SUCCESS"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    TIMEOUT = "TIMEOUT"
    STALLED = "STALLED"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    UNRESOLVED = "UNRESOLVED"
    PROVEN_UNREACHABLE = "PROVEN_UNREACHABLE"


@dataclass(frozen=True)
class SolverConfig:
    solver_name: str = "DLS"
    solver_version: str = "0.1.0"
    dtype: str = "float64"
    characteristic_length_m: float = 0.9015
    damping: float = 0.05
    max_iterations: int = 200
    max_joint_step_rad: float = 0.20
    stop_position_m: float = 0.001
    stop_orientation_rad: float = float(np.deg2rad(0.5))
    step_policy: str = "componentwise_clip_then_joint_limit_projection"
    stalled_step_inf_norm_rad: float = 1e-12
    stalled_consecutive_steps: int = 5
    adaptive_damping: bool = False
    step_acceptance: bool = False
    restarts: int = 0


def load_solver_config(path=CONFIG_PATH):
    from dataclasses import asdict
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = asdict(SolverConfig())
    if value != expected or any(type(value[k]) is not type(v) for k, v in expected.items()):
        raise ValueError("frozen solver config mismatch")
    return SolverConfig(**value)


@dataclass(frozen=True)
class SolverResult:
    status: SolverStatus
    termination_reason: str
    q_candidate: np.ndarray | None
    iterations: int | None
    first_profile_a_iteration: int | None
    first_profile_b_iteration: int | None
    elapsed_ns: int


def pose_error(current, target):
    """Base-frame left rotation increment; [metres; radians]."""
    check_transform(current)
    check_transform(target)
    return np.r_[target[:3, 3] - current[:3, 3],
                 log_so3(target[:3, :3] @ current[:3, :3].T)]


def dls_step(jacobian, error, config: SolverConfig):
    raw = np.asarray(jacobian)
    if raw.ndim != 2 or raw.shape[0] != 6 or raw.shape[1] < 1:
        raise ValueError("Jacobian must be 6 by joint_count")
    j = finite_array(raw, raw.shape).copy()
    e = finite_array(error, (6,)).copy()
    j[:3] /= config.characteristic_length_m
    e[:3] /= config.characteristic_length_m
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta = j.T @ np.linalg.solve(j @ j.T + config.damping**2 * np.eye(6), e)
    return finite_array(delta, (j.shape[1],))


def projected_update(q, delta, limits, config: SolverConfig):
    """Clip each delta component, then project into the inclusive joint box."""
    q = finite_array(q, (len(limits),))
    delta = finite_array(delta, q.shape)
    bounds = finite_array(limits, (len(q), 2))
    if np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        raise ValueError("invalid limits or initial joints")
    capped = np.clip(delta, -config.max_joint_step_rad, config.max_joint_step_rad)
    return np.clip(q + capped, bounds[:, 0], bounds[:, 1])


def profile_success(position_m, orientation_rad, profile):
    """Metric conjunction only; caller must separately check candidate limits."""
    p, r = {"A": (0.002, float(np.deg2rad(1))),
            "B": (0.001, float(np.deg2rad(0.5)))}[profile]
    return bool(np.isfinite(position_m) and np.isfinite(orientation_rad)
                and 0 <= position_m <= p and 0 <= orientation_rad <= r)


class DLS:
    """Single-thread instance. Model setup is outside per-query timing.

    RobotInputs can describe small analytic chains in unit tests. Production
    construction without arguments verifies the immutable six-axis robot.
    SUCCESS is a provisional stop flag, never an independent geometry verdict.
    """

    def __init__(self, inputs=None, *, clock=perf_counter_ns):
        self.inputs = load_robot() if inputs is None else inputs
        self.config = load_solver_config()
        self.geometric = IndependentJacobian(self.inputs)
        self.clock = clock

    def solve(self, q_current, target_position_m, target_quaternion_wxyz, *, deadline_ns=None):
        started = self.clock()
        q, iterations, first_a, first_b = None, None, None, None

        def finish(status, reason):
            elapsed = self.clock() - started
            # Cover time spent in final bookkeeping, including a nominal success.
            if (deadline_ns is not None and elapsed >= deadline_ns
                    and status not in (SolverStatus.INVALID_INPUT, SolverStatus.NUMERICAL_FAILURE)):
                status, reason = SolverStatus.TIMEOUT, "DEADLINE_REACHED"
            return SolverResult(status, reason, None if q is None else q.copy(),
                                iterations, first_a, first_b, elapsed)

        try:
            if deadline_ns is not None and (type(deadline_ns) is not int or deadline_ns <= 0):
                # Avoid comparing invalid deadline values inside finish().
                deadline_ns = None
                raise ValueError("deadline_ns must be a positive integer or None")
            position = finite_array(target_position_m, (3,))
            rotation = quaternion_rotation(target_quaternion_wxyz)
            initial = validate_q(q_current, self.inputs.joint_names, self.inputs.limits)
            target = np.eye(4, dtype=np.float64)
            target[:3, 3], target[:3, :3] = position, rotation
            q = initial.copy()
        except (ValueError, TypeError, OverflowError) as exc:
            return finish(SolverStatus.INVALID_INPUT, f"INVALID_INPUT: {exc}")

        iterations, tiny_steps = 0, 0
        while True:
            if deadline_ns is not None and self.clock() - started >= deadline_ns:
                return finish(SolverStatus.TIMEOUT, "DEADLINE_REACHED")
            try:
                current = self.geometric.fk.forward_kinematics(q)
                error = pose_error(current, target)
                p = float(np.linalg.norm(error[:3]))
                r = rotation_error(current[:3, :3], target[:3, :3])
                if profile_success(p, r, "A") and first_a is None:
                    first_a = iterations
                if profile_success(p, r, "B") and first_b is None:
                    first_b = iterations
                if deadline_ns is not None and self.clock() - started >= deadline_ns:
                    return finish(SolverStatus.TIMEOUT, "DEADLINE_REACHED")
                if p <= self.config.stop_position_m and r <= self.config.stop_orientation_rad:
                    return finish(SolverStatus.SUCCESS, "STOP_TOLERANCES_REACHED")
                if iterations >= self.config.max_iterations:
                    return finish(SolverStatus.MAX_ITERATIONS, "ITERATION_CAP_REACHED")
                if tiny_steps >= self.config.stalled_consecutive_steps:
                    return finish(SolverStatus.STALLED, "PROJECTED_STEP_STALLED")
                delta = dls_step(self.geometric.jacobian(q), error, self.config)
                updated = projected_update(q, delta, self.inputs.limits, self.config)
                small = np.linalg.norm(updated - q, ord=np.inf) <= self.config.stalled_step_inf_norm_rad
                tiny_steps = tiny_steps + 1 if small else 0
                q = updated
                iterations += 1
                if deadline_ns is not None and self.clock() - started >= deadline_ns:
                    return finish(SolverStatus.TIMEOUT, "DEADLINE_REACHED")
            except (ValueError, TypeError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                return finish(SolverStatus.NUMERICAL_FAILURE, f"NUMERICAL_FAILURE: {exc}")
