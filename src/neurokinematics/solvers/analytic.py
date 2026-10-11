"""Geometric IK for the byte-locked KR 6 R900 sixx (ADR-026).

No learned weights, target joints, iterative refinement, or reachability claim.
Candidates follow the URDF's signed axes; acceptance uses IndependentFK.
This component has no service deadline/collision contract (H2 remains separate).
"""

from dataclasses import dataclass
from itertools import product
import math
from time import perf_counter_ns

import numpy as np

from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.metrics import finite_array, quaternion_rotation, rotation_error
from neurokinematics.kinematics.model import load_robot, validate_q


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def bounded_turns(angle, lower, upper):
    """Enumerate physical joint values, not a wrapped distance metric.

    Roundoff at an endpoint is snapped INTO the unchanged box. Every returned
    candidate still undergoes FK after snapping. Larger excursions are rejected.
    """
    eps = 1e-12
    first = math.ceil((lower - angle - eps) / math.tau)
    last = math.floor((upper - angle + eps) / math.tau)
    return [min(upper, max(lower, angle + k * math.tau)) for k in range(first, last + 1)]


@dataclass(frozen=True)
class AnalyticResult:
    status: str
    q_candidate: np.ndarray | None
    position_m: float | None
    orientation_rad: float | None
    generated_candidates: int
    checked_candidates: int
    elapsed_ns: int
    collision: str = "NOT_CHECKED"


class AnalyticIK:
    """Robot-specific branch enumeration, ordered by bounded joint proximity."""

    def __init__(self):
        self.inputs = load_robot()  # Exact URDF/TCP hashes are mandatory.
        self.lower, self.upper = np.asarray(self.inputs.limits).T
        self.span = self.upper - self.lower
        self.fk = IndependentFK(self.inputs)
        self.tool_rotation = ry(math.pi / 2)
        self.forearm = math.hypot(.420, .035)
        self.elbow_offset = math.atan2(.035, .420)

    def _wrist(self, rotation, current):
        # Rx(-q4) Ry(q5) Rx(-q6). atan2 avoids acos precision loss near q5=0.
        sine = math.hypot(rotation[1, 0], rotation[2, 0])
        beta = math.atan2(sine, rotation[0, 0])
        if sine > 1e-12:
            for sign in (1, -1):
                q4 = -math.atan2(sign * rotation[1, 0], -sign * rotation[2, 0])
                # Solve the last angle after removing the first two rotations.
                # Two separate atan2s of O(sin(q5)) entries lose their cancelling
                # angle correlation near the wrist singularity.
                remainder = ry(-sign * beta) @ rx(q4) @ rotation
                q6 = -math.atan2(remainder[2, 1], remainder[1, 1])
                yield from product(bounded_turns(q4, self.lower[3], self.upper[3]),
                                   bounded_turns(sign * beta, self.lower[4], self.upper[4]),
                                   bounded_turns(q6, self.lower[5], self.upper[5]))
        elif rotation[0, 0] > 0:
            # q5=0: q4+q6 is known; choose the closest point on each feasible line.
            total = -math.atan2(rotation[2, 1], rotation[1, 1])
            for value in bounded_turns(total, self.lower[3] + self.lower[5],
                                       self.upper[3] + self.upper[5]):
                low = max(self.lower[3], value - self.upper[5])
                high = min(self.upper[3], value - self.lower[5])
                if low > high:
                    continue
                w4, w6 = 1 / self.span[3]**2, 1 / self.span[5]**2
                q4 = min(high, max(low, (w4 * current[3] + w6 * (value - current[5])) / (w4 + w6)))
                q6 = min(self.upper[5], max(self.lower[5], value - q4))
                yield q4, 0., q6
        # q5=pi has no feasible value in this robot's [-120deg,120deg] interval.

    def _candidates(self, position, rotation, current):
        r6 = rotation @ self.tool_rotation.T
        center = position - .080 * r6[:, 0]
        radial = math.hypot(center[0], center[1])
        if radial > 1e-12:
            azimuth = -math.atan2(center[1], center[0])
            shoulders = [q for a in (azimuth, azimuth + math.pi)
                         for q in bounded_turns(a, self.lower[0], self.upper[0])]
        else:
            # The continuum is not enumerated completely. No-candidate is UNRESOLVED.
            shoulders = [float(current[0]), *np.linspace(self.lower[0], self.upper[0], 33)]
        candidates = []
        for q1 in shoulders:
            local = rz(q1) @ center
            x, y = local[0] - .025, .400 - local[2]
            cosine = (x*x + y*y - .455**2 - self.forearm**2) / (2 * .455 * self.forearm)
            if not math.isfinite(cosine) or abs(cosine) > 1 + 1e-12:
                continue
            elbow = math.acos(min(1., max(-1., cosine)))
            for e in (elbow, -elbow):
                theta = math.atan2(y, x) - math.atan2(self.forearm * math.sin(e), .455 + self.forearm * math.cos(e))
                for q2, q3 in product(bounded_turns(theta, self.lower[1], self.upper[1]),
                                      bounded_turns(e + self.elbow_offset, self.lower[2], self.upper[2])):
                    wrist = ry(-(q2 + q3)) @ rz(q1) @ r6
                    for q4, q5, q6 in self._wrist(wrist, current):
                        candidates.append(np.array([q1, q2, q3, q4, q5, q6]))
        candidates.sort(key=lambda q: float(np.sum(((q - current) / self.span)**2)))
        return candidates

    def solve(self, q_current, target_position_m, target_quaternion_wxyz):
        start = perf_counter_ns()

        def result(status, q=None, p=None, r=None, generated=0, checked=0):
            return AnalyticResult(status, q, p, r, generated, checked, perf_counter_ns() - start)

        try:
            current = validate_q(q_current, self.inputs.joint_names, self.inputs.limits)
            position = finite_array(target_position_m, (3,))
            rotation = quaternion_rotation(target_quaternion_wxyz)
        except (ValueError, TypeError, OverflowError):
            return result("INVALID_INPUT")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidates = self._candidates(position, rotation, current)
            checked = 0
            for q in candidates:
                checked += 1
                pose = self.fk.forward_kinematics(q)
                p = float(np.linalg.norm(pose[:3, 3] - position))
                r = rotation_error(pose[:3, :3], rotation)
                # Stronger Profile B is the internal stop; external A/B is independent.
                if p <= .001 and r <= math.radians(.5):
                    return result("KINEMATIC_VALID", q.copy(), p, r, len(candidates), checked)
            return result("UNRESOLVED", generated=len(candidates), checked=checked)
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return result("NUMERICAL_FAILURE")
