"""Independent candidate verdict. Solver stop flags never establish geometry."""

from dataclasses import dataclass

import numpy as np

from neurokinematics.kinematics.metrics import finite_array, quaternion_rotation, position_error, rotation_error
from neurokinematics.kinematics.model import load_robot, validate_q
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.transforms import check_transform
from neurokinematics.solvers.dls import SolverStatus, profile_success


@dataclass(frozen=True)
class CandidateVerdict:
    status: SolverStatus
    position_error_m: float | None
    orientation_error_rad: float | None
    orientation_error_deg: float | None
    profile_a_geometry: bool
    profile_b_geometry: bool
    joint_limits: str
    collision: str = "NOT_CHECKED"


def deadline_success(geometry, solver_status, total_elapsed_ns, deadline_ns):
    status = SolverStatus(solver_status)
    if (type(total_elapsed_ns) is not int or total_elapsed_ns < 0
            or type(deadline_ns) is not int or deadline_ns <= 0):
        raise ValueError("invalid elapsed time or deadline")
    return bool(geometry and total_elapsed_ns <= deadline_ns
                and status not in {SolverStatus.TIMEOUT, SolverStatus.INVALID_INPUT,
                                   SolverStatus.NUMERICAL_FAILURE, SolverStatus.PROVEN_UNREACHABLE})


class CandidateValidator:
    def __init__(self, inputs=None):
        self.inputs = load_robot() if inputs is None else inputs
        self.reference = PinocchioFK(self.inputs)

    def validate(self, result, target_position_m, target_quaternion_wxyz):
        status = SolverStatus(result.status)
        if status == SolverStatus.PROVEN_UNREACHABLE:
            raise ValueError("a solver flag is not an independent reachability proof")
        if status == SolverStatus.INVALID_INPUT:
            return CandidateVerdict(status, None, None, None, False, False, "NOT_AVAILABLE")
        try:
            position = finite_array(target_position_m, (3,))
            rotation = quaternion_rotation(target_quaternion_wxyz)
        except (ValueError, TypeError):
            return CandidateVerdict(SolverStatus.INVALID_INPUT, None, None, None, False, False, "NOT_AVAILABLE")
        if result.q_candidate is None:
            # Preserve the technical cause, but a fabricated SUCCESS cannot pass.
            final = SolverStatus.UNRESOLVED if status == SolverStatus.SUCCESS else status
            return CandidateVerdict(final, None, None, None, False, False, "NOT_AVAILABLE")
        try:
            q = validate_q(result.q_candidate, self.inputs.joint_names, self.inputs.limits)
        except (ValueError, TypeError):
            return CandidateVerdict(SolverStatus.NUMERICAL_FAILURE, None, None, None, False, False, "FAIL")
        try:
            actual = self.reference.reference_forward_kinematics(q)
            check_transform(actual)
            p = position_error(actual[:3, 3], position)
            r = rotation_error(actual[:3, :3], rotation)
        except (ValueError, TypeError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            return CandidateVerdict(SolverStatus.NUMERICAL_FAILURE, None, None, None, False, False, "PASS")
        a, b = profile_success(p, r, "A"), profile_success(p, r, "B")
        final = SolverStatus.UNRESOLVED if status == SolverStatus.SUCCESS and not b else status
        return CandidateVerdict(final, p, r, float(np.rad2deg(r)), a, b, "PASS")
