"""Actual FK round trips and negative controls for the geometric recovery."""

import inspect
import itertools
import math

import numpy as np
import pytest

from neurokinematics.data.factory import canonical_quaternion
from neurokinematics.kinematics.metrics import rotation_error
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.solvers.analytic import AnalyticIK, bounded_turns


@pytest.fixture(scope="module")
def solvers():
    solver = AnalyticIK()
    return solver, PinocchioFK(solver.inputs)


def roundtrip(solvers, target_q, current):
    solver, reference = solvers
    target = reference.reference_forward_kinematics(target_q)
    result = solver.solve(current, target[:3, 3], canonical_quaternion(target[:3, :3]))
    assert result.status == "KINEMATIC_VALID"
    assert result.collision == "NOT_CHECKED"
    assert np.all(result.q_candidate >= solver.lower) and np.all(result.q_candidate <= solver.upper)
    actual = reference.reference_forward_kinematics(result.q_candidate)
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= 1e-9
    assert np.linalg.norm(actual[:3, :3] - target[:3, :3]) <= 1e-9
    assert rotation_error(actual[:3, :3], target[:3, :3]) <= math.radians(.5)
    return result


def test_random_reference_roundtrip(solvers):
    solver, _ = solvers
    rng = np.random.default_rng(2026101101)
    for _ in range(512):
        roundtrip(solvers, rng.uniform(solver.lower, solver.upper), rng.uniform(solver.lower, solver.upper))


@pytest.mark.parametrize("joint,side", list(itertools.product(range(6), (0, 1))))
def test_exact_joint_endpoints(solvers, joint, side):
    solver, _ = solvers
    q = (solver.lower + solver.upper) / 2
    q[joint] = (solver.lower, solver.upper)[side][joint]
    roundtrip(solvers, q, (solver.lower + solver.upper) / 2)


@pytest.mark.parametrize("wrist", [0., 1e-13, -1e-13, 1e-8, -1e-8])
def test_wrist_singularity_and_nearby(solvers, wrist):
    solver, _ = solvers
    rng = np.random.default_rng(2026101102)
    for _ in range(32):
        q = rng.uniform(solver.lower, solver.upper)
        q[4] = wrist
        roundtrip(solvers, q, rng.uniform(solver.lower, solver.upper))


def test_base_axis_singularity(solvers):
    solver, _ = solvers
    # q3=0: choose q2 so the wrist radius is zero; target FK still independent.
    length = math.hypot(.455 + .420, .035)
    q = np.array([.7, math.atan2(.035, .875) - math.acos(-.025 / length), 0., .3, .2, -.8])
    roundtrip(solvers, q, q.copy())


def test_preserves_valid_current_and_turns(solvers):
    solver, _ = solvers
    q = np.array([.3, -1.2, .5, 3.2, -.4, 5.9])
    result = roundtrip(solvers, q, q)
    assert np.max(np.abs(result.q_candidate - q)) < 1e-10
    assert len(bounded_turns(-.4, -6.11, 6.11)) == 2


def test_all_joint_box_corners(solvers):
    solver, _ = solvers
    for sides in itertools.product((0, 1), repeat=6):
        q = np.where(sides, solver.upper, solver.lower)
        roundtrip(solvers, q, (solver.lower+solver.upper)/2)


@pytest.mark.parametrize("offset", [0., 1e-8, -1e-8])
def test_elbow_extension_singularity(solvers, offset):
    q = np.array([.3, -.7, math.atan2(.035, .420)+offset, .7, -.4, .9])
    roundtrip(solvers, q, q.copy())


@pytest.mark.parametrize("field,value", [
    ("current", [0.]*5), ("current", [float("nan")]*6),
    ("current", [10.]*6), ("position", [0., 0.]),
    ("position", [float("inf"), 0., 0.]), ("position", [True]*3),
    ("quaternion", [0.]*4), ("quaternion", [2., 0., 0., 0.]),
    ("quaternion", [float("nan")]*4), ("quaternion", [1., 0., 0.])])
def test_invalid_input(solvers, field, value):
    solver, _ = solvers
    request = dict(current=(solver.lower + solver.upper)/2, position=[.3, 0., .5], quaternion=[1., 0., 0., 0.])
    request[field] = value
    result = solver.solve(request["current"], request["position"], request["quaternion"])
    assert result.status == "INVALID_INPUT" and result.q_candidate is None


def test_far_target_not_reachability_claim(solvers):
    solver, _ = solvers
    result = solver.solve((solver.lower+solver.upper)/2, [100., 0., 0.], [1., 0., 0., 0.])
    assert result.status == "UNRESOLVED" and result.q_candidate is None


def test_target_joints_absent_from_api():
    assert list(inspect.signature(AnalyticIK.solve).parameters) == [
        "self", "q_current", "target_position_m", "target_quaternion_wxyz"]


def test_wrong_tcp_is_caught_by_independent_acceptance(solvers, monkeypatch):
    solver, reference = solvers
    q = np.array([.3, -1.2, .5, .7, -.4, .9])
    target = reference.reference_forward_kinematics(q)
    monkeypatch.setattr(solver, "tool_rotation", np.eye(3))
    result = solver.solve(q, target[:3, 3], canonical_quaternion(target[:3, :3]))
    assert result.status != "KINEMATIC_VALID"


def test_quaternion_sign_equivalence(solvers):
    solver, reference = solvers
    q = np.array([.3, -1.2, .5, .7, -.4, .9])
    target = reference.reference_forward_kinematics(q)
    quat = canonical_quaternion(target[:3, :3])
    a = solver.solve(q, target[:3, 3], quat)
    b = solver.solve(q, target[:3, 3], -quat)
    assert np.array_equal(a.q_candidate, b.q_candidate)
