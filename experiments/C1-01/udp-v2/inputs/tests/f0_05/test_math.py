from dataclasses import replace
import inspect

import numpy as np
from numpy.testing import assert_allclose
import pytest

from neurokinematics.data.factory import canonical_quaternion
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.transforms import axis_angle
from neurokinematics.solvers import dls
from neurokinematics.benchmark.validation import CandidateValidator, deadline_success


def test_zero_pose_and_iteration_zero(robot, real_pose):
    q, p, quat = real_pose
    saved = q.copy()
    result = dls.DLS(robot).solve(q, p, quat)
    assert result.status == dls.SolverStatus.SUCCESS
    assert result.iterations == result.first_profile_a_iteration == result.first_profile_b_iteration == 0
    assert np.array_equal(q, saved)
    verdict = CandidateValidator(robot).validate(result, p, quat)
    assert verdict.profile_a_geometry and verdict.profile_b_geometry
    assert verdict.collision == 'NOT_CHECKED'


@pytest.mark.parametrize('count,qgoal', [(1, [.08]), (2, [.08, -.06])])
def test_analytic_chain_solution(analytic_robot, count, qgoal):
    robot = analytic_robot(count)
    pose = PinocchioFK(robot).reference_forward_kinematics(qgoal)
    # Hand-derived planar translation, unaffected by world mount/TCP rotation.
    a = qgoal[0]
    p = [np.cos(a), np.sin(a), 0]
    if count == 2:
        p = [np.cos(a)+np.cos(a+qgoal[1]), np.sin(a)+np.sin(a+qgoal[1]), 0]
    assert_allclose(pose[:3, 3], p, atol=2e-15, rtol=0)
    quaternion = canonical_quaternion(pose[:3, :3])
    solver = dls.DLS(robot)
    result = solver.solve(np.zeros(count), p, quaternion)
    verdict = CandidateValidator(robot).validate(result, p, quaternion)
    assert result.status == dls.SolverStatus.SUCCESS
    assert verdict.profile_a_geometry and verdict.profile_b_geometry
    assert 0 < result.first_profile_a_iteration <= result.first_profile_b_iteration <= result.iterations


def test_base_log_sign_and_linear_angular_order():
    current = np.eye(4)
    current[:3, :3] = axis_angle([1, 0, 0], .7)
    current[:3, 3] = [1, 2, 3]
    target = current.copy()
    target[:3, :3] = axis_angle([0, 0, 1], 2e-3) @ current[:3, :3]
    target[:3, 3] += [.01, -.02, .03]
    assert_allclose(dls.pose_error(current, target), [.01, -.02, .03, 0, 0, .002], atol=3e-16)
    assert_allclose(dls.pose_error(target, current), [-.01, .02, -.03, 0, 0, -.002], atol=3e-16)


@pytest.mark.parametrize('angle', [0, 1e-9, .02, np.pi-1e-7, np.pi])
def test_log_zero_small_and_pi(angle):
    current, target = np.eye(4), np.eye(4)
    target[:3, :3] = axis_angle([0, 1, 0], angle)
    assert_allclose(dls.pose_error(current, target), [0, 0, 0, 0, angle, 0], atol=3e-9, rtol=0)


def test_normalization_damping_and_solve_path(monkeypatch):
    config = dls.load_solver_config()
    diagonal = np.array([.03, .08, .1, .04, .07, .15])
    j, e = np.diag(diagonal), np.array([.01, .02, -.03, .04, -.05, .06])
    normalized = diagonal / np.array([.9015]*3+[1.]*3)
    en = e / np.array([.9015]*3+[1.]*3)
    expected = normalized * en / (normalized**2 + .05**2)
    solve, calls = np.linalg.solve, []
    def counted(a, b):
        calls.append(a.copy())
        return solve(a, b)
    def forbidden(*args):
        raise AssertionError('explicit inverse forbidden in DLS')
    monkeypatch.setattr(np.linalg, 'solve', counted)
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    assert_allclose(dls.dls_step(j, e, config), expected, atol=1e-15, rtol=1e-14)
    assert len(calls) == 1
    assert_allclose(np.diag(calls[0]), normalized**2+.05**2, atol=1e-16)


def test_joint_step_and_projection():
    cfg = dls.load_solver_config()
    q = np.array([.95, -.95, 0.])
    limits = [(-1., 1.)]*3
    result = dls.projected_update(q, [100, -100, .4], limits, cfg)
    assert_allclose(result, [1, -1, .2], atol=1e-16)
    assert np.max(abs(result-q)) <= .2
    assert_allclose(dls.projected_update([1, -1, 0], [1, -1, 0], limits, cfg), [1, -1, 0])


@pytest.mark.parametrize('j', [np.zeros((6, 6)), np.ones((6, 2)), np.diag([1, 1, 0, 0, 0, 0])])
def test_singular_jacobian(j):
    delta = dls.dls_step(j, np.ones(6), dls.load_solver_config())
    assert np.isfinite(delta).all() and delta.shape == (j.shape[1],)


@pytest.mark.parametrize('field,value', [
    ('q', [0]*5), ('q', [float('nan')]*6), ('q', [float('inf')]*6),
    ('q', [100]*6), ('q', [1j]*6), ('p', [0, 0]), ('p', [np.inf, 0, 0]),
    ('p', [np.nan, 0, 0]), ('quat', [0]*4), ('quat', [1, 0, 0]),
    ('quat', [np.nan, 0, 0, 0]), ('quat', [np.inf, 0, 0, 0]), ('quat', [2, 0, 0, 0])])
def test_invalid_input(robot, real_pose, field, value):
    q, p, quat = real_pose
    values = {'q': q, 'p': p, 'quat': quat}; values[field] = value
    result = dls.DLS(robot).solve(values['q'], values['p'], values['quat'])
    assert result.status == dls.SolverStatus.INVALID_INPUT
    assert result.iterations is None and result.q_candidate is None


@pytest.mark.parametrize('deadline', [0, -1, True, 1.5, np.nan, '10'])
def test_bad_deadline(robot, real_pose, deadline):
    q, p, quat = real_pose
    assert dls.DLS(robot).solve(q, p, quat, deadline_ns=deadline).status == dls.SolverStatus.INVALID_INPUT


@pytest.mark.parametrize('fault', ['nan', 'inf', 'shape', 'solve_error'])
def test_numerical_failure_status(monkeypatch, robot, real_pose, fault):
    q, p, quat = real_pose
    solver = dls.DLS(robot)
    if fault == 'solve_error':
        def broken(*args):
            raise np.linalg.LinAlgError('injected')
        monkeypatch.setattr(np.linalg, 'solve', broken)
    else:
        bad = np.zeros((5, 6)) if fault == 'shape' else np.full((6, 6), float(fault))
        monkeypatch.setattr(solver.geometric, 'jacobian', lambda q: bad)
    result = solver.solve(q, p+[.1, 0, 0], quat)
    assert result.status == dls.SolverStatus.NUMERICAL_FAILURE
    assert result.iterations == 0


def test_stalled_is_not_unreachable(monkeypatch, robot, real_pose):
    q, p, quat = real_pose
    solver = dls.DLS(robot)
    monkeypatch.setattr(solver.geometric, 'jacobian', lambda q: np.zeros((6, 6)))
    result = solver.solve(q, p+[.1, 0, 0], quat)
    assert result.status == dls.SolverStatus.STALLED and result.iterations == 5
    verdict = CandidateValidator(robot).validate(result, p+[.1, 0, 0], quat)
    assert verdict.status == dls.SolverStatus.STALLED
    assert not verdict.profile_a_geometry


def test_iteration_cap(monkeypatch, robot, real_pose):
    q, p, quat = real_pose
    solver = dls.DLS(robot)
    pose = solver.geometric.fk.forward_kinematics(q)
    monkeypatch.setattr(solver.geometric.fk, 'forward_kinematics', lambda q: pose.copy())
    steps = iter([np.full(6, .01), np.full(6, -.01)]*100)
    monkeypatch.setattr(dls, 'dls_step', lambda *args: next(steps))
    result = solver.solve(q, p+[.1, 0, 0], quat)
    assert result.status == dls.SolverStatus.MAX_ITERATIONS
    assert result.iterations == 200


def test_timeout_late_geometry_separate(robot, real_pose):
    q, p, quat = real_pose
    ticks = iter(range(0, 1000, 10))
    result = dls.DLS(robot, clock=lambda: next(ticks)).solve(q, p, quat, deadline_ns=15)
    assert result.status == dls.SolverStatus.TIMEOUT
    verdict = CandidateValidator(robot).validate(result, p, quat)
    assert verdict.profile_b_geometry and verdict.profile_a_geometry
    assert not deadline_success(True, result.status, 1, 15)
    assert not deadline_success(True, dls.SolverStatus.SUCCESS, 16, 15)
    assert deadline_success(True, dls.SolverStatus.SUCCESS, 15, 15)


@pytest.mark.parametrize('p,r,a,b', [(0, 0, True, True), (.0015, .005, True, False),
    (.0005, .012, True, False), (.003, 0, False, False), (0, .02, False, False),
    (np.nan, 0, False, False), (0, np.inf, False, False)])
def test_both_profile_conditions(p, r, a, b):
    assert dls.profile_success(p, r, 'A') == a
    assert dls.profile_success(p, r, 'B') == b


def test_fake_success_flag_rejected(robot, real_pose):
    q, p, quat = real_pose
    result = dls.DLS(robot).solve(q, p, quat)
    verdict = CandidateValidator(robot).validate(result, p+[.1, 0, 0], quat)
    assert verdict.status == dls.SolverStatus.UNRESOLVED
    assert not verdict.profile_a_geometry and not verdict.profile_b_geometry
    empty = CandidateValidator(robot).validate(replace(result, q_candidate=None), p, quat)
    assert empty.status == dls.SolverStatus.UNRESOLVED
    with pytest.raises(ValueError, match='proof'):
        CandidateValidator(robot).validate(replace(result, status=dls.SolverStatus.PROVEN_UNREACHABLE), p, quat)


def test_target_joint_vector_not_in_solver_api(robot, real_pose):
    assert list(inspect.signature(dls.DLS.solve).parameters) == [
        'self', 'q_current', 'target_position_m', 'target_quaternion_wxyz', 'deadline_ns']
    q, p, quat = real_pose
    with pytest.raises(TypeError):
        dls.DLS(robot).solve(q, p, quat, q_target=q)


def test_distinct_first_profile_iterations(analytic_robot):
    robot = analytic_robot(1)
    pose = PinocchioFK(robot).reference_forward_kinematics([.0015])
    result = dls.DLS(robot).solve([0.], pose[:3, 3], canonical_quaternion(pose[:3, :3]))
    assert result.status == dls.SolverStatus.SUCCESS
    assert result.first_profile_a_iteration == 0
    assert result.first_profile_b_iteration == 1


def test_different_joints_same_pose_success(robot, real_pose):
    q, _, _ = real_pose
    other = q.copy(); other[5] += 2*np.pi
    pose = PinocchioFK(robot).reference_forward_kinematics(other)
    result = dls.DLS(robot).solve(q, pose[:3, 3], canonical_quaternion(pose[:3, :3]))
    assert result.status == dls.SolverStatus.SUCCESS and result.iterations == 0
    assert abs(result.q_candidate[5] - other[5]) > 6
    verdict = CandidateValidator(robot).validate(result, pose[:3, 3], canonical_quaternion(pose[:3, :3]))
    assert verdict.profile_b_geometry


def test_solver_projects_and_stalls_at_limit(analytic_robot, monkeypatch):
    robot = analytic_robot(1)
    solver = dls.DLS(robot)
    at_limit = solver.geometric.fk.forward_kinematics([4.])
    target = at_limit.copy()
    target[:3, :3] = axis_angle([0, 0, 1], .1) @ at_limit[:3, :3]
    target[:3, 3] = [np.cos(4.1), np.sin(4.1), 0]
    visited, original = [], solver.geometric.fk.forward_kinematics
    def observed(q):
        assert np.all(np.abs(q) <= 4.)
        visited.append(q.copy())
        return original(q)
    monkeypatch.setattr(solver.geometric.fk, 'forward_kinematics', observed)
    result = solver.solve([4.], target[:3, 3], canonical_quaternion(target[:3, :3]))
    assert result.status == dls.SolverStatus.STALLED and result.iterations == 5
    assert len(visited) > 5 and all(q[0] == 4. for q in visited)


@pytest.mark.parametrize('fault', ['nan', 'shape', 'exception'])
def test_independent_validation_failure(monkeypatch, robot, real_pose, fault):
    q, p, quat = real_pose
    result = dls.DLS(robot).solve(q, p, quat)
    validator = CandidateValidator(robot)
    def broken(q):
        if fault == 'exception': raise RuntimeError('injected independent FK failure')
        return np.full((4, 4), np.nan) if fault == 'nan' else np.eye(3)
    monkeypatch.setattr(validator.reference, 'reference_forward_kinematics', broken)
    verdict = validator.validate(result, p, quat)
    assert verdict.status == dls.SolverStatus.NUMERICAL_FAILURE
    assert not verdict.profile_a_geometry and not verdict.profile_b_geometry


@pytest.mark.parametrize('joint', range(6))
def test_real_robot_small_local_smoke(robot, real_pose, joint):
    q, _, _ = real_pose
    target = q.copy(); target[joint] += .01
    pose = PinocchioFK(robot).reference_forward_kinematics(target)
    p, quat = pose[:3, 3], canonical_quaternion(pose[:3, :3])
    result = dls.DLS(robot).solve(q, p, quat)
    verdict = CandidateValidator(robot).validate(result, p, quat)
    assert result.status == dls.SolverStatus.SUCCESS
    assert verdict.profile_a_geometry and verdict.profile_b_geometry
