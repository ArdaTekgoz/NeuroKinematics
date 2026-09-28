"""Run the SAME analytic acceptance assertion before/after production mutation.

Each mutant replaces an executed production math/validation function. A mutant
is killed only when the acceptance assertion fails, not by array inequality.
"""

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from neurokinematics.solvers import dls
from neurokinematics.benchmark import validation
from neurokinematics.kinematics.transforms import axis_angle


@pytest.mark.parametrize('mutation', ['orientation_sign', 'row_order', 'local_log', 'remove_length',
    'zero_damping', 'changed_damping', 'explicit_inverse', 'remove_step_cap', 'ignore_limits',
    'position_only', 'orientation_only', 'blind_success_flag', 'timeout_pass', 'late_pass'])
def test_production_mutant_fails_acceptance(monkeypatch, robot, real_pose, mutation, record_property):
    cfg = dls.load_solver_config()
    if mutation in ('orientation_sign', 'row_order', 'local_log'):
        original = dls.pose_error
        current = np.eye(4); current[:3, :3] = axis_angle([1, 0, 0], .7)
        target = current.copy(); target[:3, 3] = [.01, -.02, .03]
        target[:3, :3] = axis_angle([0, 0, 1], .01) @ current[:3, :3]
        def gate():
            assert_allclose(dls.pose_error(current, target), [.01, -.02, .03, 0, 0, .01], atol=1e-15)
        def mutant(a, b):
            error = original(a, b)
            if mutation == 'orientation_sign': error[3:] *= -1
            elif mutation == 'row_order': error = np.r_[error[3:], error[:3]]
            else: error[3:] = dls.log_so3(a[:3, :3].T @ b[:3, :3])
            return error
        owner, attr = dls, 'pose_error'
    elif mutation in ('remove_length', 'zero_damping', 'changed_damping', 'explicit_inverse'):
        original = dls.dls_step
        diagonal = np.array([.03, .08, .1, .04, .07, .15])
        error = np.array([.01, .02, -.03, .04, -.05, .06])
        scaled = diagonal/np.array([.9015]*3+[1.]*3)
        expected = scaled*(error/np.array([.9015]*3+[1.]*3))/(scaled**2+.05**2)
        def forbid_inverse(*args):
            raise AssertionError('explicit inverse path executed')
        monkeypatch.setattr(np.linalg, 'inv', forbid_inverse)
        def gate():
            assert_allclose(dls.dls_step(np.diag(diagonal), error, cfg), expected, atol=1e-15)
        def mutant(j, e, config):
            if mutation == 'explicit_inverse':
                return j.T @ np.linalg.inv(j@j.T+config.damping**2*np.eye(6)) @ e
            changed = replace(config, **({'characteristic_length_m': 1.} if mutation == 'remove_length'
                                        else {'damping': 0. if mutation == 'zero_damping' else .1}))
            return original(j, e, changed)
        owner, attr = dls, 'dls_step'
    elif mutation in ('remove_step_cap', 'ignore_limits'):
        original = dls.projected_update
        def gate():
            assert_allclose(dls.projected_update([.95, 0], [1., 1.], [(-1., 1.)]*2, cfg), [1., .2], atol=1e-15)
        def mutant(q, delta, limits, config):
            if mutation == 'remove_step_cap':
                return original(q, delta, limits, replace(config, max_joint_step_rad=100.))
            return np.asarray(q)+np.clip(delta, -.2, .2)
        owner, attr = dls, 'projected_update'
    elif mutation in ('position_only', 'orientation_only'):
        original = validation.profile_success
        q, p, quat = real_pose
        result = dls.DLS(robot).solve(q, p, quat)
        validator = validation.CandidateValidator(robot)
        def gate():
            # One pose differs only in translation, the other only in orientation.
            a = validator.validate(result, p+[.1, 0, 0], quat)
            b = validator.validate(result, p, [1., 0, 0, 0])
            assert not a.profile_a_geometry and not b.profile_a_geometry
        def mutant(p, r, profile):
            return original(p, 0, profile) if mutation == 'position_only' else original(0, r, profile)
        owner, attr = validation, 'profile_success'
    elif mutation == 'blind_success_flag':
        original = validation.CandidateValidator.validate
        q, p, quat = real_pose
        result = dls.DLS(robot).solve(q, p, quat)
        validator = validation.CandidateValidator(robot)
        def gate():
            verdict = validator.validate(result, p+[.1, 0, 0], quat)
            assert verdict.status == dls.SolverStatus.UNRESOLVED
            assert not verdict.profile_b_geometry
        def mutant(self, result, p, quat):
            verdict = original(self, result, p, quat)
            return replace(verdict, status=result.status, profile_a_geometry=True, profile_b_geometry=True)
        owner, attr = validation.CandidateValidator, 'validate'
    else:
        def gate():
            assert not validation.deadline_success(True, dls.SolverStatus.TIMEOUT, 1, 10)
            assert not validation.deadline_success(True, dls.SolverStatus.SUCCESS, 11, 10)
        def mutant(geometry, status, elapsed, deadline):
            return bool(geometry and (elapsed <= deadline if mutation == 'timeout_pass' else status != dls.SolverStatus.TIMEOUT))
        owner, attr = validation, 'deadline_success'
    gate()
    monkeypatch.setattr(owner, attr, mutant)
    with pytest.raises(AssertionError):
        gate()
    record_property('mutation', mutation)
    record_property('production_target', f'{owner.__name__}.{attr}')
    record_property('detected_by_acceptance_assertion', True)
