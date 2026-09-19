from dataclasses import replace
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from numpy.testing import assert_allclose
import pinocchio as pin
import pytest

from conftest import make_robot
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian
from neurokinematics.kinematics.finite_difference import CentralDifference, log_so3
from neurokinematics.kinematics.transforms import axis_angle


@pytest.mark.parametrize("q", [[0, 0], [np.pi/2, 0], [np.pi/2, -np.pi/2], [.3, -.7]])
@pytest.mark.parametrize("backend", [IndependentJacobian, PinocchioJacobian, CentralDifference])
def test_planar_fixed_tcp_and_nonidentity_base(planar, q, backend):
    a, b = q
    expected = np.array([[-3*np.sin(a)-2*np.sin(a+b), -2*np.sin(a+b)],
                         [3*np.cos(a)+2*np.cos(a+b), 2*np.cos(a+b)],
                         [0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.float64)
    actual = backend(planar).jacobian(q)
    assert actual.dtype == np.float64
    assert_allclose(actual, expected, atol=2e-9 if backend == CentralDifference else 3e-15, rtol=0)


@pytest.mark.parametrize("axis,offset,expected", [
    ("0 0 1", "2 0 0", [0, 2, 0, 0, 0, 1]),
    ("0 1 0", "2 0 0", [0, 0, -2, 0, 1, 0]),
    ("1 0 0", "0 2 0", [0, 0, 2, 1, 0, 0]),
])
def test_one_joint_different_axes(axis, offset, expected):
    inputs = make_robot([
        ("mount", "fixed", "world", "base", "1 2 3", "0.7 0.4 -0.3", ""),
        ("j", "revolute", "base", "flange", "0 0 0", "0 0 0", axis),
        ("tool", "fixed", "flange", "tcp", offset, "0 0 0", "")], ["j"])
    for backend in (IndependentJacobian, PinocchioJacobian, CentralDifference):
        assert_allclose(backend(inputs).jacobian([0])[:, 0], expected, atol=2e-12, rtol=0)


def test_rotated_joint_origin(planar):
    root = ET.fromstring(planar.urdf)
    root.find("joint[@name='j1']/origin").set("rpy", "1.5707963267948966 0 0")
    inputs = replace(planar, urdf=ET.tostring(root))
    # Rx(pi/2) takes local +Z -> base -Y and planar +Y -> base +Z.
    expected = np.array([[-3, 0], [0, 0], [2, 2], [0, 0], [-1, -1], [0, 0]], dtype=float)
    for backend in (IndependentJacobian, PinocchioJacobian, CentralDifference):
        assert_allclose(backend(inputs).jacobian([np.pi/2, -np.pi/2]), expected, atol=2e-9, rtol=0)


def test_manifest_order_and_idx_v(planar):
    q = [.3, -.7]
    reversed_inputs = replace(planar, joint_names=planar.joint_names[::-1])
    for backend in (IndependentJacobian, PinocchioJacobian, CentralDifference):
        assert_allclose(backend(reversed_inputs).jacobian(q[::-1]), backend(planar).jacobian(q)[:, ::-1], atol=1e-12, rtol=0)
    assert PinocchioJacobian(reversed_inputs).v_indices == {"j2": 1, "j1": 0}


def test_reference_frame_semantics_and_motion_properties(planar):
    ref = PinocchioJacobian(planar)
    actual = ref.jacobian([.3, -.7])
    frame = ref.frame_ids[planar.tcp]
    world_base = ref.data.oMf[ref.frame_ids[planar.base]].rotation
    world_tcp = ref.data.oMf[frame]
    lwa = pin.getFrameJacobian(ref.model, ref.data, frame, pin.LOCAL_WORLD_ALIGNED)
    local = pin.getFrameJacobian(ref.model, ref.data, frame, pin.LOCAL)
    world = pin.getFrameJacobian(ref.model, ref.data, frame, pin.WORLD)
    for i in range(2):
        l, w, a = pin.Motion(local[:, i]), pin.Motion(world[:, i]), pin.Motion(lwa[:, i])
        assert_allclose(world_tcp.rotation @ l.linear, a.linear, atol=3e-15)
        assert_allclose(world_tcp.rotation @ l.angular, a.angular, atol=3e-15)
        assert_allclose(w.linear + np.cross(w.angular, world_tcp.translation), a.linear, atol=3e-15)
        assert_allclose(world_base @ actual[:3, i], a.linear, atol=3e-15)
        assert_allclose(world_base @ actual[3:, i], a.angular, atol=3e-15)
    assert np.linalg.norm(local - actual) > .1
    assert np.linalg.norm(lwa - actual) > .1
    assert np.linalg.norm(world - lwa) > .1


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -2, 3]])
@pytest.mark.parametrize("angle", [0, 1e-12, -1e-8, 1e-5, .3, np.pi/2, np.pi-1e-7, np.pi, -np.pi+1e-7])
def test_so3_log_analytic_small_and_pi(axis, angle):
    axis = np.array(axis, dtype=float); axis /= np.linalg.norm(axis)
    actual = log_so3(axis_angle(axis, angle))
    expected = axis * angle
    assert_allclose(actual, expected, atol=3e-9, rtol=0)
    assert_allclose(axis_angle(actual if np.linalg.norm(actual) else [1, 0, 0], np.linalg.norm(actual)),
                    axis_angle(axis, angle), atol=3e-9, rtol=0)


def test_so3_left_increment_is_base_frame():
    r = axis_angle([1, 0, 0], .7)
    left = axis_angle([0, 0, 1], 2e-6) @ r
    assert_allclose(log_so3(left @ r.T), [0, 0, 2e-6], atol=1e-16, rtol=0)
    assert np.linalg.norm(log_so3(r.T @ left) - [0, 0, 2e-6]) > 1e-7


@pytest.mark.parametrize("h", [0, -1e-6, np.inf, np.nan])
def test_bad_step(robot, h):
    with pytest.raises(ValueError):
        CentralDifference(robot).jacobian(np.zeros(6), h)


@pytest.mark.parametrize("bound", [0, 1])
def test_central_limit_violation_rejected(robot, bound):
    with pytest.raises(ValueError, match="outside joint limits"):
        CentralDifference(robot).jacobian(np.array(robot.limits)[:, bound], 1e-6)


def test_no_pinocchio_dependency():
    script = '''
import builtins, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if 'pinocchio' in name:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from neurokinematics.kinematics.model import load_robot
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.finite_difference import CentralDifference
for cls in (IndependentJacobian, CentralDifference):
    assert cls(load_robot()).jacobian([0]*6).shape == (6,6)
assert 'pinocchio' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
