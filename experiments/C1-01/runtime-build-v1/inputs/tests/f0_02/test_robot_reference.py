from dataclasses import replace
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from numpy.testing import assert_allclose
import pytest

from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.model import FROZEN_HASHES, ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.validation import compare, handpicked_configurations


def test_immutable_inputs(robot):
    assert robot.hashes == FROZEN_HASHES
    assert robot.base == "base_link" and robot.tip == "flange" and robot.tcp == "tool0"


@pytest.mark.parametrize("changed", list(FROZEN_HASHES))
def test_changed_immutable_bytes_rejected(tmp_path, changed):
    for relative in FROZEN_HASHES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (ROOT / relative).read_bytes()
        path.write_bytes(payload + b" " if relative == changed else payload)
    with pytest.raises(ValueError, match="immutable hash mismatch"):
        load_robot(tmp_path)


def test_chain_fixed_joints_and_side_branch(robot):
    fk = IndependentFK(robot)
    assert [j.name for j in fk.chain] == list(robot.joint_names) + ["joint_6-flange", "flange-tool0"]
    assert "base_link-base" not in [j.name for j in fk.chain]
    assert len(fk.q_index) == 6


@pytest.mark.parametrize("name", ["zero", "midpoints", "lower_near", "upper_near", "lower_exact", "upper_exact", "mixed_1", "mixed_2", "mixed_3"] + [f"joint_{i}_{s}" for i in range(1, 7) for s in ("positive", "negative")])
def test_handpicked_robot(robot, name):
    q = handpicked_configurations(robot)[name]
    a = IndependentFK(robot).forward_kinematics(q)
    b = PinocchioFK(robot).reference_forward_kinematics(q)
    metrics = compare(a, b)
    assert metrics["position_error_m"] <= 1e-9
    assert metrics["rotation_frobenius_error"] <= 1e-9
    if name == "zero":
        # Sum of supplied URDF offsets and explicit +90deg TCP about Y.
        assert_allclose(a[:3, 3], [.980, 0, .435], rtol=0, atol=2e-16)
        assert_allclose(a[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], rtol=0, atol=2e-16)


def test_reference_relative_base_and_name_mapping(small_robot, robot):
    q = [np.pi / 2, -np.pi / 2]
    reference = PinocchioFK(small_robot)
    result = reference.reference_forward_kinematics(q)
    expected = np.array([[1, 0, 0, 3], [0, 0, -1, 1], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    assert_allclose(result, expected, atol=2e-15, rtol=0)
    world_base = reference.data.oMf[reference.frame_ids["base"]].homogeneous
    assert not np.allclose(world_base, np.eye(4))
    actual_robot = PinocchioFK(robot)
    actual_robot.reference_forward_kinematics(np.zeros(6))
    assert_allclose(actual_robot.data.oMf[actual_robot.frame_ids["base_link"]].homogeneous, np.eye(4), atol=0, rtol=0)
    reversed_inputs = replace(small_robot, joint_names=("j2", "j1"))
    reverse_reference = PinocchioFK(reversed_inputs)
    assert reverse_reference.q_indices == {"j2": 1, "j1": 0}
    assert_allclose(reverse_reference.reference_forward_kinematics(q[::-1]), expected, atol=2e-15, rtol=0)
    assert_allclose(IndependentFK(reversed_inputs).forward_kinematics(q[::-1]), expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("q", [[0] * 5, [[0] * 6], [0] * 7, [np.nan] * 6, [np.inf] * 6, [-np.inf] * 6, [90, 0, 0, 0, 0, 0], [0, 0.786, 0, 0, 0, 0], [1j] * 6, ["0"] * 6])
def test_bad_q_both_backends(robot, q):
    for call in (IndependentFK(robot).forward_kinematics, PinocchioFK(robot).reference_forward_kinematics):
        with pytest.raises(ValueError):
            call(q)


def test_one_ulp_outside_limits_rejected(robot):
    for column, direction in ((0, -np.inf), (1, np.inf)):
        q = np.asarray(robot.limits)[:, column].copy()
        for index in range(6):
            invalid = q.copy()
            invalid[index] = np.nextafter(invalid[index], direction)
            for call in (IndependentFK(robot).forward_kinematics, PinocchioFK(robot).reference_forward_kinematics):
                with pytest.raises(ValueError):
                    call(invalid)


@pytest.mark.parametrize("fault", ["axis_sign", "omit_tcp", "wrong_order", "degrees", "wrong_base"])
def test_robot_regression_mutants_detected(robot, fault):
    q = np.array([.3, -.6, .8, -1., .5, -.7])
    expected = PinocchioFK(robot).reference_forward_kinematics(q)
    root = ET.fromstring(robot.urdf)
    if fault == "axis_sign":
        root.find("joint[@name='joint_1']/axis").set("xyz", "0 0 1")
    elif fault == "omit_tcp":
        root.find("joint[@name='flange-tool0']/origin").set("rpy", "0 0 0")
    elif fault == "wrong_base":
        # Real side-base is identity; a nonidentity mutation exposes frame confusion.
        wrong_base = np.eye(4); wrong_base[0, 3] = 1
        actual = wrong_base @ expected
    mutated = replace(robot, urdf=ET.tostring(root))
    if fault != "wrong_base":
        actual = IndependentFK(mutated).forward_kinematics(q[::-1] if fault == "wrong_order" else np.deg2rad(q) if fault == "degrees" else q)
    errors = compare(actual, expected)
    assert errors["position_error_m"] > 1e-9 or errors["rotation_frobenius_error"] > 1e-9
    if fault == "omit_tcp":
        flange = PinocchioFK(replace(robot, tcp="flange")).reference_forward_kinematics(q)
        assert_allclose(actual, flange, atol=2e-15, rtol=0)


def test_nonidentity_fixed_joint_omission_detected(small_robot):
    q = [.3, -.7]
    original = IndependentFK(small_robot).forward_kinematics(q)
    root = ET.fromstring(small_robot.urdf)
    root.find("joint[@name='fixed_spacer']/origin").set("xyz", "0 0 0")
    changed = IndependentFK(replace(small_robot, urdf=ET.tostring(root))).forward_kinematics(q)
    assert np.linalg.norm(original[:3, 3] - changed[:3, 3]) > 1.9


def test_custom_fk_without_pinocchio_import_or_oracle():
    script = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if any(part in name.split('.') for part in ('pinocchio', 'pinocchio_fk', 'data', 'solvers', 'training', 'gui')):
        raise AssertionError('forbidden dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from neurokinematics.kinematics import IndependentFK, load_robot
result = IndependentFK(load_robot()).forward_kinematics([0] * 6)
assert result.shape == (4,4)
assert 'pinocchio' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("frame", ["base", "tcp"])
def test_missing_reference_frame_rejected(robot, frame):
    with pytest.raises(ValueError, match="frame missing"):
        PinocchioFK(replace(robot, **{frame: "absent"}))


def test_missing_reference_joint_rejected(robot):
    with pytest.raises(ValueError, match="joint missing"):
        PinocchioFK(replace(robot, joint_names=("absent",) + robot.joint_names[1:]))
