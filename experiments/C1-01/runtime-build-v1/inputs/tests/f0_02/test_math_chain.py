from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np
from numpy.testing import assert_allclose
import pytest

from neurokinematics.kinematics.chain import extract_chain
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.transforms import axis_angle, origin_transform, check_transform


@pytest.mark.parametrize("axis,quarter,half", [
    ([1, 0, 0], [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [1, -1, -1]),
    ([0, 1, 0], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [-1, 1, -1]),
    ([0, 0, 1], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [-1, -1, 1]),
])
@pytest.mark.parametrize("turn", [0, 1, -1, 2])
def test_axis_rotations_analytical(axis, quarter, half, turn):
    expected = {0: np.eye(3), 1: np.array(quarter), -1: np.array(quarter).T, 2: np.diag(half)}[turn]
    assert_allclose(axis_angle(axis, turn * np.pi / 2), expected, rtol=0, atol=3e-16)


def test_general_axis_angle():
    # 120 degrees about (1,1,1) cyclically permutes the Cartesian basis.
    expected = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert_allclose(axis_angle([1, 1, 1], 2 * np.pi / 3), expected, atol=5e-16, rtol=0)


def test_identity_translation_and_rpy_order():
    assert_allclose(origin_transform(), np.eye(4), rtol=0, atol=0)
    t = origin_transform([2, -3, 4])
    assert_allclose(t @ [1, 1, 1, 1], [3, -2, 5, 1], rtol=0, atol=0)
    # Three +90 degree fixed-axis rotations Rz Ry Rx reduce to Ry(+90).
    t = origin_transform(rpy=[np.pi / 2] * 3)
    assert_allclose(t[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=2e-16, rtol=0)
    wrong = axis_angle([1, 0, 0], np.pi / 2) @ axis_angle([0, 1, 0], np.pi / 2) @ axis_angle([0, 0, 1], np.pi / 2)
    assert np.linalg.norm(t[:3, :3] - wrong) > 1


def test_two_joint_fixed_tcp_and_origin_before_motion(small_robot):
    fk = IndependentFK(small_robot)
    result = fk.forward_kinematics([np.pi / 2, -np.pi / 2])
    expected = np.array([[1, 0, 0, 3], [0, 0, -1, 1], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    assert_allclose(result, expected, atol=5e-16, rtol=0)
    check_transform(result)
    assert [j.name for j in fk.chain] == ["j1", "j2", "fixed_spacer", "tcp_joint"]
    # Origin belongs before motion; swapping on j2 changes the hand-derived pose.
    wrong = np.eye(4)
    for j in fk.chain:
        motion = np.eye(4)
        if j.axis is not None:
            motion[:3, :3] = axis_angle(j.axis, [np.pi / 2, -np.pi / 2][fk.q_index[j.name]])
        wrong = wrong @ motion @ j.origin
    assert np.linalg.norm(wrong[:3, 3] - expected[:3, 3]) > 1


def test_joint_axis_is_local_after_rotated_origin(small_robot):
    root = ET.fromstring(small_robot.urdf)
    ET.SubElement(root.find("joint[@name='j1']"), "origin", rpy="1.5707963267948966 0 0")
    fk = IndependentFK(replace(small_robot, urdf=ET.tostring(root)))
    result = fk.forward_kinematics([np.pi / 2, 0])
    assert_allclose(result[:3, 3], [0, 0, 4], atol=1e-15, rtol=0)


def test_fixed_joint_between_active_joints(small_robot):
    root = ET.fromstring(small_robot.urdf)
    root.find("joint[@name='fixed_spacer']/parent").set("link", "a")
    root.find("joint[@name='j2']/parent").set("link", "spacer")
    root.find("joint[@name='tcp_joint']/parent").set("link", "b")
    fk = IndependentFK(replace(small_robot, urdf=ET.tostring(root)))
    assert [j.name for j in fk.chain] == ["j1", "fixed_spacer", "j2", "tcp_joint"]
    result = fk.forward_kinematics([np.pi / 2, -np.pi / 2])
    assert_allclose(result[:3, 3], [1, 3, 0], atol=5e-16, rtol=0)


@pytest.mark.parametrize("fault", ["duplicate_link", "duplicate_joint", "multiple_parent", "cycle", "disconnected", "missing_frame", "prismatic", "mimic", "missing_axis", "bad_axis", "missing_limit", "nonfinite_limit", "bad_origin", "wrong_names", "extra_active"])
def test_bad_urdf_rejected(small_robot, fault):
    root = ET.fromstring(small_robot.urdf)
    j = root.find("joint[@name='j1']")
    names = small_robot.joint_names
    base = "base"
    if fault == "duplicate_link":
        ET.SubElement(root, "link", name="base")
    elif fault == "duplicate_joint":
        root.append(ET.fromstring(ET.tostring(j)))
    elif fault == "multiple_parent":
        other = ET.fromstring(ET.tostring(j)); other.set("name", "other"); root.append(other)
    elif fault == "cycle":
        root.find("joint[@name='mount']/parent").set("link", "tcp")
    elif fault == "disconnected":
        ET.SubElement(root, "link", name="loose")
    elif fault == "missing_frame":
        base = "absent"
    elif fault == "prismatic":
        j.set("type", "prismatic")
    elif fault == "mimic":
        ET.SubElement(j, "mimic", joint="j2")
    elif fault == "missing_axis":
        j.remove(j.find("axis"))
    elif fault == "bad_axis":
        j.find("axis").set("xyz", "0 0 0")
    elif fault == "missing_limit":
        j.remove(j.find("limit"))
    elif fault == "nonfinite_limit":
        j.find("limit").set("lower", "nan")
    elif fault == "bad_origin":
        ET.SubElement(j, "origin", xyz="nan 0 0")
    elif fault == "wrong_names":
        names = ("j1", "missing")
    elif fault == "extra_active":
        root.find("joint[@name='mount']").set("type", "revolute")
        mount = root.find("joint[@name='mount']")
        ET.SubElement(mount, "axis", xyz="1 0 0")
        ET.SubElement(mount, "limit", lower="-1", upper="1")
    with pytest.raises(ValueError):
        extract_chain(ET.tostring(root), base, "tcp", names)


@pytest.mark.parametrize("axis,angle", [([0, 0, 0], 0), ([1, 0, 0], np.nan), ([1, 0], 0), ([np.inf, 0, 0], 0)])
def test_bad_axis_angle(axis, angle):
    with pytest.raises(ValueError):
        axis_angle(axis, angle)


def test_fixed_only_identity_and_translation():
    xml = b'<robot name="fixed"><link name="b"/><link name="t"/><joint name="f" type="fixed"><parent link="b"/><child link="t"/><origin xyz="2 3 4"/></joint></robot>'
    chain = extract_chain(xml, "b", "t", ())
    assert_allclose(chain[0].origin[:3, 3], [2, 3, 4], atol=0, rtol=0)
    assert_allclose(chain[0].origin[:3, :3], np.eye(3), atol=0, rtol=0)
