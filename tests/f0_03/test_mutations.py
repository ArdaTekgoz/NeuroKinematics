"""Faults live only in in-memory XML/arrays or temporary fixture functions."""

from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian
from neurokinematics.kinematics.finite_difference import CentralDifference, log_so3
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.transforms import axis_angle


FAULTS = ["row_swap", "reversed_cross", "flange_not_tcp", "local_axis", "world_not_base",
          "reverse_joint_order", "skip_fixed", "axis_sign", "pinocchio_local", "euler_derivative",
          "forward_difference", "degrees_radians"]


@pytest.mark.parametrize("fault", FAULTS)
def test_mutations_are_detected(planar, robot, fault, record_property):
    q = np.array([.3, -.7])
    inputs = planar
    root = ET.fromstring(inputs.urdf)
    # Tilt the first joint so local-axis and Euler faults cannot hide on planar Z.
    root.find("joint[@name='j1']/origin").set("rpy", "0.4 -0.6 0.2")
    inputs = replace(inputs, urdf=ET.tostring(root))
    expected = IndependentJacobian(inputs).jacobian(q)
    wrong = expected.copy()
    threshold = 1e-5
    if fault == "row_swap":
        wrong = np.vstack((wrong[3:], wrong[:3]))
    elif fault == "reversed_cross":
        wrong[:3] *= -1
    elif fault == "flange_not_tcp":
        wrong = IndependentJacobian(replace(inputs, tcp="flange")).jacobian(q)
    elif fault == "local_axis":
        fk = IndependentFK(inputs)
        p = fk.forward_kinematics(q)[:3, 3]
        t = np.eye(4)
        for joint in fk.chain:
            t = t @ joint.origin
            if joint.axis is not None:
                i = fk.q_index[joint.name]
                wrong[:3, i] = np.cross(joint.axis, p-t[:3, 3])
                wrong[3:, i] = joint.axis
                m = np.eye(4); m[:3, :3] = axis_angle(joint.axis, q[i])
                t = t @ m
    elif fault in ("world_not_base", "pinocchio_local"):
        ref = PinocchioJacobian(inputs); ref.jacobian(q)
        frame = pin.LOCAL_WORLD_ALIGNED if fault == "world_not_base" else pin.LOCAL
        raw = pin.getFrameJacobian(ref.model, ref.data, ref.frame_ids[inputs.tcp], frame)
        for i in range(2):
            motion = pin.Motion(raw[:, i])
            wrong[:, i] = np.r_[motion.linear, motion.angular]
    elif fault == "reverse_joint_order":
        wrong = wrong[:, ::-1]
    elif fault in ("skip_fixed", "axis_sign"):
        if fault == "skip_fixed":
            root.find("joint[@name='spacer']/origin").set("xyz", "0 0 0")
        else:
            root.find("joint[@name='j1']/axis").set("xyz", "0 0 -1")
        wrong = IndependentJacobian(replace(inputs, urdf=ET.tostring(root))).jacobian(q)
    elif fault in ("euler_derivative", "forward_difference"):
        fk = IndependentFK(inputs)
        h = 1e-6
        t = fk.forward_kinematics(q)
        def euler(r):
            return np.array([np.arctan2(r[2, 1], r[2, 2]),
                             np.arcsin(-r[2, 0]), np.arctan2(r[1, 0], r[0, 0])])
        for i in range(2):
            d = np.zeros(2); d[i] = h
            plus, minus = fk.forward_kinematics(q+d), fk.forward_kinematics(q-d)
            if fault == "euler_derivative":
                wrong[3:, i] = (euler(plus[:3, :3])-euler(minus[:3, :3]))/(2*h)
            else:
                wrong[:3, i] = (plus[:3, 3]-t[:3, 3])/h
                wrong[3:, i] = log_so3(plus[:3, :3] @ t[:3, :3].T)/h
        if fault == "forward_difference":
            # The 1e-5 robot gate alone cannot distinguish this stencil at h=1e-6.
            # The analytic central-accuracy gate must do so (no threshold tuning).
            threshold = 1e-8
            assert np.linalg.norm(CentralDifference(inputs).jacobian(q)-expected) < threshold
    else:
        wrong = IndependentJacobian(inputs).jacobian(np.deg2rad(q))
    error = float(np.linalg.norm(wrong-expected))
    record_property("mutation", fault)
    record_property("error_frobenius", error)
    record_property("detection_threshold", threshold)
    record_property("detected", error > threshold)
    assert error > threshold


def test_central_second_order_not_forward(planar):
    q = [.3, -.7]
    correct = IndependentJacobian(planar).jacobian(q)
    fd = CentralDifference(planar)
    errors = [np.linalg.norm(fd.jacobian(q, h)-correct) for h in (.01, .005)]
    assert 3.99 < errors[0]/errors[1] < 4.01
