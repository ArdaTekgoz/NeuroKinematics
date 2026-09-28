"""Independent geometric Jacobian at the TCP, expressed in base axes.

Uses only the F0-02 independently parsed chain and IndependentFK. Fixed joints
participate in propagation but never create columns. No reference backend.
"""

import numpy as np

from .custom_fk import IndependentFK
from .model import validate_q
from .transforms import axis_angle


class IndependentJacobian:
    def __init__(self, inputs):
        self.fk = IndependentFK(inputs)
        self.inputs = inputs

    def jacobian(self, q):
        q = validate_q(q, self.inputs.joint_names, self.inputs.limits)
        tcp = self.fk.forward_kinematics(q)[:3, 3]
        result = np.zeros((6, len(q)), dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        for joint in self.fk.chain:
            transform = transform @ joint.origin
            if joint.kind == "revolute":
                index = self.fk.q_index[joint.name]
                axis = transform[:3, :3] @ joint.axis
                result[:3, index] = np.cross(axis, tcp - transform[:3, 3])
                result[3:, index] = axis
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = axis_angle(joint.axis, q[index])
                transform = transform @ motion
        return result
