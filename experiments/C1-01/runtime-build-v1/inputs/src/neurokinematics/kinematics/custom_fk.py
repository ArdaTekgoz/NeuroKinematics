"""NumPy FK from independently parsed URDF; no Pinocchio/model placement dependency."""

import numpy as np

from .chain import extract_chain
from .model import RobotInputs, validate_q
from .transforms import axis_angle


class IndependentFK:
    def __init__(self, inputs: RobotInputs):
        self.inputs = inputs
        self.chain = extract_chain(inputs.urdf, inputs.base, inputs.tcp, inputs.joint_names)
        self.q_index = {name: index for index, name in enumerate(inputs.joint_names)}
        for joint in self.chain:
            if joint.kind == "revolute":
                if joint.limits != inputs.limits[self.q_index[joint.name]]:
                    raise ValueError(f"URDF/RobotSpec limits differ: {joint.name}")

    def forward_kinematics(self, q) -> np.ndarray:
        values = validate_q(q, self.inputs.joint_names, self.inputs.limits)
        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            # Origin is applied BEFORE rotation about the joint-local axis.
            transform = transform @ joint.origin
            if joint.kind == "revolute":
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = axis_angle(joint.axis, values[self.q_index[joint.name]])
                transform = transform @ motion
        return transform
