"""Pinocchio 4.1.0 reference adapter. Nothing here is consumed by custom_fk."""

import numpy as np
import pinocchio as pin

from .model import RobotInputs, validate_q


class PinocchioFK:
    """One mutable Pinocchio Data per instance; use separate instances per thread."""

    def __init__(self, inputs: RobotInputs):
        if pin.__version__ != "4.1.0":
            raise ValueError(f"expected Pinocchio 4.1.0, got {pin.__version__}")
        self.inputs = inputs
        self.model = pin.buildModelFromXML(inputs.urdf.decode("utf-8"))
        self.data = self.model.createData()
        if self.model.nq != len(inputs.joint_names) or self.model.nv != len(inputs.joint_names):
            raise ValueError("reference model must have exactly the contracted scalar joints")
        self.q_indices = {}
        for name in inputs.joint_names:
            if not self.model.existJointName(name):
                raise ValueError(f"reference joint missing: {name}")
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"reference joint is not scalar: {name}")
            self.q_indices[name] = joint.idx_q
        self.frame_ids = {}
        for name in (inputs.base, inputs.tcp):
            frame_id = self.model.getFrameId(name, pin.FrameType.BODY)
            if frame_id >= self.model.nframes:
                raise ValueError(f"reference body frame missing: {name}")
            self.frame_ids[name] = frame_id

    def reference_forward_kinematics(self, q) -> np.ndarray:
        values = validate_q(q, self.inputs.joint_names, self.inputs.limits)
        pin_q = np.zeros(self.model.nq, dtype=np.float64)
        for index, name in enumerate(self.inputs.joint_names):
            pin_q[self.q_indices[name]] = values[index]
        pin.forwardKinematics(self.model, self.data, pin_q)
        pin.updateFramePlacements(self.model, self.data)
        world_base = self.data.oMf[self.frame_ids[self.inputs.base]]
        world_tcp = self.data.oMf[self.frame_ids[self.inputs.tcp]]
        # Invert the represented float64 matrix explicitly. SE3.inverse uses R.T,
        # which assumes exact orthonormality and magnifies rounding at an offset base.
        relative = np.linalg.inv(np.asarray(world_base.homogeneous, dtype=np.float64)) @ np.asarray(
            world_tcp.homogeneous, dtype=np.float64
        )
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = relative[:3, :3]
        result[:3, 3] = relative[:3, 3]
        return result
