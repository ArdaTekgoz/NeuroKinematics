"""Pinocchio 4.1.0 TCP Jacobian -> [linear; angular] in fixed base axes.

LOCAL_WORLD_ALIGNED is at the frame origin with world-oriented axes. WORLD
uses the world origin; LOCAL uses TCP axes. Neither is interchangeable here.
Rotate both LWA components by R_world_base.T, without shifting the TCP point.
See https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/group__pinocchio__multibody.html
"""

import numpy as np
import pinocchio as pin

from .model import validate_q
from .pinocchio_fk import PinocchioFK


class PinocchioJacobian(PinocchioFK):
    def __init__(self, inputs):
        # F0-02 builds from the supplied immutable XML bytes and resolves BODY
        # frames and name -> idx_q. Jacobian columns additionally need idx_v.
        super().__init__(inputs)
        self.v_indices = {name: self.model.joints[self.model.getJointId(name)].idx_v
                          for name in inputs.joint_names}
        if self.model.frames[self.frame_ids[inputs.base]].parentJoint != 0:
            raise ValueError("Jacobian base must be fixed to world")

    def jacobian(self, q):
        q = validate_q(q, self.inputs.joint_names, self.inputs.limits)
        pin_q = np.zeros(self.model.nq, dtype=np.float64)
        for i, name in enumerate(self.inputs.joint_names):
            pin_q[self.q_indices[name]] = q[i]
        pin.computeJointJacobians(self.model, self.data, pin_q)
        pin.updateFramePlacements(self.model, self.data)
        raw = pin.getFrameJacobian(self.model, self.data, self.frame_ids[self.inputs.tcp],
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # EigenPy squeezes a single-column Matrix6x to a length-six vector.
        raw = np.asarray(raw, dtype=np.float64).reshape(6, self.model.nv)
        rotation = self.data.oMf[self.frame_ids[self.inputs.base]].rotation.T
        result = np.empty((6, len(q)), dtype=np.float64)
        for i, name in enumerate(self.inputs.joint_names):
            # Motion interprets Pinocchio's own representation; never slice its
            # raw vector under an assumed linear/angular row convention.
            motion = pin.Motion(raw[:, self.v_indices[name]])
            result[:3, i] = rotation @ motion.linear
            result[3:, i] = rotation @ motion.angular
        return result
