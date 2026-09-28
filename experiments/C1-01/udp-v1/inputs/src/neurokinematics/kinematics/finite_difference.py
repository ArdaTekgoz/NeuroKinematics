"""Central TCP finite difference using only IndependentFK; no Euler derivative."""

import numpy as np

from .custom_fk import IndependentFK
from .metrics import rotation_matrix
from .model import validate_q


def log_so3(value):
    """Principal rotation vector, right-hand rule, angle in [0, pi].

vee(R-R.T)/2 = sin(theta)*axis. atan2 avoids arccos loss near zero.
Small angles use the theta/sin(theta) series. Near pi the symmetric matrix's
unit eigenvector avoids division by sin(theta); skew fixes its sign off pi.
At exact pi either axis sign is valid; choose largest component positive.
For R_plus @ R_minus.T this vector is expressed in BASE axes (left increment).
"""
    r = rotation_matrix(value)
    v = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]) / 2
    sine = float(np.linalg.norm(v))
    cosine = float(np.clip((np.trace(r)-1)/2, -1, 1))
    theta = float(np.arctan2(sine, cosine))
    if theta < 1e-4:
        return v * (1 + theta**2 / 6 + 7 * theta**4 / 360)
    if np.pi - theta < 1e-5:
        _, vectors = np.linalg.eigh((r + r.T) / 2)
        axis = vectors[:, -1]
        if sine > 1e-14:
            if np.dot(axis, v) < 0:
                axis = -axis
        elif axis[np.argmax(abs(axis))] < 0:
            axis = -axis
        return theta * axis
    return theta / sine * v


class CentralDifference:
    def __init__(self, inputs):
        self.inputs = inputs
        self.fk = IndependentFK(inputs)

    def jacobian(self, q, h=1e-6):
        q = validate_q(q, self.inputs.joint_names, self.inputs.limits)
        if not np.isscalar(h) or not np.isfinite(h) or h <= 0:
            raise ValueError("h must be positive finite radians")
        result = np.empty((6, len(q)), dtype=np.float64)
        for i in range(len(q)):
            delta = np.zeros(len(q), dtype=np.float64)
            delta[i] = h
            # FK rejects either out-of-limit point; never clip or switch stencil.
            plus = self.fk.forward_kinematics(q + delta)
            minus = self.fk.forward_kinematics(q - delta)
            result[:3, i] = (plus[:3, 3] - minus[:3, 3]) / (2*h)
            result[3:, i] = log_so3(plus[:3, :3] @ minus[:3, :3].T) / (2*h)
        return result
