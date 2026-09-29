"""Differentiable, relative base-to-TCP FK. No numerical reference in forward.

Constants originate in the verified fixed/revolute URDF parser. The q-dependent
calculation uses only ordinary Torch operations; no custom backward is used.
"""
from pathlib import Path

import torch

from .chain import extract_chain
from .model import ROOT, RobotInputs, load_robot


class TorchSerialChain:
    """Internal kernel; explicit RobotInputs also permit isolated analytic fixtures.

    Production callers should use TorchFK.from_frozen(), which verifies asset bytes.
    Constants retain float64 master precision across alternating input dtypes.
    """

    def __init__(self, inputs: RobotInputs):
        self.joint_names = tuple(inputs.joint_names)
        self.robot_id = inputs.robot_id
        self.base, self.tcp = inputs.base, inputs.tcp
        self.chain = extract_chain(inputs.urdf, inputs.base, inputs.tcp, self.joint_names)
        self._limits = torch.tensor(inputs.limits, dtype=torch.float64)
        if (self._limits.shape != (len(self.joint_names), 2)
                or not torch.isfinite(self._limits).all()
                or not torch.all(self._limits[:, 0] < self._limits[:, 1])):
            raise ValueError('invalid joint limits')
        self._steps = []
        for joint in self.chain:
            index = self.joint_names.index(joint.name) if joint.kind == 'revolute' else None
            if index is not None and joint.limits != inputs.limits[index]:
                raise ValueError('URDF/RobotSpec limits differ: ' + joint.name)
            origin = torch.tensor(joint.origin.copy(), dtype=torch.float64)
            if index is None:
                skew = None
            else:
                x, y, z = joint.axis.tolist()
                skew = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float64)
            self._steps.append((index, origin, skew))

    def __call__(self, q):
        if not isinstance(q, torch.Tensor):
            raise TypeError('q must be a Torch tensor')
        if q.layout != torch.strided or q.dtype not in (torch.float32, torch.float64):
            raise TypeError('q must be dense float32 or float64')
        if q.ndim not in (1, 2) or q.shape[-1] != len(self.joint_names) or (q.ndim == 2 and q.shape[0] == 0):
            raise ValueError('q shape must be (dof,) or (N,dof), N >= 1')
        if not torch.isfinite(q).all():
            raise ValueError('q must be finite')
        limits = self._limits.to(device=q.device)
        checked = q.to(dtype=torch.float64)
        if torch.any(checked < limits[:, 0]) or torch.any(checked > limits[:, 1]):
            raise ValueError('q outside joint limits (radians)')
        values = q.unsqueeze(0) if q.ndim == 1 else q
        n = values.shape[0]
        transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(n, 4, 4)
        eye = torch.eye(3, dtype=q.dtype, device=q.device)
        for index, origin64, skew64 in self._steps:
            origin = origin64.to(dtype=q.dtype, device=q.device)
            transform = transform @ origin
            if index is not None:
                skew = skew64.to(dtype=q.dtype, device=q.device)
                angle = values[:, index, None, None]
                rotation = eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)
                upper = torch.cat((rotation, values.new_zeros((n, 3, 1))), dim=-1)
                lower = values.new_tensor([0., 0., 0., 1.]).expand(n, 1, 4)
                motion = torch.cat((upper, lower), dim=-2)
                transform = transform @ motion
        return transform[0] if q.ndim == 1 else transform


class TorchFK:
    """Frozen six-joint public API; explicit metadata guards order and units.

    (6,) -> (4,4), (N,6) -> (N,4,4), same input dtype/device.
    Tensor values alone cannot reveal a falsely labelled order or unit.
    """

    def __init__(self, root: Path = ROOT):
        inputs = load_robot(Path(root))
        self.robot_id = inputs.robot_id
        self.joint_names = inputs.joint_names
        self.hashes = dict(inputs.hashes)
        self._kernel = TorchSerialChain(inputs)

    @classmethod
    def from_frozen(cls, root: Path = ROOT):
        return cls(root)

    def __call__(self, q, *, robot_id, joint_names, units='rad'):
        if robot_id != self.robot_id or tuple(joint_names) != self.joint_names or units != 'rad':
            raise ValueError('robot identity, ordered joint names or units mismatch')
        return self._kernel(q)
