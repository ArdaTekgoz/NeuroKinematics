"""Opt-in ideal revolute FK extension. This is NOT a joint-validity API (ADR-013).

The small Torch numeric body is intentionally separate from the byte-frozen
C1-03 kernel. Independent forward/FD tests cover both interior and exterior q.
"""
from pathlib import Path
import torch
from neurokinematics.kinematics.chain import extract_chain
from neurokinematics.kinematics.model import ROOT, RobotInputs, load_robot

DOMAIN = 'finite_revolute_extension'


class TrainingSerialChain:
    def __init__(self, inputs: RobotInputs):
        self.joint_names = tuple(inputs.joint_names)
        self.robot_id = inputs.robot_id
        self.chain = extract_chain(inputs.urdf, inputs.base, inputs.tcp, self.joint_names)
        self._steps = []
        for joint in self.chain:
            index = self.joint_names.index(joint.name) if joint.kind == 'revolute' else None
            if index is not None and joint.limits != inputs.limits[index]:
                raise ValueError('URDF/RobotSpec limits differ')
            origin = torch.tensor(joint.origin.copy(), dtype=torch.float64)
            skew = None
            if index is not None:
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
                transform = transform @ torch.cat((upper, lower), dim=-2)
        return transform[0] if q.ndim == 1 else transform


class TrainingFK:
    def __init__(self, root: Path = ROOT, *, domain):
        if domain != DOMAIN:
            raise ValueError('explicit finite_revolute_extension opt-in required')
        inputs = load_robot(Path(root))
        self.robot_id, self.joint_names = inputs.robot_id, inputs.joint_names
        self.hashes = dict(inputs.hashes)
        self._kernel = TrainingSerialChain(inputs)

    @classmethod
    def from_frozen(cls, *, domain, root=ROOT):
        return cls(root, domain=domain)

    def __call__(self, q, *, robot_id, joint_names, units='rad'):
        if robot_id != self.robot_id or tuple(joint_names) != self.joint_names or units != 'rad':
            raise ValueError('robot identity, ordered joint names or units mismatch')
        return self._kernel(q)
