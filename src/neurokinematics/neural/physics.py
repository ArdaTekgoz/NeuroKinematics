"""Dimensionless losses and separately controlled output heads for C1-05."""
import torch
from neurokinematics.kinematics.model import load_robot
from .training_fk import TrainingFK, DOMAIN


def quaternion_matrix(quaternion):
    if quaternion.shape[-1] != 4 or not torch.isfinite(quaternion).all():
        raise ValueError('finite wxyz required')
    if torch.any(torch.abs(torch.linalg.vector_norm(quaternion, dim=-1) - 1) > 1e-6):
        raise ValueError('unit quaternion required')
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z), -1).reshape(*quaternion.shape[:-1], 3, 3)


def normalized_head(logits, variant):
    if variant not in ('Q', 'FK', 'FK_LIMIT', 'FK_TANH'):
        raise ValueError('unknown variant')
    return (torch.tanh(logits) + 1) / 2 if variant == 'FK_TANH' else logits


def limit_loss(q, lower, upper):
    span = upper - lower
    return ((torch.relu(lower - q) / span)**2 + (torch.relu(q - upper) / span)**2).sum(-1)


class PhysicsLoss:
    def __init__(self):
        inputs = load_robot()
        self.lower = torch.tensor([v[0] for v in inputs.limits], dtype=torch.float64)
        self.upper = torch.tensor([v[1] for v in inputs.limits], dtype=torch.float64)
        self.fk = TrainingFK.from_frozen(domain=DOMAIN)

    def raw(self, normalized):
        lower = self.lower.to(normalized)
        span = (self.upper - self.lower).to(normalized)
        return lower + normalized * span

    def components(self, normalized, target_normalized, position, rotation):
        # Use the same normalized float32 subtraction as C1-04 for exact control.
        lq = ((normalized - target_normalized)**2).sum(-1)
        q = self.raw(normalized)
        transform = self.fk(q, robot_id=self.fk.robot_id, joint_names=self.fk.joint_names)
        lp = (((transform[..., :3, 3] - position) / 0.9015)**2).sum(-1)
        lr = ((transform[..., :3, :3] - rotation)**2).sum((-1, -2)) / 8
        ll = limit_loss(q, self.lower.to(q), self.upper.to(q))
        return dict(q=lq, p=lp, R=lr, lim=ll), q, transform


def combined(terms, arm):
    # Omit zero-weight terms from backward so Q is bitwise C1-04 supervised loss.
    value = terms['q']
    for term, key in [('p', 'lambda_p'), ('R', 'lambda_R'), ('lim', 'lambda_lim')]:
        if arm[key] != 0:
            value = value + arm[key] * terms[term]
    return value.mean()
