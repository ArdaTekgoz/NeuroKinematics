"""Versioned endpoint-exact joint decoding; historical PhysicsLoss is frozen."""
import torch
from neurokinematics.kinematics.model import load_robot


def decode_joints(normalized):
    """Preserve gradients and real out-of-range values, with no clipping."""
    if not isinstance(normalized, torch.Tensor) or normalized.ndim not in (1, 2) or normalized.shape[-1] != 6:
        raise ValueError("six normalized joint values required")
    if normalized.dtype not in (torch.float32, torch.float64):
        raise ValueError("float32 or float64 required")
    z = normalized.to(torch.float64)
    bounds = torch.tensor(load_robot().limits, dtype=torch.float64, device=z.device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    span = upper-lower
    return torch.where(z < .5, lower+z*span, upper-(1-z)*span)
