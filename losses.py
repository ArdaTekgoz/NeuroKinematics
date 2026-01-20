import torch


def fk_error_weights(
    fk_error: torch.Tensor,
    alpha: float = 2.0,
    w_min: float = 0.5,
    w_max: float = 3.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute per-sample weights based on FK position error.

    Args:
        fk_error (Tensor): Shape (B,) FK L2 position errors (meters)
        alpha (float): Weight scaling factor
        w_min (float): Minimum clamp value
        w_max (float): Maximum clamp value
        eps (float): Numerical stability

    Returns:
        Tensor: Shape (B,) sample weights
    """

    # Batch mean FK error
    mean_error = fk_error.mean().detach() + eps

    # Relative error
    relative_error = fk_error / mean_error

    # Weight computation
    weights = 1.0 + alpha * relative_error

    # Clamp for stability
    weights = torch.clamp(weights, min=w_min, max=w_max)

    return weights
