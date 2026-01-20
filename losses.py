import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Utility
# ============================================================

def clamp_tensor(x, min_val, max_val):
    return torch.clamp(x, min=min_val, max=max_val)


# ============================================================
# Joint Limit Loss
# ============================================================

def joint_limit_loss(theta, joint_limits):
    """
    theta: [B, DOF]
    joint_limits: [(min, max), ...] length = DOF
    """
    loss = 0.0
    for i, (jmin, jmax) in enumerate(joint_limits):
        loss += torch.mean(
            F.relu(jmin - theta[:, i]) ** 2 +
            F.relu(theta[:, i] - jmax) ** 2
        )
    return loss


# ============================================================
# Smoothness Loss
# ============================================================

def smoothness_loss(theta):
    """
    Penalizes large joint values (acts as weak regularizer)
    """
    return torch.mean(theta ** 2)


# ============================================================
# Forward Kinematics Loss (Hard-Sample Aware)
# ============================================================

def fk_loss_hard_sample(
    theta_pred,
    fk_func,
    target_pos,
    alpha=2.0,
    max_weight=5.0
):
    """
    Hard-sample aware FK loss

    theta_pred : [B, DOF]
    fk_func    : function(theta) -> [B, 3]
    target_pos : [B, 3]
    """

    # Predicted end-effector position
    fk_pos_pred = fk_func(theta_pred)          # [B, 3]

    # Sample-wise FK error
    fk_error = torch.norm(fk_pos_pred - target_pos, dim=1)  # [B]

    # Mean FK error (detach for stability)
    mean_fk = fk_error.mean().detach()

    # Hard-sample weighting
    weights = 1.0 + alpha * (fk_error / (mean_fk + 1e-6))
    weights = torch.clamp(weights, max=max_weight)

    # Weighted FK loss
    fk_loss = torch.mean(weights * fk_error)

    # Metrics for logging
    fk_mean = fk_error.mean().item()
    fk_max = fk_error.max().item()

    return fk_loss, fk_mean, fk_max


# ============================================================
# TOTAL LOSS (A-4.3.1)
# ============================================================

def total_loss(
    theta_pred,
    theta_gt,
    target_pos,
    fk_func,
    joint_limits,
    weights_cfg
):
    """
    theta_pred : [B, DOF]
    theta_gt   : [B, DOF]
    target_pos : [B, 3]
    fk_func    : FK function
    joint_limits : list of (min, max)
    weights_cfg : dict with loss weights
    """

    # ------------------
    # MSE (Joint space)
    # ------------------
    mse = F.mse_loss(theta_pred, theta_gt)

    # ------------------
    # FK Hard-Sample Loss
    # ------------------
    fk_loss, fk_mean, fk_max = fk_loss_hard_sample(
        theta_pred=theta_pred,
        fk_func=fk_func,
        target_pos=target_pos,
        alpha=weights_cfg.get("fk_alpha", 2.0)
    )

    # ------------------
    # Joint Limit Loss
    # ------------------
    jl = joint_limit_loss(theta_pred, joint_limits)

    # ------------------
    # Smoothness Loss
    # ------------------
    smooth = smoothness_loss(theta_pred)

    # ------------------
    # Total
    # ------------------
    total = (
        weights_cfg["mse"] * mse +
        weights_cfg["fk"] * fk_loss +
        weights_cfg["joint"] * jl +
        weights_cfg["smooth"] * smooth
    )

    metrics = {
        "total": total,
        "mse": mse.item(),
        "fk_mean": fk_mean,
        "fk_max": fk_max,
        "joint": jl.item(),
        "smooth": smooth.item()
    }

    return total, metrics
