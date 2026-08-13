"""
NeuroKinematics — Jacobian, Singularity Analysis and Manipulability

Implements:
  - Geometric Jacobian computation
  - Jacobian rank analysis
  - Singularity detection
  - Yoshikawa manipulability index
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from neurokinematics.core.robot_model import RobotModel
from neurokinematics.core.forward_kinematics import ForwardKinematics


# ==============================================================================
# Geometric Jacobian (NumPy)
# ==============================================================================

def compute_geometric_jacobian(fk: ForwardKinematics, q: np.ndarray) -> np.ndarray:
    """
    Computes the 6 x n geometric Jacobian matrix.

    The Jacobian maps joint velocities to end-effector spatial velocity:
        v = J(q) * dq
    where v = [linear_vel (3); angular_vel (3)]

    For revolute joints:
        J_i = [z_i x (p_e - p_i)]   (linear part)
              [z_i                ]   (angular part)

    Uses pre-rotation frames: z_i is computed from the frame that exists
    AFTER the static transform but BEFORE joint i's own rotation.

    Args:
        fk: ForwardKinematics instance.
        q: Joint angles, shape (n_joints,).

    Returns:
        Jacobian matrix, shape (6, n_joints).
    """
    from neurokinematics.core.forward_kinematics import (
        _axis_angle_to_rotation_np, _make_transform_np
    )

    n = fk.model.n_joints
    T_ee = fk.compute(q)
    p_ee = T_ee[:3, 3]

    # Build pre-rotation transforms: the frame AFTER static transform
    # but BEFORE the joint's own rotation is applied.
    T = np.eye(4)
    pre_rotation_transforms = []
    for i, joint in enumerate(fk.model.chain_joints):
        T = T @ fk._static_transforms[i]  # Apply static transform
        pre_rotation_transforms.append(T.copy())  # Save BEFORE joint rotation
        R_joint = _axis_angle_to_rotation_np(joint.axis, q[i])
        T_joint = _make_transform_np(R_joint, np.zeros(3))
        T = T @ T_joint  # Apply joint rotation

    J = np.zeros((6, n))
    for i in range(n):
        T_pre = pre_rotation_transforms[i]
        joint = fk.model.chain_joints[i]

        R_i = T_pre[:3, :3]
        z_i = R_i @ joint.axis   # Joint axis in world frame
        p_i = T_pre[:3, 3]       # Joint origin in world frame

        if joint.joint_type in ('revolute', 'continuous'):
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        elif joint.joint_type == 'prismatic':
            J[:3, i] = z_i
            J[3:, i] = 0.0

    return J


# ==============================================================================
# Jacobian Analysis
# ==============================================================================

def jacobian_rank(J: np.ndarray) -> int:
    """
    Compute the numerical rank of the Jacobian matrix.

    Args:
        J: Jacobian matrix, shape (6, n_joints).

    Returns:
        Integer rank of the Jacobian.
    """
    return int(np.linalg.matrix_rank(J))


def jacobian_condition_number(J: np.ndarray) -> float:
    """
    Compute the condition number of the Jacobian.
    High condition number indicates proximity to singularity.

    Args:
        J: Jacobian matrix, shape (6, n_joints).

    Returns:
        Condition number (float). Returns inf at exact singularity.
    """
    singular_values = np.linalg.svd(J, compute_uv=False)
    if singular_values[-1] < 1e-10:
        return float('inf')
    return float(singular_values[0] / singular_values[-1])


def is_singular(J: np.ndarray, threshold: float = 1e-4) -> bool:
    """
    Detect if the robot configuration is near a singularity.

    Uses the Yoshikawa manipulability measure: if w < threshold,
    the configuration is considered singular.

    Args:
        J: Jacobian matrix, shape (6, n_joints).
        threshold: Manipulability threshold for singularity detection.

    Returns:
        True if configuration is near singular.
    """
    w = yoshikawa_manipulability(J)
    return w < threshold


def yoshikawa_manipulability(J: np.ndarray) -> float:
    """
    Compute the Yoshikawa manipulability index.

    w = sqrt(det(J * J^T))

    This measures how far the robot is from a singular configuration.
    w = 0 at singularity, higher values indicate better manipulability.

    Reference:
        T. Yoshikawa, "Manipulability of Robotic Mechanisms",
        Int. J. Robotics Research, 1985.

    Args:
        J: Jacobian matrix, shape (6, n_joints).

    Returns:
        Manipulability index (non-negative float).
    """
    JJT = J @ J.T
    det_val = np.linalg.det(JJT)
    # Clamp to avoid sqrt of small negative due to numerical errors
    return float(np.sqrt(max(det_val, 0.0)))


# ==============================================================================
# Singular Value Decomposition Analysis
# ==============================================================================

def singular_values(J: np.ndarray) -> np.ndarray:
    """
    Compute singular values of the Jacobian.
    Useful for detailed singularity analysis.

    Args:
        J: Jacobian matrix, shape (6, n_joints).

    Returns:
        Array of singular values in descending order.
    """
    return np.linalg.svd(J, compute_uv=False)


# ==============================================================================
# PyTorch Differentiable Jacobian (for training)
# ==============================================================================

if HAS_TORCH:
    def compute_jacobian_torch(diff_fk, q):
        """
        Compute the Jacobian using PyTorch autograd (differentiable).

        Args:
            diff_fk: DifferentiableFK instance.
            q: Joint angles tensor, shape (n_joints,), requires_grad=True.

        Returns:
            Jacobian tensor, shape (6, n_joints).
        """
        import torch
        q_grad = q.detach().requires_grad_(True)
        pos, rot6d = diff_fk.compute(q_grad)
        output = torch.cat([pos, rot6d[:3]])
        J = torch.zeros(6, q_grad.shape[0], device=q.device, dtype=q.dtype)
        for i in range(6):
            grad = torch.autograd.grad(output[i], q_grad, retain_graph=True, create_graph=False)[0]
            J[i] = grad
        return J

    def yoshikawa_manipulability_torch(J):
        """Differentiable Yoshikawa manipulability index (PyTorch)."""
        import torch
        JJT = J @ J.T
        det_val = torch.det(JJT)
        return torch.sqrt(torch.clamp(det_val, min=0.0))

