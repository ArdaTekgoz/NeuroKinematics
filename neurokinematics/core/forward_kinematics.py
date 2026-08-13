"""
NeuroKinematics — Forward Kinematics Engine

Implements forward kinematics using homogeneous transformation matrices.
Provides both NumPy (for validation) and PyTorch (differentiable, for training) versions.

Supports:
  - Full 4x4 homogeneous transform chain
  - End-effector position extraction
  - Orientation as rotation matrix, quaternion, and 6D continuous representation
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch as _torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False

from neurokinematics.core.robot_model import RobotModel, JointInfo


# ==============================================================================
# NumPy Utility Functions
# ==============================================================================

def _rpy_to_rotation_np(rpy: np.ndarray) -> np.ndarray:
    """Roll-Pitch-Yaw (XYZ intrinsic) to 3x3 rotation matrix (NumPy)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr               ],
    ])
    return R


def _axis_angle_to_rotation_np(axis: np.ndarray, angle: float) -> np.ndarray:
    """Axis-angle to 3x3 rotation matrix using Rodrigues' formula (NumPy)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _make_transform_np(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct 4x4 homogeneous transform from 3x3 rotation and 3D translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rotation_matrix_to_quaternion_np(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z] (NumPy)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _rotation_matrix_to_6d_np(R: np.ndarray) -> np.ndarray:
    """Extract 6D continuous rotation representation (first two columns of R)."""
    return np.concatenate([R[:, 0], R[:, 1]])


# ==============================================================================
# Forward Kinematics Class (NumPy)
# ==============================================================================

class ForwardKinematics:
    """
    Computes forward kinematics for a given RobotModel using
    homogeneous transformation matrices (NumPy).

    Usage:
        fk = ForwardKinematics(model)
        T = fk.compute(q)          # 4x4 end-effector transform
        pos = fk.position(q)       # (3,) xyz
        quat = fk.quaternion(q)    # (4,) wxyz
        rot6d = fk.rotation_6d(q)  # (6,)
    """

    def __init__(self, robot_model: RobotModel):
        self.model = robot_model
        self._static_transforms = []
        for j in robot_model.chain_joints:
            R_static = _rpy_to_rotation_np(j.origin_rpy)
            T_static = _make_transform_np(R_static, j.origin_xyz)
            self._static_transforms.append(T_static)

        self._tcp_transform = np.eye(4)
        for j in robot_model.all_joints:
            if j.joint_type == 'fixed' and j.parent_link == robot_model.chain_joints[-1].child_link:
                R_tcp = _rpy_to_rotation_np(j.origin_rpy)
                self._tcp_transform = _make_transform_np(R_tcp, j.origin_xyz)
                break

    def compute(self, q: np.ndarray) -> np.ndarray:
        """Compute the full 4x4 end-effector homogeneous transform."""
        assert len(q) == self.model.n_joints, \
            f"Expected {self.model.n_joints} joint values, got {len(q)}"
        T = np.eye(4)
        for i, joint in enumerate(self.model.chain_joints):
            T = T @ self._static_transforms[i]
            R_joint = _axis_angle_to_rotation_np(joint.axis, q[i])
            T_joint = _make_transform_np(R_joint, np.zeros(3))
            T = T @ T_joint
        T = T @ self._tcp_transform
        return T

    def compute_all_transforms(self, q: np.ndarray) -> List[np.ndarray]:
        """Compute transforms for every joint frame (useful for Jacobian)."""
        transforms = [np.eye(4)]
        T = np.eye(4)
        for i, joint in enumerate(self.model.chain_joints):
            T = T @ self._static_transforms[i]
            R_joint = _axis_angle_to_rotation_np(joint.axis, q[i])
            T_joint = _make_transform_np(R_joint, np.zeros(3))
            T = T @ T_joint
            transforms.append(T.copy())
        return transforms

    def position(self, q: np.ndarray) -> np.ndarray:
        """End-effector position (x, y, z)."""
        return self.compute(q)[:3, 3]

    def orientation_matrix(self, q: np.ndarray) -> np.ndarray:
        """End-effector orientation as 3x3 rotation matrix."""
        return self.compute(q)[:3, :3]

    def quaternion(self, q: np.ndarray) -> np.ndarray:
        """End-effector orientation as quaternion [w, x, y, z]."""
        return _rotation_matrix_to_quaternion_np(self.orientation_matrix(q))

    def rotation_6d(self, q: np.ndarray) -> np.ndarray:
        """End-effector orientation as 6D continuous rotation representation."""
        return _rotation_matrix_to_6d_np(self.orientation_matrix(q))


# ==============================================================================
# Differentiable Forward Kinematics (PyTorch) — Only loaded when torch is available
# ==============================================================================

if HAS_TORCH:
    def _rpy_to_rotation_torch(rpy):
        """Roll-Pitch-Yaw to 3x3 rotation matrix (PyTorch, differentiable)."""
        r, p, y = rpy[0], rpy[1], rpy[2]
        cr, sr = _torch.cos(r), _torch.sin(r)
        cp, sp = _torch.cos(p), _torch.sin(p)
        cy, sy = _torch.cos(y), _torch.sin(y)
        R = _torch.stack([
            _torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            _torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            _torch.stack([-sp,     cp * sr,                cp * cr]),
        ])
        return R

    def _axis_angle_to_rotation_torch(axis, angle):
        """Axis-angle to 3x3 rotation matrix using Rodrigues' formula (PyTorch)."""
        axis = axis / (_torch.norm(axis) + 1e-12)
        zero = _torch.zeros(1, device=axis.device, dtype=axis.dtype).squeeze()
        K = _torch.stack([
            _torch.stack([zero, -axis[2], axis[1]]),
            _torch.stack([axis[2], zero, -axis[0]]),
            _torch.stack([-axis[1], axis[0], zero]),
        ])
        R = _torch.eye(3, device=axis.device, dtype=axis.dtype) + \
            _torch.sin(angle) * K + (1.0 - _torch.cos(angle)) * (K @ K)
        return R

    def _make_transform_torch(R, t):
        """Construct 4x4 homogeneous transform (PyTorch)."""
        T = _torch.eye(4, device=R.device, dtype=R.dtype)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    class DifferentiableFK:
        """
        Differentiable forward kinematics using PyTorch for backpropagation
        in the physics-aware training loop.
        """

        def __init__(self, robot_model: RobotModel, device='cpu', dtype=None):
            if dtype is None:
                dtype = _torch.float32
            self.model = robot_model
            self.device = device
            self.dtype = dtype

            self._static_transforms = []
            for j in robot_model.chain_joints:
                rpy = _torch.tensor(j.origin_rpy, device=device, dtype=dtype)
                xyz = _torch.tensor(j.origin_xyz, device=device, dtype=dtype)
                R = _rpy_to_rotation_torch(rpy)
                T = _make_transform_torch(R, xyz)
                self._static_transforms.append(T)

            self._axes = [_torch.tensor(j.axis, device=device, dtype=dtype)
                          for j in robot_model.chain_joints]

            tcp_np = np.eye(4)
            for j in robot_model.all_joints:
                if j.joint_type == 'fixed' and j.parent_link == robot_model.chain_joints[-1].child_link:
                    R_tcp = _rpy_to_rotation_np(j.origin_rpy)
                    tcp_np = _make_transform_np(R_tcp, j.origin_xyz)
                    break
            self._tcp_transform = _torch.tensor(tcp_np, device=device, dtype=dtype)

        def compute(self, q):
            """Differentiable FK: returns (position [3], rotation_6d [6])."""
            T = _torch.eye(4, device=self.device, dtype=self.dtype)
            for i in range(self.model.n_joints):
                T = T @ self._static_transforms[i]
                R_joint = _axis_angle_to_rotation_torch(self._axes[i], q[i])
                T_joint = _make_transform_torch(R_joint, _torch.zeros(3, device=self.device, dtype=self.dtype))
                T = T @ T_joint
            T = T @ self._tcp_transform
            pos = T[:3, 3]
            rot_6d = _torch.cat([T[:3, 0], T[:3, 1]])
            return pos, rot_6d

        def compute_batch(self, q_batch):
            """Batch differentiable FK for multiple configurations."""
            B = q_batch.shape[0]
            positions = _torch.zeros(B, 3, device=self.device, dtype=self.dtype)
            rotations = _torch.zeros(B, 6, device=self.device, dtype=self.dtype)
            for b in range(B):
                pos, rot = self.compute(q_batch[b])
                positions[b] = pos
                rotations[b] = rot
            return positions, rotations
else:
    # Placeholder when torch is not available
    DifferentiableFK = None
