"""
NeuroKinematics — Data Split and Leakage Detection

Implements:
  - Normalized joint-space leakage detection
  - Cartesian pose-aware leakage detection (position + orientation)
  - Split validation utilities
"""

import numpy as np
from typing import Dict, Tuple, Optional


def normalize_joint_space(
    q: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> np.ndarray:
    """
    Normalize joint configurations to [0, 1] range.

    q_tilde_i = (q_i - q_min_i) / (q_max_i - q_min_i)

    Args:
        q: Joint configurations, shape (N, n_joints).
        q_min: Lower limits, shape (n_joints,).
        q_max: Upper limits, shape (n_joints,).

    Returns:
        Normalized configurations, shape (N, n_joints).
    """
    q_range = q_max - q_min
    q_range = np.where(q_range < 1e-12, 1.0, q_range)  # Avoid division by zero
    return (q - q_min) / q_range


def check_joint_space_leakage(
    q_set_a: np.ndarray,
    q_set_b: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    threshold: float = 0.01,
    max_check: int = 50000,
) -> Dict:
    """
    Check for data leakage between two sets in normalized joint-space.

    Uses brute-force nearest-neighbor on a subsample for efficiency.

    Args:
        q_set_a: First set of joint configs, shape (Na, n_joints).
        q_set_b: Second set of joint configs, shape (Nb, n_joints).
        q_min: Lower limits.
        q_max: Upper limits.
        threshold: Distance threshold for "too close".
        max_check: Maximum number of pairs to check.

    Returns:
        Dict with leakage statistics.
    """
    q_a_norm = normalize_joint_space(q_set_a, q_min, q_max)
    q_b_norm = normalize_joint_space(q_set_b, q_min, q_max)

    # Subsample for efficiency
    n_a = min(len(q_a_norm), max_check)
    n_b = min(len(q_b_norm), max_check)

    if n_a < len(q_a_norm):
        idx_a = np.random.default_rng(0).choice(len(q_a_norm), n_a, replace=False)
        q_a_sub = q_a_norm[idx_a]
    else:
        q_a_sub = q_a_norm

    if n_b < len(q_b_norm):
        idx_b = np.random.default_rng(1).choice(len(q_b_norm), n_b, replace=False)
        q_b_sub = q_b_norm[idx_b]
    else:
        q_b_sub = q_b_norm

    # Find minimum distances from A to B using batched computation
    min_distances = []
    batch_size = 1000
    for i in range(0, len(q_a_sub), batch_size):
        batch = q_a_sub[i:i + batch_size]
        # (batch, 1, joints) - (1, n_b, joints)
        diffs = batch[:, np.newaxis, :] - q_b_sub[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        min_distances.extend(min_dists.tolist())

    min_distances = np.array(min_distances)
    leaking = int((min_distances < threshold).sum())

    return {
        'n_checked_a': n_a,
        'n_checked_b': n_b,
        'threshold': threshold,
        'leaking_count': leaking,
        'leaking_ratio': leaking / max(n_a, 1),
        'min_distance': float(min_distances.min()) if len(min_distances) > 0 else float('inf'),
        'mean_min_distance': float(min_distances.mean()) if len(min_distances) > 0 else 0,
        'median_min_distance': float(np.median(min_distances)) if len(min_distances) > 0 else 0,
    }


def check_pose_leakage(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    rotations_a: np.ndarray,
    rotations_b: np.ndarray,
    pos_threshold: float = 0.001,  # 1mm
    rot_threshold: float = 0.01,   # ~0.57 degrees
    sigma_p: float = 0.001,
    sigma_r: float = 0.01,
    max_check: int = 10000,
) -> Dict:
    """
    Check for leakage using combined position + orientation distance.

    d_pose = sqrt((d_p/sigma_p)^2 + (d_R/sigma_R)^2)

    Args:
        positions_a, positions_b: (N, 3) position arrays.
        rotations_a, rotations_b: (N, 6) 6D rotation arrays.
        pos_threshold: Position distance threshold (meters).
        rot_threshold: Rotation distance threshold.
        sigma_p: Position normalization factor.
        sigma_r: Rotation normalization factor.
        max_check: Maximum samples to check.

    Returns:
        Dict with pose leakage statistics.
    """
    n_a = min(len(positions_a), max_check)
    n_b = min(len(positions_b), max_check)

    rng = np.random.default_rng(42)
    idx_a = rng.choice(len(positions_a), n_a, replace=False) if n_a < len(positions_a) else np.arange(n_a)
    idx_b = rng.choice(len(positions_b), n_b, replace=False) if n_b < len(positions_b) else np.arange(n_b)

    pos_a = positions_a[idx_a]
    pos_b = positions_b[idx_b]
    rot_a = rotations_a[idx_a]
    rot_b = rotations_b[idx_b]

    leaking = 0
    min_pose_dist = float('inf')

    batch_size = 500
    for i in range(0, n_a, batch_size):
        batch_pos = pos_a[i:i + batch_size]
        batch_rot = rot_a[i:i + batch_size]

        # Position distances
        pos_diffs = batch_pos[:, np.newaxis, :] - pos_b[np.newaxis, :, :]
        pos_dists = np.linalg.norm(pos_diffs, axis=2)  # (batch, n_b)

        # Rotation distances (L2 on 6D representation)
        rot_diffs = batch_rot[:, np.newaxis, :] - rot_b[np.newaxis, :, :]
        rot_dists = np.linalg.norm(rot_diffs, axis=2)  # (batch, n_b)

        # Combined pose distance
        pose_dists = np.sqrt((pos_dists / sigma_p) ** 2 + (rot_dists / sigma_r) ** 2)

        batch_min = pose_dists.min(axis=1)
        min_pose_dist = min(min_pose_dist, float(batch_min.min()))

        # Count leaking: both position AND rotation are close
        both_close = (pos_dists < pos_threshold) & (rot_dists < rot_threshold)
        leaking += int(both_close.any(axis=1).sum())

    return {
        'n_checked_a': n_a,
        'n_checked_b': n_b,
        'pos_threshold': pos_threshold,
        'rot_threshold': rot_threshold,
        'leaking_count': leaking,
        'leaking_ratio': leaking / max(n_a, 1),
        'min_pose_distance': min_pose_dist,
    }


def validate_split_ratios(
    train_size: int,
    val_size: int,
    test_size: int,
    expected_train: float = 0.70,
    expected_val: float = 0.10,
    expected_test: float = 0.10,
    tolerance: float = 0.15,
) -> Dict:
    """
    Validate that split ratios are within acceptable tolerance.

    Note: Spatial split ratios may deviate from exact targets because
    macro-blocks may have unequal sample counts.

    Returns:
        Dict with actual ratios and pass/fail status.
    """
    total = train_size + val_size + test_size
    actual_train = train_size / total
    actual_val = val_size / total
    actual_test = test_size / total

    return {
        'train_ratio': actual_train,
        'val_ratio': actual_val,
        'test_ratio': actual_test,
        'train_ok': abs(actual_train - expected_train) < tolerance,
        'val_ok': abs(actual_val - expected_val) < tolerance,
        'test_ok': abs(actual_test - expected_test) < tolerance,
    }
