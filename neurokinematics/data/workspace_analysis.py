"""
NeuroKinematics — Workspace Analysis and Voxelization

Provides:
  - 3D workspace voxel grid construction
  - Spatial coverage analysis (position + orientation)
  - Reachability mapping
  - Spatial group split for train/val/test
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VoxelGrid:
    """3D voxel grid over the robot workspace."""
    bounds_min: np.ndarray   # (3,) workspace lower bounds
    bounds_max: np.ndarray   # (3,) workspace upper bounds
    resolution: np.ndarray   # (3,) number of voxels per axis
    voxel_size: np.ndarray   # (3,) size of each voxel

    def position_to_voxel(self, positions: np.ndarray) -> np.ndarray:
        """
        Map Cartesian positions to voxel indices.

        Args:
            positions: (N, 3) array of xyz positions.

        Returns:
            (N, 3) array of integer voxel indices, clipped to valid range.
        """
        normalized = (positions - self.bounds_min) / self.voxel_size
        indices = np.floor(normalized).astype(int)
        # Clip to valid range
        for d in range(3):
            indices[:, d] = np.clip(indices[:, d], 0, int(self.resolution[d]) - 1)
        return indices

    def voxel_to_flat_index(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Convert 3D voxel indices to flat 1D indices."""
        rx, ry, rz = int(self.resolution[0]), int(self.resolution[1]), int(self.resolution[2])
        return (voxel_indices[:, 0] * ry * rz +
                voxel_indices[:, 1] * rz +
                voxel_indices[:, 2])

    @property
    def total_voxels(self) -> int:
        return int(np.prod(self.resolution))


def estimate_workspace_bounds(
    fk_instance,
    q_min: np.ndarray,
    q_max: np.ndarray,
    n_samples: int = 50000,
    seed: int = 0,
    padding: float = 0.05,
    voxel_resolution: Tuple[int, int, int] = (10, 10, 10),
) -> Tuple[np.ndarray, np.ndarray, Optional[set]]:
    """
    Estimate the reachable workspace bounds and reachable voxels.

    Args:
        fk_instance: ForwardKinematics instance.
        q_min: Lower joint limits.
        q_max: Upper joint limits.
        n_samples: Number of random configs to sample.
        seed: Random seed.
        padding: Padding fraction beyond observed min/max.
        voxel_resolution: Resolution for reachability map.

    Returns:
        (bounds_min, bounds_max, reachable_voxels_set).
    """
    rng = np.random.default_rng(seed)
    n_joints = len(q_min)
    configs = rng.uniform(q_min, q_max, size=(n_samples, n_joints))

    positions = np.array([fk_instance.position(q) for q in configs])

    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)

    # Add padding
    extent = pos_max - pos_min
    pos_min -= padding * extent
    pos_max += padding * extent

    # Build reachability map: which voxels are actually reachable
    grid = create_voxel_grid(pos_min, pos_max, voxel_resolution)
    voxel_idx = grid.position_to_voxel(positions)
    flat_idx = grid.voxel_to_flat_index(voxel_idx)
    reachable_voxels = set(flat_idx.tolist())

    return pos_min, pos_max, reachable_voxels


def create_voxel_grid(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    resolution: Tuple[int, int, int] = (10, 10, 10),
) -> VoxelGrid:
    """
    Create a 3D voxel grid over the workspace.

    Args:
        bounds_min: (3,) lower bounds.
        bounds_max: (3,) upper bounds.
        resolution: Number of voxels per axis (nx, ny, nz).

    Returns:
        VoxelGrid instance.
    """
    res = np.array(resolution, dtype=float)
    voxel_size = (bounds_max - bounds_min) / res
    return VoxelGrid(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        resolution=res,
        voxel_size=voxel_size,
    )


def compute_workspace_coverage(
    grid: VoxelGrid,
    positions: np.ndarray,
    reachable_voxels: Optional[set] = None,
) -> Dict:
    """
    Compute workspace coverage statistics.

    Coverage is measured against reachable voxels (not total bounding box).

    Args:
        grid: VoxelGrid instance.
        positions: (N, 3) positions.
        reachable_voxels: Set of flat voxel indices known to be reachable.
            If None, uses all voxels as denominator.

    Returns:
        Dict with coverage metrics.
    """
    voxel_idx = grid.position_to_voxel(positions)
    flat_idx = grid.voxel_to_flat_index(voxel_idx)

    occupied = set(flat_idx.tolist())
    total = grid.total_voxels

    if reachable_voxels is not None:
        n_reachable = len(reachable_voxels)
        covered_reachable = len(occupied & reachable_voxels)
        coverage = covered_reachable / max(n_reachable, 1)
    else:
        n_reachable = total
        coverage = len(occupied) / total

    # Samples per voxel distribution
    unique, counts = np.unique(flat_idx, return_counts=True)
    samples_per_voxel = np.zeros(total, dtype=int)
    samples_per_voxel[unique] = counts

    return {
        'total_voxels': total,
        'reachable_voxels': n_reachable,
        'occupied_voxels': len(occupied),
        'coverage_ratio': coverage,
        'min_samples_per_occupied': int(counts.min()) if len(counts) > 0 else 0,
        'max_samples_per_occupied': int(counts.max()) if len(counts) > 0 else 0,
        'mean_samples_per_occupied': float(counts.mean()) if len(counts) > 0 else 0,
        'samples_per_voxel': samples_per_voxel,
    }


def compute_orientation_coverage(
    grid: VoxelGrid,
    positions: np.ndarray,
    orientations_6d: np.ndarray,
    min_orientations_per_voxel: int = 3,
) -> Dict:
    """
    Check orientation diversity within each occupied voxel.

    For each voxel, counts how many distinct orientation clusters exist.
    Uses angular distance between the first column of the rotation matrix.

    Args:
        grid: VoxelGrid instance.
        positions: (N, 3) positions.
        orientations_6d: (N, 6) 6D rotation representations.
        min_orientations_per_voxel: Minimum distinct orientations required.

    Returns:
        Dict with orientation coverage metrics.
    """
    voxel_idx = grid.position_to_voxel(positions)
    flat_idx = grid.voxel_to_flat_index(voxel_idx)

    # Group orientations by voxel
    voxel_orientations = {}
    for i, v_idx in enumerate(flat_idx):
        v_idx = int(v_idx)
        if v_idx not in voxel_orientations:
            voxel_orientations[v_idx] = []
        voxel_orientations[v_idx].append(orientations_6d[i])

    # Check diversity: compute angular spread of first rotation column
    sufficient_count = 0
    total_occupied = len(voxel_orientations)

    for v_idx, orients in voxel_orientations.items():
        if len(orients) < min_orientations_per_voxel:
            continue

        orients_arr = np.array(orients)
        # Use first 3 components (first column of R) for angular diversity
        r1_cols = orients_arr[:, :3]
        # Normalize
        norms = np.linalg.norm(r1_cols, axis=1, keepdims=True)
        r1_cols = r1_cols / (norms + 1e-12)

        # Compute pairwise angular spread using dot products
        mean_dir = r1_cols.mean(axis=0)
        mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)
        dots = np.clip(r1_cols @ mean_dir, -1, 1)
        angles = np.arccos(dots)
        angular_spread = angles.std()

        # If there's meaningful angular spread, count as sufficient
        if angular_spread > 0.1:  # ~5.7 degrees
            sufficient_count += 1

    return {
        'total_occupied_voxels': total_occupied,
        'voxels_with_sufficient_orientation': sufficient_count,
        'orientation_coverage_ratio': sufficient_count / max(total_occupied, 1),
        'min_orientations_per_voxel': min_orientations_per_voxel,
    }


def spatial_group_split(
    grid: VoxelGrid,
    positions: np.ndarray,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    n_macro: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test using spatial group assignment.

    Divides the workspace into contiguous macro-blocks and assigns
    each block to train, val, or test. Uses sample-count-aware
    assignment to balance actual ratios.

    Algorithm:
      1. Create a macro-grid (n_macro^3 blocks)
      2. Count samples per macro-block
      3. Greedily assign blocks to val/test first (target ratio),
         then remaining to train
      4. Each sample inherits the assignment of its macro-block

    Args:
        grid: VoxelGrid instance.
        positions: (N, 3) array of sample positions.
        train_ratio: Target fraction for training.
        val_ratio: Target fraction for validation.
        test_ratio: Target fraction for testing.
        seed: Random seed.
        n_macro: Macro-grid divisions per axis (default 4 -> 64 blocks).

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    rng = np.random.default_rng(seed)
    N = len(positions)

    macro_size = (grid.bounds_max - grid.bounds_min) / n_macro
    n_total_macros = n_macro ** 3

    # Assign each sample to a macro-block
    macro_idx = np.floor((positions - grid.bounds_min) / macro_size).astype(int)
    macro_idx = np.clip(macro_idx, 0, n_macro - 1)
    flat_macro = (macro_idx[:, 0] * n_macro * n_macro +
                  macro_idx[:, 1] * n_macro +
                  macro_idx[:, 2])

    # Count samples per macro-block
    block_counts = {}
    for m in flat_macro:
        m = int(m)
        block_counts[m] = block_counts.get(m, 0) + 1

    # Get occupied blocks, shuffle them
    occupied_blocks = list(block_counts.keys())
    rng.shuffle(occupied_blocks)

    # Greedy assignment: fill val and test to target, rest goes to train
    target_val = int(np.round(N * val_ratio))
    target_test = int(np.round(N * test_ratio))

    val_blocks = set()
    test_blocks = set()
    train_blocks = set()
    val_count = 0
    test_count = 0

    for block_id in occupied_blocks:
        bc = block_counts[block_id]
        if val_count < target_val:
            val_blocks.add(block_id)
            val_count += bc
        elif test_count < target_test:
            test_blocks.add(block_id)
            test_count += bc
        else:
            train_blocks.add(block_id)

    # Assign samples
    train_mask = np.array([int(m) in train_blocks for m in flat_macro])
    val_mask = np.array([int(m) in val_blocks for m in flat_macro])
    test_mask = np.array([int(m) in test_blocks for m in flat_macro])

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    return train_idx, val_idx, test_idx
