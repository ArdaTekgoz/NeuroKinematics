"""
NeuroKinematics — Dataset Factory

Master pipeline that orchestrates:
  1. Multi-strategy sampling (uniform / boundary / singularity)
  2. FK computation for ground-truth poses
  3. Workspace voxelization and coverage analysis
  4. Spatial group split (train/val/test)
  5. Leakage verification
  6. HDF5 export with structured groups

Fully deterministic given a seed.
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from neurokinematics.core.robot_model import RobotModel
from neurokinematics.core.forward_kinematics import ForwardKinematics
from neurokinematics.core.jacobian import (
    compute_geometric_jacobian,
    yoshikawa_manipulability,
)
from neurokinematics.data.sampling import (
    uniform_joint_sampling,
    workspace_boundary_sampling,
    singularity_enriched_sampling,
    generate_previous_configs,
)
from neurokinematics.data.workspace_analysis import (
    estimate_workspace_bounds,
    create_voxel_grid,
    compute_workspace_coverage,
    compute_orientation_coverage,
    spatial_group_split,
)
from neurokinematics.data.data_split import (
    check_joint_space_leakage,
    check_pose_leakage,
    validate_split_ratios,
)


def compute_fk_batch(
    fk: ForwardKinematics,
    q_batch: np.ndarray,
    progress_interval: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FK for a batch of configurations.

    Returns:
        positions (N, 3), rotations_6d (N, 6), manipulabilities (N,)
    """
    N = len(q_batch)
    positions = np.zeros((N, 3))
    rotations_6d = np.zeros((N, 6))
    manipulabilities = np.zeros(N)

    for i in range(N):
        positions[i] = fk.position(q_batch[i])
        rotations_6d[i] = fk.rotation_6d(q_batch[i])
        J = compute_geometric_jacobian(fk, q_batch[i])
        manipulabilities[i] = yoshikawa_manipulability(J)

        if progress_interval > 0 and (i + 1) % progress_interval == 0:
            print(f"  FK computed: {i + 1}/{N}")

    return positions, rotations_6d, manipulabilities


def generate_dataset(
    urdf_path: str,
    output_path: str,
    n_total: int = 1_000_000,
    uniform_ratio: float = 0.70,
    boundary_ratio: float = 0.20,
    singularity_ratio: float = 0.10,
    seed: int = 42,
    voxel_resolution: Tuple[int, int, int] = (10, 10, 10),
    split_seed: int = 42,
    singularity_threshold: float = 0.01,
    q_prev_noise_std: float = 0.3,
    progress: bool = True,
) -> Dict:
    """
    Generate the complete synthetic IK dataset.

    This is the main entry point for Faz 2.

    Args:
        urdf_path: Path to robot URDF file.
        output_path: Path for output HDF5 file.
        n_total: Total number of samples.
        uniform_ratio: Fraction of uniform samples.
        boundary_ratio: Fraction of boundary samples.
        singularity_ratio: Fraction of singularity-enriched samples.
        seed: Master random seed for reproducibility.
        voxel_resolution: Voxel grid resolution (nx, ny, nz).
        split_seed: Seed for spatial split.
        singularity_threshold: Manipulability threshold.
        q_prev_noise_std: Noise std for q_{t-1} generation.
        progress: Whether to print progress.

    Returns:
        Dict with generation statistics and validation results.
    """
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Load Robot Model
    # -------------------------------------------------------------------------
    if progress:
        print("=" * 60)
        print("NeuroKinematics — Dataset Factory")
        print("=" * 60)

    model = RobotModel.from_urdf(urdf_path)
    fk = ForwardKinematics(model)
    q_min, q_max = model.joint_limits

    if progress:
        print(f"Robot: {model.robot_name} ({model.n_joints} DoF)")
        print(f"Target: {n_total} samples")

    # -------------------------------------------------------------------------
    # 2. Sampling
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    n_uniform = int(n_total * uniform_ratio)
    n_boundary = int(n_total * boundary_ratio)
    n_singularity = n_total - n_uniform - n_boundary  # Remainder

    if progress:
        print(f"\n--- Sampling ---")
        print(f"  Uniform:     {n_uniform}")
        print(f"  Boundary:    {n_boundary}")
        print(f"  Singularity: {n_singularity}")

    # 2a. Uniform sampling
    if progress:
        print("  Generating uniform samples...")
    q_uniform = uniform_joint_sampling(n_uniform, q_min, q_max, rng)

    # 2b. Boundary sampling
    if progress:
        print("  Generating boundary samples...")
    q_boundary = workspace_boundary_sampling(n_boundary, q_min, q_max, rng)

    # 2c. Singularity sampling
    if progress:
        print("  Generating singularity-enriched samples...")
    q_singular = singularity_enriched_sampling(
        n_singularity, q_min, q_max, fk,
        compute_geometric_jacobian, yoshikawa_manipulability,
        rng, singularity_threshold,
    )

    # Combine all samples
    q_all = np.vstack([q_uniform, q_boundary, q_singular])

    # Track sampling method for each sample
    method_labels = np.concatenate([
        np.full(n_uniform, 0, dtype=np.int8),       # 0 = uniform
        np.full(n_boundary, 1, dtype=np.int8),       # 1 = boundary
        np.full(len(q_singular), 2, dtype=np.int8),  # 2 = singularity
    ])

    # Adjust if singularity returned fewer than requested
    actual_total = len(q_all)
    if progress:
        print(f"  Total generated: {actual_total}")

    # -------------------------------------------------------------------------
    # 3. Generate q_{t-1} (previous configurations)
    # -------------------------------------------------------------------------
    if progress:
        print("\n--- Generating q_previous ---")
    q_previous = generate_previous_configs(q_all, q_min, q_max, rng, q_prev_noise_std)

    # -------------------------------------------------------------------------
    # 4. FK Computation
    # -------------------------------------------------------------------------
    if progress:
        print("\n--- Computing FK ---")
    positions, rotations_6d, manipulabilities = compute_fk_batch(
        fk, q_all, progress_interval=100000 if progress else 0
    )

    # Compute sin/cos representations
    sin_q = np.sin(q_all)
    cos_q = np.cos(q_all)

    # Joint limit margin: min distance to nearest limit (normalized)
    q_range = q_max - q_min
    margin_lower = (q_all - q_min) / q_range
    margin_upper = (q_max - q_all) / q_range
    joint_limit_margin = np.minimum(margin_lower, margin_upper).min(axis=1)

    # -------------------------------------------------------------------------
    # 5. Workspace Analysis
    # -------------------------------------------------------------------------
    if progress:
        print("\n--- Workspace Analysis ---")

    bounds_min, bounds_max, reachable_voxels = estimate_workspace_bounds(
        fk, q_min, q_max, seed=seed, voxel_resolution=voxel_resolution,
    )
    grid = create_voxel_grid(bounds_min, bounds_max, voxel_resolution)

    ws_coverage = compute_workspace_coverage(grid, positions, reachable_voxels)
    orient_coverage = compute_orientation_coverage(grid, positions, rotations_6d)

    if progress:
        print(f"  Workspace bounds: [{bounds_min}] to [{bounds_max}]")
        print(f"  Reachable voxels: {ws_coverage['reachable_voxels']}/{ws_coverage['total_voxels']}")
        print(f"  Position coverage: {ws_coverage['coverage_ratio']:.1%} "
              f"({ws_coverage['occupied_voxels']} occupied / {ws_coverage['reachable_voxels']} reachable)")
        print(f"  Orientation coverage: {orient_coverage['orientation_coverage_ratio']:.1%}")

    # -------------------------------------------------------------------------
    # 6. Spatial Group Split
    # -------------------------------------------------------------------------
    if progress:
        print("\n--- Spatial Group Split ---")

    train_idx, val_idx, test_idx = spatial_group_split(
        grid, positions, seed=split_seed,
    )

    if progress:
        print(f"  Train: {len(train_idx)} ({len(train_idx) / actual_total:.1%})")
        print(f"  Val:   {len(val_idx)} ({len(val_idx) / actual_total:.1%})")
        print(f"  Test:  {len(test_idx)} ({len(test_idx) / actual_total:.1%})")

    # -------------------------------------------------------------------------
    # 7. Leakage Detection
    # -------------------------------------------------------------------------
    if progress:
        print("\n--- Leakage Detection ---")

    joint_leakage = check_joint_space_leakage(
        q_all[train_idx], q_all[test_idx], q_min, q_max,
        threshold=0.01, max_check=min(10000, len(train_idx)),
    )
    pose_leakage = check_pose_leakage(
        positions[train_idx], positions[test_idx],
        rotations_6d[train_idx], rotations_6d[test_idx],
        max_check=min(5000, len(train_idx)),
    )

    if progress:
        print(f"  Joint-space leakage: {joint_leakage['leaking_count']} "
              f"({joint_leakage['leaking_ratio']:.4%})")
        print(f"  Pose leakage: {pose_leakage['leaking_count']} "
              f"({pose_leakage['leaking_ratio']:.4%})")
        print(f"  Min joint distance: {joint_leakage['min_distance']:.4f}")

    # -------------------------------------------------------------------------
    # 8. FK Consistency Check
    # -------------------------------------------------------------------------
    if progress:
        print("\n--- FK Consistency ---")

    # Re-compute FK for a subset and check consistency
    n_check = min(10000, actual_total)
    check_idx = np.random.default_rng(0).choice(actual_total, n_check, replace=False)
    fk_errors = np.zeros(n_check)
    for i, idx in enumerate(check_idx):
        pos_check = fk.position(q_all[idx])
        fk_errors[i] = np.linalg.norm(positions[idx] - pos_check)

    if progress:
        print(f"  Max FK error: {fk_errors.max():.2e} m")
        print(f"  Mean FK error: {fk_errors.mean():.2e} m")

    # -------------------------------------------------------------------------
    # 9. Joint Limit Validity
    # -------------------------------------------------------------------------
    jl_violations = ((q_all < q_min) | (q_all > q_max)).any(axis=1).sum()
    if progress:
        print(f"  Joint limit violations: {jl_violations}")

    # -------------------------------------------------------------------------
    # 10. Save to HDF5
    # -------------------------------------------------------------------------
    if progress:
        print(f"\n--- Saving to HDF5: {output_path} ---")

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Inputs group
        inp = f.create_group('inputs')
        inp.create_dataset('target_position', data=positions.astype(np.float32))
        inp.create_dataset('target_rotation_6d', data=rotations_6d.astype(np.float32))
        inp.create_dataset('q_previous', data=q_previous.astype(np.float32))

        # Outputs group
        out = f.create_group('outputs')
        out.create_dataset('q', data=q_all.astype(np.float32))
        out.create_dataset('sin_q', data=sin_q.astype(np.float32))
        out.create_dataset('cos_q', data=cos_q.astype(np.float32))

        # Physics group
        phys = f.create_group('physics')
        phys.create_dataset('manipulability', data=manipulabilities.astype(np.float32))
        phys.create_dataset('joint_limit_margin', data=joint_limit_margin.astype(np.float32))

        # Split indices
        splits = f.create_group('splits')
        splits.create_dataset('train_indices', data=train_idx)
        splits.create_dataset('val_indices', data=val_idx)
        splits.create_dataset('test_indices', data=test_idx)

        # Sampling method labels
        f.create_dataset('sampling_method', data=method_labels)

        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['robot'] = model.robot_name
        meta.attrs['n_joints'] = model.n_joints
        meta.attrs['seed'] = seed
        meta.attrs['split_seed'] = split_seed
        meta.attrs['n_total'] = actual_total
        meta.attrs['n_uniform'] = n_uniform
        meta.attrs['n_boundary'] = n_boundary
        meta.attrs['n_singularity'] = len(q_singular)
        meta.attrs['singularity_threshold'] = singularity_threshold
        meta.attrs['voxel_resolution'] = list(voxel_resolution)
        meta.attrs['workspace_bounds_min'] = bounds_min.tolist()
        meta.attrs['workspace_bounds_max'] = bounds_max.tolist()
        meta.attrs['version'] = '2.0'

        # Normalization info
        norm = f.create_group('normalization')
        norm.create_dataset('q_min', data=q_min.astype(np.float32))
        norm.create_dataset('q_max', data=q_max.astype(np.float32))
        norm.create_dataset('pos_mean', data=positions.mean(axis=0).astype(np.float32))
        norm.create_dataset('pos_std', data=positions.std(axis=0).astype(np.float32))

    t_elapsed = time.time() - t_start

    # -------------------------------------------------------------------------
    # Build results summary
    # -------------------------------------------------------------------------
    results = {
        'n_total': actual_total,
        'n_uniform': n_uniform,
        'n_boundary': n_boundary,
        'n_singularity': len(q_singular),
        'workspace_coverage': ws_coverage['coverage_ratio'],
        'orientation_coverage': orient_coverage['orientation_coverage_ratio'],
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'joint_leakage': joint_leakage,
        'pose_leakage': pose_leakage,
        'max_fk_error': float(fk_errors.max()),
        'joint_limit_violations': int(jl_violations),
        'elapsed_seconds': t_elapsed,
        'output_path': output_path,
    }

    if progress:
        print(f"\n{'=' * 60}")
        print(f"Dataset generated in {t_elapsed:.1f}s")
        print(f"Saved to: {output_path}")
        print(f"{'=' * 60}")

    return results
