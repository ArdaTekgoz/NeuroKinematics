"""
NeuroKinematics — Sampling Engine

Three sampling strategies for synthetic IK dataset generation:
  1. Uniform Joint-Space Sampling (70%)
  2. Workspace Boundary Sampling (20%)
  3. Singularity-Enriched Sampling (10%)

All strategies are deterministic given a seed.
"""

import numpy as np
from typing import Tuple, Optional


def uniform_joint_sampling(
    n_samples: int,
    q_min: np.ndarray,
    q_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Uniform random sampling across the full joint space.

    Args:
        n_samples: Number of configurations to generate.
        q_min: Lower joint limits, shape (n_joints,).
        q_max: Upper joint limits, shape (n_joints,).
        rng: NumPy random generator for determinism.

    Returns:
        Joint configurations, shape (n_samples, n_joints).
    """
    n_joints = len(q_min)
    return rng.uniform(q_min, q_max, size=(n_samples, n_joints))


def workspace_boundary_sampling(
    n_samples: int,
    q_min: np.ndarray,
    q_max: np.ndarray,
    rng: np.random.Generator,
    boundary_fraction: float = 0.15,
) -> np.ndarray:
    """
    Sampling biased toward workspace boundaries (joint limit regions).

    At least one joint is forced near its limit (within boundary_fraction
    of the total range from either end). Other joints are uniform.

    Args:
        n_samples: Number of configurations to generate.
        q_min: Lower joint limits, shape (n_joints,).
        q_max: Upper joint limits, shape (n_joints,).
        rng: NumPy random generator.
        boundary_fraction: Fraction of joint range considered "near boundary".

    Returns:
        Joint configurations, shape (n_samples, n_joints).
    """
    n_joints = len(q_min)
    q_range = q_max - q_min
    margin = boundary_fraction * q_range

    # Start with uniform samples
    samples = rng.uniform(q_min, q_max, size=(n_samples, n_joints))

    for i in range(n_samples):
        # Pick a random joint to push toward boundary
        joint_idx = rng.integers(0, n_joints)

        if rng.random() < 0.5:
            # Near lower limit
            samples[i, joint_idx] = rng.uniform(
                q_min[joint_idx],
                q_min[joint_idx] + margin[joint_idx]
            )
        else:
            # Near upper limit
            samples[i, joint_idx] = rng.uniform(
                q_max[joint_idx] - margin[joint_idx],
                q_max[joint_idx]
            )

    return samples


def singularity_enriched_sampling(
    n_samples: int,
    q_min: np.ndarray,
    q_max: np.ndarray,
    fk_instance,
    jacobian_fn,
    manipulability_fn,
    rng: np.random.Generator,
    singularity_threshold: float = 0.01,
    max_attempts_multiplier: int = 20,
) -> np.ndarray:
    """
    Sampling enriched around singularity regions (low manipulability).

    Generates candidates uniformly and keeps those with manipulability
    below and around the threshold, creating a gradual distribution
    from near-singular to singular configurations.

    Args:
        n_samples: Number of singularity-enriched samples desired.
        q_min: Lower joint limits.
        q_max: Upper joint limits.
        fk_instance: ForwardKinematics instance.
        jacobian_fn: Function(fk, q) -> Jacobian matrix.
        manipulability_fn: Function(J) -> float manipulability.
        rng: NumPy random generator.
        singularity_threshold: Manipulability threshold for "near singular".
        max_attempts_multiplier: How many more candidates to generate.

    Returns:
        Joint configurations near singularity, shape (n_collected, n_joints).
        May return fewer than n_samples if singularities are rare.
    """
    n_joints = len(q_min)
    collected = []
    manipulabilities = []

    # We use 3 tiers to get gradual coverage:
    # Tier 1: w < threshold (deep singularity)
    # Tier 2: threshold <= w < 5*threshold (near singularity)
    # Tier 3: 5*threshold <= w < 10*threshold (approaching singularity)
    tier_limits = [singularity_threshold, 5 * singularity_threshold, 10 * singularity_threshold]
    tier_targets = [int(n_samples * 0.4), int(n_samples * 0.35), n_samples]  # cumulative-ish
    tier_collected = [[], [], []]

    total_attempts = n_samples * max_attempts_multiplier
    batch_size = min(10000, total_attempts)

    attempts = 0
    while sum(len(t) for t in tier_collected) < n_samples and attempts < total_attempts:
        batch = rng.uniform(q_min, q_max, size=(batch_size, n_joints))
        for q in batch:
            J = jacobian_fn(fk_instance, q)
            w = manipulability_fn(J)

            if w < tier_limits[0] and len(tier_collected[0]) < tier_targets[0]:
                tier_collected[0].append(q)
            elif w < tier_limits[1] and len(tier_collected[1]) < tier_targets[1]:
                tier_collected[1].append(q)
            elif w < tier_limits[2] and len(tier_collected[2]) < (n_samples - len(tier_collected[0]) - len(tier_collected[1])):
                tier_collected[2].append(q)

            if sum(len(t) for t in tier_collected) >= n_samples:
                break

        attempts += batch_size

    # Combine all tiers
    all_samples = []
    for tier in tier_collected:
        all_samples.extend(tier)

    if len(all_samples) == 0:
        # Fallback: return uniform samples
        return rng.uniform(q_min, q_max, size=(n_samples, n_joints))

    result = np.array(all_samples[:n_samples])
    return result


def generate_previous_configs(
    q_targets: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.3,
) -> np.ndarray:
    """
    Generate plausible q_{t-1} configurations for state-conditioned input.

    For each target q, creates a "previous" configuration by adding
    Gaussian noise (simulating the robot was nearby before moving).

    Args:
        q_targets: Target joint angles, shape (N, n_joints).
        q_min: Lower limits.
        q_max: Upper limits.
        rng: NumPy random generator.
        noise_std: Standard deviation of Gaussian noise (radians).

    Returns:
        Previous configurations, shape (N, n_joints), clipped to limits.
    """
    noise = rng.normal(0, noise_std, size=q_targets.shape)
    q_prev = q_targets + noise
    return np.clip(q_prev, q_min, q_max)
