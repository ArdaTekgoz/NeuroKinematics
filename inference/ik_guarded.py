import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from inference.ik_inference import solve_ik_pose, pose_to_tensor, load_ik_model


DEFAULT_LIMITS_MIN = np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14], dtype=np.float32)
DEFAULT_LIMITS_MAX = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14], dtype=np.float32)


@dataclass
class IKGuardConfig:
    """Configuration for IK guard layer."""
    joint_min: np.ndarray  # shape (6,)
    joint_max: np.ndarray  # shape (6,)
    fk_error_threshold: float = 0.25
    reject_if_over_threshold: bool = False


def clamp_joints(theta: np.ndarray, cfg: IKGuardConfig) -> Tuple[np.ndarray, bool]:
    """
    Clamp joint angles to valid range.
    
    Args:
        theta: Joint angles, shape (6,)
        cfg: Guard configuration with joint limits
        
    Returns:
        Tuple of (clamped_theta, was_clamped_flag)
    """
    if theta.shape != (6,):
        raise ValueError(f"theta must be shape (6,), got {theta.shape}")
    
    clamped_theta = np.clip(theta, cfg.joint_min, cfg.joint_max)
    was_clamped = not np.allclose(theta, clamped_theta)
    return clamped_theta, was_clamped


def confidence_from_fk_error(err: float, threshold: float) -> float:
    """
    Map FK error to confidence score.
    
    Args:
        err: Forward kinematics error
        threshold: Error threshold
        
    Returns:
        Confidence score in [0, 1]
    """
    confidence = np.exp(-err / threshold)
    return float(np.clip(confidence, 0.0, 1.0))


def guarded_solve_ik_pose(
    model: nn.Module,
    pose7d: Union[list, np.ndarray],
    device: torch.device,
    cfg: IKGuardConfig
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Solve IK with safety guards and validation.
    
    Args:
        model: Trained IKNet model
        pose7d: Input pose [x, y, z, qx, qy, qz, qw]
        device: Computation device
        cfg: Guard configuration
        
    Returns:
        Dictionary with:
            - 'theta': Clamped joint angles, shape (6,)
            - 'fk_error': Forward kinematics error
            - 'confidence': Confidence score [0, 1]
            - 'status': Status string
    """
    # Solve IK
    result = solve_ik_pose(model, pose7d, device, return_fk_error=True)
    theta = result["theta"]
    fk_error = result["fk_error"]
    
    if fk_error is None:
        raise RuntimeError("FK error not computed — return_fk_error must be True")
    
    # Clamp joints
    theta_clamped, was_clamped = clamp_joints(theta, cfg)
    
    # Compute confidence
    confidence = confidence_from_fk_error(fk_error, cfg.fk_error_threshold)
    
    # Build status string
    if fk_error <= cfg.fk_error_threshold:
        status = "OK"
    else:
        status = "HIGH_FK_ERROR"
    
    if was_clamped:
        status += "_CLAMPED"
    
    if fk_error > cfg.fk_error_threshold and cfg.reject_if_over_threshold:
        status = "REJECTED"
    
    return {
        "theta": theta_clamped,
        "fk_error": fk_error,
        "confidence": confidence,
        "status": status
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guarded IK Inference Test")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_ik_model(args.checkpoint, device)
    print("Model loaded successfully")
    
    # Build default config
    config = IKGuardConfig(
        joint_min=DEFAULT_LIMITS_MIN,
        joint_max=DEFAULT_LIMITS_MAX,
        fk_error_threshold=0.25,
        reject_if_over_threshold=False
    )
    
    # Create random test pose
    test_pose = np.random.randn(7).astype(np.float32)
    test_pose[3:] /= np.linalg.norm(test_pose[3:])  # Normalize quaternion
    
    print(f"\nTest pose: {test_pose}")
    print(f"Joint limits: [{config.joint_min[0]:.2f}, {config.joint_max[0]:.2f}]")
    print(f"FK error threshold: {config.fk_error_threshold}")
    
    # Run guarded solve
    result = guarded_solve_ik_pose(model, test_pose, device, config)
    
    print(f"\nResults:")
    print(f"  Joint angles: {result['theta']}")
    print(f"  FK error: {result['fk_error']:.6f}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Status: {result['status']}")