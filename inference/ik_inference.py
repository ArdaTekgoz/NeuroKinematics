import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from core.model import IKNet
from core.fk import forward_kinematics_position


def load_ik_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained IKNet model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch.device to load model onto
        
    Returns:
        IKNet model in eval mode
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If checkpoint missing 'model_state_dict'
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict' key")
    
    model = IKNet(in_dim=7, out_dim=6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    return model


def pose_to_tensor(
    pose: Union[list, np.ndarray], 
    device: torch.device
) -> torch.Tensor:
    """
    Convert pose to PyTorch tensor.
    
    Args:
        pose: Input pose [x, y, z, qx, qy, qz, qw]
        device: Target device
        
    Returns:
        Tensor of shape (1, 7)
        
    Raises:
        ValueError: If pose length is not 7
    """
    pose_array = np.array(pose, dtype=np.float32)
    
    if pose_array.shape != (7,):
        raise ValueError(f"Pose must have shape (7,), got {pose_array.shape}")
    
    q = pose_array[3:]
    n = np.linalg.norm(q)
    if n > 0:
        pose_array[3:] = q / n
    
    tensor = torch.from_numpy(pose_array).unsqueeze(0).to(device)
    return tensor


def solve_ik_pose(
    model: nn.Module,
    pose7d: Union[list, np.ndarray],
    device: torch.device,
    return_fk_error: bool = True
) -> Dict[str, Union[np.ndarray, Optional[float]]]:
    """
    Solve inverse kinematics for a single pose.
    
    Args:
        model: Trained IKNet model
        pose7d: Input pose [x, y, z, qx, qy, qz, qw]
        device: Computation device
        return_fk_error: Whether to compute forward kinematics error
        
    Returns:
        Dictionary with:
            - 'theta': Joint angles, shape (6,)
            - 'fk_error': L2 position error or None
    """
    with torch.no_grad():
        pose_tensor = pose_to_tensor(pose7d, device)
        theta_tensor = model(pose_tensor)
        theta = theta_tensor.cpu().numpy().squeeze()
        
        fk_error = None
        if return_fk_error:
            fk_pos = forward_kinematics_position(theta_tensor)
            target_pos = pose_tensor[:, :3]
            error_tensor = torch.norm(fk_pos - target_pos, dim=1)
            fk_error = float(error_tensor.item())
    
    return {
        "theta": theta,
        "fk_error": fk_error
    }


def solve_ik_batch(
    model: nn.Module,
    poses7d: np.ndarray,
    device: torch.device,
    return_fk_error: bool = True
) -> Dict[str, Union[np.ndarray, Optional[np.ndarray]]]:
    """
    Solve inverse kinematics for a batch of poses.
    
    Args:
        model: Trained IKNet model
        poses7d: Input poses, shape (N, 7)
        device: Computation device
        return_fk_error: Whether to compute forward kinematics errors
        
    Returns:
        Dictionary with:
            - 'theta': Joint angles, shape (N, 6)
            - 'fk_error': L2 position errors, shape (N,) or None
            
    Raises:
        ValueError: If input shape is invalid
    """
    poses_array = np.asarray(poses7d, dtype=np.float32)
    
    if poses_array.ndim != 2 or poses_array.shape[1] != 7:
        raise ValueError(f"Poses must have shape (N, 7), got {poses_array.shape}")
    
    with torch.no_grad():
        poses_tensor = torch.from_numpy(poses_array).contiguous().to(device)
        theta_tensor = model(poses_tensor)
        theta = theta_tensor.cpu().numpy()
        
        fk_errors = None
        if return_fk_error:
            fk_positions = forward_kinematics_position(theta_tensor)
            target_positions = poses_tensor[:, :3]
            error_tensor = torch.norm(fk_positions - target_positions, dim=1)
            fk_errors = error_tensor.cpu().numpy()
    
    return {
        "theta": theta,
        "fk_error": fk_errors
    }


def benchmark_single_pose(
    model: nn.Module,
    pose7d: Union[list, np.ndarray],
    device: torch.device,
    runs: int = 100
) -> float:
    """
    Benchmark inference latency for a single pose.
    
    Args:
        model: Trained IKNet model
        pose7d: Input pose [x, y, z, qx, qy, qz, qw]
        device: Computation device
        runs: Number of inference runs
        
    Returns:
        Average inference time in milliseconds
    """
    pose_tensor = pose_to_tensor(pose7d, device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(pose_tensor)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(pose_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return float(np.mean(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IK Inference Test")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_ik_model(args.checkpoint, device)
    print("Model loaded successfully")
    
    # Create random test pose
    test_pose = np.random.randn(7).astype(np.float32)
    test_pose[3:] /= np.linalg.norm(test_pose[3:])  # Normalize quaternion
    
    print(f"\nTest pose: {test_pose}")
    
    # Solve IK
    result = solve_ik_pose(model, test_pose, device, return_fk_error=True)
    print(f"Joint angles: {result['theta']}")
    print(f"FK error: {result['fk_error']:.6f} units")
    
    # Benchmark
    latency = benchmark_single_pose(model, test_pose, device, runs=100)
    print(f"\nAverage latency: {latency:.3f} ms")