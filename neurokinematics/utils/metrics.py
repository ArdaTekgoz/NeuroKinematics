"""
NeuroKinematics - Evaluation Metrics and Kinematic Analysis Utilities.

This module provides a comprehensive suite of evaluation metrics for inverse kinematics (IK),
trajectory smoothness, rotation conversions, hardware performance profiling, and resource monitoring.
"""

import time
import numpy as np
import torch


# ==============================================================================
# 1. Position & Orientation Kinematic Errors
# ==============================================================================

def compute_position_error(pos_pred, pos_target):
    """
    Computes Euclidean position error (e_p) in Cartesian space.

    Args:
        pos_pred: Tensor or NumPy array of predicted positions, shape (B, 3) or (..., 3).
        pos_target: Tensor or NumPy array of target positions, shape (B, 3) or (..., 3).

    Returns:
        Euclidean distance error, shape (B,) or (...,).
    """
    if isinstance(pos_pred, torch.Tensor):
        if not isinstance(pos_target, torch.Tensor):
            pos_target = torch.as_tensor(pos_target, device=pos_pred.device, dtype=pos_pred.dtype)
        return torch.norm(pos_pred - pos_target, dim=-1)
    else:
        pos_pred = np.asarray(pos_pred)
        pos_target = np.asarray(pos_target)
        return np.linalg.norm(pos_pred - pos_target, axis=-1)


def compute_orientation_error_6d(rot_pred, rot_target):
    """
    Computes L2 distance error between 6D continuous rotation representations.

    Args:
        rot_pred: Tensor or NumPy array of predicted 6D rotations, shape (B, 6) or (..., 6).
        rot_target: Tensor or NumPy array of target 6D rotations, shape (B, 6) or (..., 6).

    Returns:
        L2 norm distance error, shape (B,) or (...,).
    """
    if isinstance(rot_pred, torch.Tensor):
        if not isinstance(rot_target, torch.Tensor):
            rot_target = torch.as_tensor(rot_target, device=rot_pred.device, dtype=rot_pred.dtype)
        return torch.norm(rot_pred - rot_target, dim=-1)
    else:
        rot_pred = np.asarray(rot_pred)
        rot_target = np.asarray(rot_target)
        return np.linalg.norm(rot_pred - rot_target, axis=-1)


def compute_orientation_error_geodesic(R_pred, R_target):
    """
    Computes geodesic distance on SO(3) between two 3x3 rotation matrices.

    Formula: arccos((tr(R_pred^T @ R_target) - 1) / 2)

    This metric is more mathematically rigorous than the 6D L2 norm for orientation errors.

    Args:
        R_pred: Tensor or NumPy array of shape (B, 3, 3) or (3, 3).
        R_target: Tensor or NumPy array of shape (B, 3, 3) or (3, 3).

    Returns:
        Geodesic distance error in radians, shape (B,) or scalar.
    """
    if isinstance(R_pred, torch.Tensor):
        if not isinstance(R_target, torch.Tensor):
            R_target = torch.as_tensor(R_target, device=R_pred.device, dtype=R_pred.dtype)
        tr_val = torch.sum(R_pred * R_target, dim=(-2, -1))
        cos_angle = (tr_val - 1.0) / 2.0
        cos_angle_clamped = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_angle_clamped)
    else:
        R_pred = np.asarray(R_pred)
        R_target = np.asarray(R_target)
        tr_val = np.sum(R_pred * R_target, axis=(-2, -1))
        cos_angle = (tr_val - 1.0) / 2.0
        cos_angle_clamped = np.clip(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        return np.arccos(cos_angle_clamped)


# ==============================================================================
# 2. Rotation & Angle Conversions
# ==============================================================================

def rotation_6d_to_matrix(rot_6d):
    """
    Converts 6D continuous rotation representation to 3x3 rotation matrix
    via Gram-Schmidt orthogonalization.

    Reference: Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks", CVPR 2019.

    Args:
        rot_6d: Tensor or NumPy array of shape (B, 6) or (6,).

    Returns:
        Rotation matrix of shape (B, 3, 3) or (3, 3).
    """
    if isinstance(rot_6d, torch.Tensor):
        orig_shape = rot_6d.shape
        if rot_6d.dim() == 1:
            rot_6d = rot_6d.unsqueeze(0)

        x_raw = rot_6d[:, 0:3]
        y_raw = rot_6d[:, 3:6]

        e1 = x_raw / (torch.norm(x_raw, dim=-1, keepdim=True) + 1e-8)
        dot = torch.sum(e1 * y_raw, dim=-1, keepdim=True)
        e2_raw = y_raw - dot * e1
        e2 = e2_raw / (torch.norm(e2_raw, dim=-1, keepdim=True) + 1e-8)
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)
        if len(orig_shape) == 1:
            R = R.squeeze(0)
        return R
    else:
        rot_6d = np.asarray(rot_6d)
        orig_shape = rot_6d.shape
        if rot_6d.ndim == 1:
            rot_6d = np.expand_dims(rot_6d, axis=0)

        x_raw = rot_6d[:, 0:3]
        y_raw = rot_6d[:, 3:6]

        e1 = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
        dot = np.sum(e1 * y_raw, axis=-1, keepdims=True)
        e2_raw = y_raw - dot * e1
        e2 = e2_raw / (np.linalg.norm(e2_raw, axis=-1, keepdims=True) + 1e-8)
        e3 = np.cross(e1, e2, axis=-1)

        R = np.stack([e1, e2, e3], axis=-1)
        if len(orig_shape) == 1:
            R = np.squeeze(R, axis=0)
        return R


def recover_angles_from_sincos(sin_cos_tensor):
    """
    Recovers joint angles in radians from (sin, cos) network outputs.
    Applies normalization before atan2 calculation to ensure numerical stability.

    Args:
        sin_cos_tensor: Tensor or NumPy array of shape (B, 2*num_joints) or (..., 2*num_joints).

    Returns:
        Joint angles in radians, shape (B, num_joints) or (..., num_joints).
    """
    if isinstance(sin_cos_tensor, torch.Tensor):
        s_val = sin_cos_tensor[..., 0::2]
        c_val = sin_cos_tensor[..., 1::2]
        norm = torch.sqrt(s_val**2 + c_val**2 + 1e-8)
        s_norm = s_val / norm
        c_norm = c_val / norm
        return torch.atan2(s_norm, c_norm)
    else:
        sin_cos_tensor = np.asarray(sin_cos_tensor)
        s_val = sin_cos_tensor[..., 0::2]
        c_val = sin_cos_tensor[..., 1::2]
        norm = np.sqrt(s_val**2 + c_val**2 + 1e-8)
        s_norm = s_val / norm
        c_norm = c_val / norm
        return np.arctan2(s_norm, c_norm)


# ==============================================================================
# 3. IK Success Rates & Singularity Evaluation
# ==============================================================================

def compute_ik_success_rate(pos_errors, threshold_mm=1.0, is_meters=True):
    """
    Calculates the percentage of IK predictions where position error is below the tolerance threshold.

    Supports both 1.0 mm and 5.0 mm thresholds (or any custom threshold value).

    Args:
        pos_errors: Tensor or NumPy array of position errors.
        threshold_mm: float, tolerance threshold in millimeters (default: 1.0).
        is_meters: bool, set to True if pos_errors are in meters (default: True).

    Returns:
        float: Success rate percentage in range [0.0, 100.0].
    """
    threshold = threshold_mm / 1000.0 if is_meters else threshold_mm
    if isinstance(pos_errors, torch.Tensor):
        if pos_errors.numel() == 0:
            return 0.0
        success_mask = (pos_errors < threshold) | torch.isclose(
            pos_errors, torch.tensor(threshold, device=pos_errors.device, dtype=pos_errors.dtype)
        )
        return float(success_mask.float().mean().item() * 100.0)
    else:
        pos_errors = np.asarray(pos_errors)
        if pos_errors.size == 0:
            return 0.0
        success_mask = pos_errors <= threshold
        return float(np.mean(success_mask) * 100.0)


def compute_singularity_success_rate(
    pos_errors, manipulability_scores, singularity_threshold=0.01, error_threshold_mm=5.0, is_meters=True
):
    """
    Calculates IK success rate specifically for configurations near singularity
    (where Yoshikawa manipulability index < singularity_threshold).

    Args:
        pos_errors: Tensor or NumPy array of position errors.
        manipulability_scores: Tensor or NumPy array of manipulability scores.
        singularity_threshold: float, threshold below which a configuration is near singular (default: 0.01).
        error_threshold_mm: float, position error success threshold in mm (default: 5.0).
        is_meters: bool, set to True if pos_errors are in meters (default: True).

    Returns:
        float: IK success rate percentage near singularities in range [0.0, 100.0].
    """
    threshold = error_threshold_mm / 1000.0 if is_meters else error_threshold_mm

    if isinstance(pos_errors, torch.Tensor):
        if not isinstance(manipulability_scores, torch.Tensor):
            manipulability_scores = torch.as_tensor(manipulability_scores, device=pos_errors.device)
        mask = manipulability_scores < singularity_threshold
        if not mask.any():
            return 0.0
        singular_pos_errors = pos_errors[mask]
        success_mask = (singular_pos_errors < threshold) | torch.isclose(
            singular_pos_errors, torch.tensor(threshold, device=pos_errors.device, dtype=pos_errors.dtype)
        )
        return float(success_mask.float().mean().item() * 100.0)
    else:
        pos_errors = np.asarray(pos_errors)
        manipulability_scores = np.asarray(manipulability_scores)
        mask = manipulability_scores < singularity_threshold
        if not np.any(mask):
            return 0.0
        singular_pos_errors = pos_errors[mask]
        success_mask = singular_pos_errors <= threshold
        return float(np.mean(success_mask) * 100.0)


# ==============================================================================
# 4. Joint Constraints & Trajectory Dynamics Metrics
# ==============================================================================

def compute_joint_limit_violation(q_pred, q_min, q_max):
    """
    Computes total joint limit violation amount for joint angle predictions.

    Args:
        q_pred: Tensor or NumPy array of predicted joint angles, shape (B, D) or (D,).
        q_min: Tensor or NumPy array of lower joint limits, shape (D,) or (B, D).
        q_max: Tensor or NumPy array of upper joint limits, shape (D,) or (B, D).

    Returns:
        Total limit violation sum per sample, shape (B,) or scalar.
    """
    if isinstance(q_pred, torch.Tensor):
        if not isinstance(q_min, torch.Tensor):
            q_min = torch.as_tensor(q_min, device=q_pred.device, dtype=q_pred.dtype)
        if not isinstance(q_max, torch.Tensor):
            q_max = torch.as_tensor(q_max, device=q_pred.device, dtype=q_pred.dtype)
        violation_lower = torch.relu(q_min - q_pred)
        violation_upper = torch.relu(q_pred - q_max)
        return torch.sum(violation_lower + violation_upper, dim=-1)
    else:
        q_pred = np.asarray(q_pred)
        q_min = np.asarray(q_min)
        q_max = np.asarray(q_max)
        violation_lower = np.maximum(0.0, q_min - q_pred)
        violation_upper = np.maximum(0.0, q_pred - q_max)
        return np.sum(violation_lower + violation_upper, axis=-1)


def compute_jerk(q_seq, dt=1.0):
    """
    Computes jerk (3rd time derivative) of joint trajectories for motion continuity analysis.

    Args:
        q_seq: Tensor or NumPy array of joint trajectories, shape (B, T, D) or (T, D).
        dt: float, sampling time step between frames (default: 1.0).

    Returns:
        Mean jerk norm per trajectory sequence, shape (B,) or scalar.
    """
    if isinstance(q_seq, torch.Tensor):
        is_2d = q_seq.dim() == 2
        if is_2d:
            q_seq = q_seq.unsqueeze(0)
        if q_seq.shape[1] < 4:
            res = torch.zeros(q_seq.shape[0], device=q_seq.device, dtype=q_seq.dtype)
            return res.squeeze(0) if is_2d else res

        dq = (q_seq[:, 1:] - q_seq[:, :-1]) / dt
        ddq = (dq[:, 1:] - dq[:, :-1]) / dt
        dddq = (ddq[:, 1:] - ddq[:, :-1]) / dt
        res = torch.norm(dddq, dim=-1).mean(dim=1)
        return res.squeeze(0) if is_2d else res
    else:
        q_seq = np.asarray(q_seq)
        is_2d = q_seq.ndim == 2
        if is_2d:
            q_seq = np.expand_dims(q_seq, axis=0)
        if q_seq.shape[1] < 4:
            res = np.zeros(q_seq.shape[0], dtype=q_seq.dtype)
            return res.squeeze(0) if is_2d else res

        dq = (q_seq[:, 1:] - q_seq[:, :-1]) / dt
        ddq = (dq[:, 1:] - dq[:, :-1]) / dt
        dddq = (ddq[:, 1:] - ddq[:, :-1]) / dt
        res = np.linalg.norm(dddq, axis=-1).mean(axis=1)
        return res.squeeze(0) if is_2d else res


def compute_smoothness(q_seq, dt=1.0):
    """
    Computes trajectory smoothness as the norm of joint acceleration (2nd derivative).

    Args:
        q_seq: Tensor or NumPy array of joint trajectories, shape (B, T, D) or (T, D).
        dt: float, sampling time step between frames (default: 1.0).

    Returns:
        Mean acceleration norm per trajectory sequence, shape (B,) or scalar.
    """
    if isinstance(q_seq, torch.Tensor):
        is_2d = q_seq.dim() == 2
        if is_2d:
            q_seq = q_seq.unsqueeze(0)
        if q_seq.shape[1] < 3:
            res = torch.zeros(q_seq.shape[0], device=q_seq.device, dtype=q_seq.dtype)
            return res.squeeze(0) if is_2d else res

        dq = (q_seq[:, 1:] - q_seq[:, :-1]) / dt
        ddq = (dq[:, 1:] - dq[:, :-1]) / dt
        res = torch.norm(ddq, dim=-1).mean(dim=1)
        return res.squeeze(0) if is_2d else res
    else:
        q_seq = np.asarray(q_seq)
        is_2d = q_seq.ndim == 2
        if is_2d:
            q_seq = np.expand_dims(q_seq, axis=0)
        if q_seq.shape[1] < 3:
            res = np.zeros(q_seq.shape[0], dtype=q_seq.dtype)
            return res.squeeze(0) if is_2d else res

        dq = (q_seq[:, 1:] - q_seq[:, :-1]) / dt
        ddq = (dq[:, 1:] - dq[:, :-1]) / dt
        res = np.linalg.norm(ddq, axis=-1).mean(axis=1)
        return res.squeeze(0) if is_2d else res


def compute_jitter(q_seq):
    """
    Computes jitter as the standard deviation of consecutive joint angle differences.

    Args:
        q_seq: Tensor or NumPy array of joint angle sequences, shape (B, T, D) or (T, D).

    Returns:
        Jitter metric (standard deviation of consecutive joint angle differences).
    """
    if isinstance(q_seq, torch.Tensor):
        if q_seq.dim() == 2:
            diff = q_seq[1:] - q_seq[:-1]
            return torch.std(diff)
        else:
            diff = q_seq[:, 1:] - q_seq[:, :-1]
            return torch.std(diff, dim=(1, 2))
    else:
        q_seq = np.asarray(q_seq)
        if q_seq.ndim == 2:
            diff = q_seq[1:] - q_seq[:-1]
            return np.std(diff)
        else:
            diff = q_seq[:, 1:] - q_seq[:, :-1]
            return np.std(diff, axis=(1, 2))


# ==============================================================================
# 5. Performance Profiling & System Resource Usage
# ==============================================================================

def compute_inference_latency(model, sample_input, n_runs=1000, warmup=100):
    """
    Profiles inference timing and computes latency statistics.

    Uses torch.cuda.Event for GPU timing if CUDA is available and model/inputs are on CUDA,
    otherwise uses time.perf_counter for CPU timing.

    Args:
        model: PyTorch module (torch.nn.Module) or callable inference function.
        sample_input: Tensor or tuple/list of Tensors matching model input signature.
        n_runs: int, number of timed runs (default: 1000).
        warmup: int, number of warmup runs before profiling (default: 100).

    Returns:
        dict: Latency statistics with keys:
              'mean_ms', 'std_ms', 'p95_ms', 'p99_ms', 'min_ms', 'max_ms'.
    """
    if hasattr(model, 'eval'):
        model.eval()

    is_tuple_input = isinstance(sample_input, (tuple, list))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if isinstance(model, torch.nn.Module):
            try:
                first_param = next(model.parameters())
                use_cuda = first_param.is_cuda
            except StopIteration:
                pass
        elif is_tuple_input and len(sample_input) > 0:
            use_cuda = isinstance(sample_input[0], torch.Tensor) and sample_input[0].is_cuda
        elif isinstance(sample_input, torch.Tensor):
            use_cuda = sample_input.is_cuda

    latencies_ms = []

    with torch.no_grad():
        for _ in range(warmup):
            if is_tuple_input:
                _ = model(*sample_input)
            else:
                _ = model(sample_input)

        if use_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(n_runs):
                start_event.record()
                if is_tuple_input:
                    _ = model(*sample_input)
                else:
                    _ = model(sample_input)
                end_event.record()
                torch.cuda.synchronize()
                latencies_ms.append(start_event.elapsed_time(end_event))
        else:
            for _ in range(n_runs):
                t0 = time.perf_counter()
                if is_tuple_input:
                    _ = model(*sample_input)
                else:
                    _ = model(sample_input)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    return {
        'mean_ms': float(np.mean(latencies_arr)),
        'std_ms': float(np.std(latencies_arr)),
        'p95_ms': float(np.percentile(latencies_arr, 95)),
        'p99_ms': float(np.percentile(latencies_arr, 99)),
        'min_ms': float(np.min(latencies_arr)),
        'max_ms': float(np.max(latencies_arr))
    }


def compute_resource_usage():
    """
    Returns current system resource usage metrics.

    Uses psutil for CPU/RAM when available, and torch.cuda for GPU memory/utilization.

    Returns:
        dict: Resource statistics containing 'cpu_percent', 'ram_mb'.
              If CUDA is available, also includes 'gpu_memory_mb', 'gpu_utilization_percent'.
    """
    stats = {}

    try:
        import psutil
        stats['cpu_percent'] = float(psutil.cpu_percent(interval=None))
        stats['ram_mb'] = float(psutil.virtual_memory().used / (1024 * 1024))
    except ImportError:
        stats['cpu_percent'] = 0.0
        stats['ram_mb'] = 0.0

    if torch.cuda.is_available():
        stats['gpu_memory_mb'] = float(torch.cuda.memory_allocated() / (1024 * 1024))
        gpu_util = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util.gpu)
        except Exception:
            gpu_util = 0.0
        stats['gpu_utilization_percent'] = gpu_util

    return stats
