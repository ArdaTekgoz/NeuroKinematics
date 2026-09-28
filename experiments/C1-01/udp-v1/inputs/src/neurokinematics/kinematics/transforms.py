"""Column-vector transforms T_A_B. URDF fixed-axis RPY is Rz(yaw) Ry(pitch) Rx(roll)."""

import numpy as np


def vector3(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("expected a finite three-vector")
    return result


def axis_angle(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation about a joint-local axis (right-hand rule)."""
    axis = vector3(axis)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 0 or not np.isfinite(angle):
        raise ValueError("axis must be nonzero and angle finite")
    x, y, z = axis / length
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def origin_transform(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    roll, pitch, yaw = vector3(rpy)
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = vector3(xyz)
    return result


def check_transform(transform) -> None:
    """Structural validity, separate from the frozen FK comparison thresholds."""
    if transform.shape != (4, 4) or transform.dtype != np.dtype("float64"):
        raise ValueError("transform must be 4x4 float64")
    if not np.isfinite(transform).all():
        raise ValueError("nonfinite transform")
    if not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-12, rtol=0):
        raise ValueError("rotation determinant is not +1")
