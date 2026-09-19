"""Float64 pose and dimensionless 6x6 Jacobian metrics (m, rad, wxyz)."""

import numpy as np


def finite_array(value, shape):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"expected real numeric shape {shape}")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("input must be finite")
    return result


def rotation_matrix(value):
    r = finite_array(value, (3, 3))
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(r), 1, atol=1e-12, rtol=0):
        raise ValueError("rotation determinant must be +1")
    return r


def position_error(est, target):
    """Euclidean distance in metres; multiply by 1000 only for presentation."""
    return float(np.linalg.norm(finite_array(est, (3,)) - finite_array(target, (3,))))


def rotation_error(est, target):
    """Geodesic angle in radians, trace/arccos with roundoff clipping."""
    relative = rotation_matrix(target).T @ rotation_matrix(est)
    return float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))


def quaternion_wxyz(value):
    """Accept norm within 1e-6 of one; return normalized COPY. No silent repair."""
    q = finite_array(value, (4,))
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm) or abs(norm - 1) > 1e-6:
        raise ValueError("quaternion norm must be nonzero and within 1e-6 of one")
    return q / norm


def quaternion_rotation(value):
    w, x, y, z = quaternion_wxyz(value)
    # Symmetric squared-component form avoids subtracting twice a rounded
    # half from one at quarter turns.
    return np.array([[w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z]], dtype=np.float64)


def quaternion_error(est, target):
    a, b = quaternion_wxyz(est), quaternion_wxyz(target)
    if np.array_equal(a, b) or np.array_equal(a, -b):
        return 0.0
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))


def normalize_jacobian(jacobian, length):
    j = finite_array(jacobian, (6, 6)).copy()
    if not np.isscalar(length) or not np.isfinite(length) or length <= 0:
        raise ValueError("characteristic length must be positive and finite")
    j[:3] /= length
    return j


def normalized_difference(actual, reference, length):
    a, b = normalize_jacobian(actual, length), normalize_jacobian(reference, length)
    return float(np.linalg.norm(a-b, ord="fro") / max(1., np.linalg.norm(b, ord="fro")))


def singularity_metrics(jacobian, length):
    j = normalize_jacobian(jacobian, length)
    singular = np.linalg.svd(j, compute_uv=False)
    lo, hi = float(singular[-1]), float(singular[0])
    # Report numerical rank explicitly; do not modify singular values or add
    # epsilon to the denominator. Rank tolerance is the standard SVD roundoff
    # criterion, NOT a tuned singularity classification or acceptance gate.
    tolerance = float(6 * np.finfo(np.float64).eps * hi)
    rank = int(np.count_nonzero(singular > tolerance))
    return {"singular_values": singular.tolist(), "sigma_min": lo, "sigma_max": hi,
            "condition": float(hi / lo) if lo != 0 else float("inf"),
            "manipulability": float(np.prod(singular)),
            "numerical_rank": rank, "rank_tolerance": tolerance}
