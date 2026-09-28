import numpy as np
from numpy.testing import assert_allclose
import pytest

from neurokinematics.kinematics.metrics import (
    position_error, rotation_error, quaternion_error, quaternion_rotation,
    quaternion_wxyz, normalize_jacobian, normalized_difference, singularity_metrics,
)
from neurokinematics.kinematics.finite_difference import log_so3
from neurokinematics.kinematics.transforms import axis_angle


@pytest.mark.parametrize("angle", [0, np.pi/2, np.pi])
@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_rotation_and_wxyz_analytic(axis, angle):
    q = np.r_[np.cos(angle/2), np.array(axis)*np.sin(angle/2)]
    r = axis_angle(axis, angle)
    assert_allclose(quaternion_rotation(q), r, atol=3e-16, rtol=0)
    assert rotation_error(r, np.eye(3)) == pytest.approx(angle, abs=1e-15)
    assert quaternion_error(q, [1, 0, 0, 0]) == pytest.approx(angle, abs=1e-15)
    assert quaternion_error(q, -q) == 0


def test_wxyz_order_not_xyzw():
    assert_allclose(quaternion_rotation([1, 0, 0, 0]), np.eye(3), atol=0)
    assert_allclose(quaternion_rotation([0, 0, 0, 1]), np.diag([-1, -1, 1]), atol=0)


def test_quaternion_normalization_copy_and_sign():
    q = np.array([.5, .5, .5, .5]) * (1+1e-7)
    saved = q.copy()
    norm = quaternion_wxyz(q)
    assert_allclose(norm, [.5]*4, atol=1e-15, rtol=0)
    assert quaternion_error(q, -q) == 0
    assert np.array_equal(q, saved) and not np.shares_memory(q, norm)


@pytest.mark.parametrize("q", [[0]*4, [np.nan, 0, 0, 1], [np.inf, 0, 0, 1], [-np.inf, 0, 0, 1],
                               [1, 0, 0], [[1, 0, 0, 0]], [2, 0, 0, 0], [1j, 0, 0, 0]])
def test_invalid_quaternion(q):
    for function in (quaternion_wxyz, quaternion_rotation, lambda v: quaternion_error(v, [1, 0, 0, 0])):
        with pytest.raises(ValueError):
            function(q)


@pytest.mark.parametrize("r", [np.zeros((3, 3)), np.diag([1, 1, -1]), np.eye(4),
                               np.full((3, 3), np.nan), np.full((3, 3), np.inf),
                               np.diag([1, 1, 1.00001])])
def test_invalid_rotation(r):
    for function in (log_so3, lambda v: rotation_error(v, np.eye(3))):
        with pytest.raises(ValueError):
            function(r)


def test_rotation_clip_roundoff_at_zero_and_pi():
    assert rotation_error(np.eye(3)*(1+2e-16), np.eye(3)) == 0
    assert rotation_error(np.diag([1., -1-2e-16, -1-2e-16]), np.eye(3)) == np.pi
    assert rotation_error(axis_angle([0, 1, 0], np.pi-1e-7), np.eye(3)) == pytest.approx(np.pi-1e-7, abs=2e-9)


@pytest.mark.parametrize("est,target,expected", [([0, 0, 0], [0, 0, 0], 0),
                                                  ([.003, .004, 0], [0, 0, 0], .005),
                                                  ([1, 2, 3], [1, 2, 4], 1)])
def test_position_metres_and_mm(est, target, expected):
    assert position_error(est, target) == pytest.approx(expected, abs=1e-15)
    assert 1000*position_error(est, target) == pytest.approx(1000*expected, abs=1e-12)


@pytest.mark.parametrize("value", [[0, 0], [0, np.nan, 0], [0, np.inf, 0]])
def test_invalid_position(value):
    with pytest.raises(ValueError):
        position_error(value, [0, 0, 0])


@pytest.mark.parametrize("diagonal", [[1]*6, [1, 2, 3, 4, 5, 6], [0]*6,
                                     [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1e-12],
                                     [1, 1, 1, 1, 1, 1e-18]])
def test_svd_analytic(diagonal):
    result = singularity_metrics(np.diag(diagonal), 1.)
    assert_allclose(result["singular_values"], sorted(diagonal, reverse=True), atol=0, rtol=0)
    assert result["sigma_min"] == min(diagonal)
    assert result["sigma_max"] == max(diagonal)
    expected = max(diagonal)/min(diagonal) if min(diagonal) else np.inf
    assert result["condition"] == expected
    assert result["manipulability"] == pytest.approx(np.prod(diagonal), rel=1e-14, abs=0)


def test_rank_deficient_non_diagonal():
    j = np.eye(6); j[-1] = j[0]
    result = singularity_metrics(j, 1.)
    assert result["numerical_rank"] == 5
    assert result["manipulability"] == 0


def test_scale_and_difference():
    j = np.diag([.9015]*3 + [1.]*3)
    saved = j.copy()
    assert_allclose(normalize_jacobian(j, .9015), np.eye(6), atol=0, rtol=0)
    assert_allclose(singularity_metrics(j, .9015)["singular_values"], np.ones(6), atol=0)
    assert np.array_equal(j, saved)
    assert normalized_difference(j, np.zeros((6, 6)), .9015) == pytest.approx(np.sqrt(6))
    assert normalized_difference(2*j, j, .9015) == pytest.approx(1.)


@pytest.mark.parametrize("length", [0, -1, np.nan, np.inf])
def test_invalid_scale(length):
    with pytest.raises(ValueError):
        singularity_metrics(np.eye(6), length)


@pytest.mark.parametrize("j", [np.eye(5), np.ones((6, 5)), np.full((6, 6), np.nan), np.full((6, 6), np.inf)])
def test_invalid_jacobian(j):
    with pytest.raises(ValueError):
        singularity_metrics(j, .9015)
