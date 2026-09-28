import hashlib
import json

import numpy as np
import pytest

from neurokinematics.data.factory import (assign_splits, boundary_samples, canonical_array_hash,
    canonical_quaternion, inherited_split, latin_hypercube, normalization, validate_arrays)
from neurokinematics.kinematics.metrics import quaternion_rotation


def test_lhs_stratifies_every_dimension(limits):
    q = latin_hypercube(100, limits, 20260920)
    bounds = np.asarray(limits)
    cells = np.floor(((q-bounds[:,0])/(bounds[:,1]-bounds[:,0]))*100).astype(int)
    assert all(np.array_equal(np.sort(cells[:,d]), np.arange(100)) for d in range(6))


def test_lhs_same_seed_same_bytes_and_different_seed_differs(limits):
    a, b = latin_hypercube(64, limits, 1), latin_hypercube(64, limits, 1)
    c = latin_hypercube(64, limits, 2)
    assert a.tobytes() == b.tobytes()
    assert hashlib.sha256(a.tobytes()).digest() != hashlib.sha256(c.tobytes()).digest()


def test_split_is_deterministic_and_exact():
    groups = [f"g{i}" for i in range(100)]
    a = assign_splits(groups, 3, {"train":.7,"validation":.15,"test":.15})
    assert np.array_equal(a, assign_splits(groups, 3, {"train":.7,"validation":.15,"test":.15}))
    assert [np.count_nonzero(a == x) for x in (b"train",b"validation",b"test")] == [70,15,15]


def test_duplicate_root_rejected():
    with pytest.raises(ValueError, match="duplicate group_id"):
        assign_splits(["same", "same"], 1, {"train":.7,"validation":.15,"test":.15})


def test_variants_inherit_root_split():
    result = inherited_split(["a","b"], ["a","a","b"], [b"train",b"test"])
    assert result.tolist() == [b"train",b"train",b"test"]


def test_quaternion_pi_tie_and_sign_are_canonical():
    rotation = np.diag([1.,-1.,-1.])
    quaternion = canonical_quaternion(rotation)
    assert np.array_equal(quaternion, np.array([0.,1.,0.,0.]))
    assert np.allclose(quaternion_rotation(quaternion), rotation, atol=1e-15, rtol=0)


def test_boundary_is_strict_and_balanced(limits):
    q, labels = boundary_samples(1000, limits, 7, .02)
    bounds = np.asarray(limits)
    distance = np.minimum((q-bounds[:,0])/(bounds[:,1]-bounds[:,0]), (bounds[:,1]-q)/(bounds[:,1]-bounds[:,0]))
    assert np.all(np.min(distance,axis=1) < .02)
    lower = sum(x["side"] == "lower" for x in labels)
    assert lower == 500


def test_train_only_normalization_ignores_validation_and_test(valid_arrays):
    before = normalization(valid_arrays)
    changed = {k:v.copy() for k,v in valid_arrays.items()}
    changed["q"][changed["split"] != b"train"] += .1
    changed["position_m"][changed["split"] != b"train"] += 99
    assert before == normalization(changed)


def test_schema_validation_accepts_fixture(valid_arrays, schema, limits):
    validate_arrays(valid_arrays, schema, limits)


def test_canonical_hash_has_field_dtype_and_shape_domain_separation(valid_arrays, schema):
    order = [x["name"] for x in schema["fields"]]
    digest = canonical_array_hash(valid_arrays, order)
    changed = {k:v.copy() for k,v in valid_arrays.items()}; changed["q"][0,0] += 1e-12
    assert digest != canonical_array_hash(changed, order)
