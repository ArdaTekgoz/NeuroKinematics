import json
from pathlib import Path

import numpy as np
import pytest

from neurokinematics.data.factory import FIELD_DTYPES


@pytest.fixture
def config():
    return json.loads(Path("experiments/F0-04/config.json").read_text(encoding="utf-8"))


@pytest.fixture
def schema():
    return json.loads(Path("experiments/F0-04/schema.json").read_text(encoding="utf-8"))


@pytest.fixture
def limits():
    from neurokinematics.kinematics.model import load_robot
    return load_robot().limits


@pytest.fixture
def valid_arrays(schema, limits):
    n = 6
    q = np.asarray([(np.asarray(limits)[:, 0] + np.asarray(limits)[:, 1]) / 2] * n, dtype="<f8")
    q[:, 0] += np.arange(n) * 1e-4
    values = {
        "robot_id": np.full(n, b"kuka_kr6_r900_sixx", dtype="|S32"),
        "model_hash": np.full(n, b"a"*64, dtype="|S64"), "tcp_hash": np.full(n, b"b"*64, dtype="|S64"),
        "sample_id": np.asarray([f"s-{i}" for i in range(n)], dtype="|S40"),
        "group_id": np.asarray([f"g-{i}" for i in range(n)], dtype="|S40"),
        "split": np.asarray(["train","train","train","train","validation","test"], dtype="|S10"),
        "q": q, "position_m": np.zeros((n,3), dtype="<f8"),
        "quaternion_wxyz": np.tile(np.array([1.,0.,0.,0.], dtype="<f8"), (n,1)),
        "sampling_class": np.full(n, b"main_lhs", dtype="|S16"),
        "sigma_min": np.ones(n, dtype="<f8"), "sigma_max": np.ones(n, dtype="<f8"),
        "condition": np.ones(n, dtype="<f8"), "manipulability": np.ones(n, dtype="<f8"),
        "numerical_rank": np.full(n, 6, dtype="|i1")}
    assert {k: v.dtype.str for k,v in values.items()} == FIELD_DTYPES
    return values
