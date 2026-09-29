"""C1-02 negative controls against real immutable roots and a synthetic pair."""

from copy import deepcopy

import numpy as np
import pytest

from neurokinematics.data.pair_validation import checked_input_projection, validate_one
from neurokinematics.data.pairs import load_contract, make_base, roots
from neurokinematics.kinematics.model import load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK


@pytest.fixture(scope="module")
def fixture_pair():
    config, _ = load_contract()
    root = roots()[0][0]
    inputs = load_robot()
    fk = PinocchioFK(inputs)
    bounds = np.asarray(inputs.limits)
    row = make_base(root, "local", config, bounds)
    row.update({"q_target": root["q"].copy(), "label_present": True,
                "teacher_status": "NOT_APPLICABLE", "teacher_failure_class": "NONE"})
    return row, root, inputs, fk


def test_positive_local_contract(fixture_pair):
    row, root, inputs, fk = fixture_pair
    validate_one(row, root, inputs, fk)
    assert list(checked_input_projection(row)) == ["position_m", "quaternion_wxyz", "q_current"]


@pytest.mark.parametrize("mutate", [
    lambda r, x: r.update(split="test" if r["split"] != "test" else "train"),
    lambda r, x: r.update(quaternion_wxyz=r["quaternion_wxyz"][[1, 2, 3, 0]]),
    lambda r, x: r["q_current"].__setitem__(0, x.limits[0][1] + .01),
    lambda r, x: r["q_current"].__setitem__(0, np.nan),
    lambda r, x: r.update(position_m=r["position_m"] + np.array([.01, 0, 0])),
])
def test_bad_pair_rejected(fixture_pair, mutate):
    row, root, inputs, fk = fixture_pair
    broken = deepcopy(row)
    mutate(broken, inputs)
    with pytest.raises(ValueError):
        validate_one(broken, root, inputs, fk)


def test_q_target_never_input(fixture_pair, monkeypatch):
    row, _, _, _ = fixture_pair
    import neurokinematics.data.pair_validation as validation
    config, schema = load_contract()
    config["normalization"]["input_fields"].append("q_target")
    monkeypatch.setattr(validation, "load_contract", lambda: (config, schema))
    with pytest.raises(ValueError, match="input leakage"):
        checked_input_projection(row)
