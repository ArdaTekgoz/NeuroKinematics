import numpy as np
import pytest

from neurokinematics.data.factory import canonical_quaternion, coverage, validate_arrays
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.model import load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK


def test_reference_and_independent_fk_on_fixed_samples(limits):
    inputs = load_robot(); reference, independent = PinocchioFK(inputs), IndependentFK(inputs)
    bounds = np.asarray(limits); samples = np.vstack(((bounds[:,0]+bounds[:,1])/2, bounds[:,0], bounds[:,1]))
    for q in samples:
        a,b = reference.reference_forward_kinematics(q), independent.forward_kinematics(q)
        assert np.linalg.norm(a[:3,3]-b[:3,3]) <= 1e-9
        assert np.linalg.norm(a[:3,:3]-b[:3,:3], ord="fro") <= 1e-9
        assert np.allclose(__import__("neurokinematics.kinematics.metrics", fromlist=["quaternion_rotation"]).quaternion_rotation(canonical_quaternion(a[:3,:3])), a[:3,:3], atol=1e-12)


def test_coverage_is_quaternion_sign_invariant(limits):
    q = np.zeros((2,6)); positions=np.zeros((2,3)); quaternions=np.array([[1.,0,0,0],[-1.,0,0,0]])
    profile={"joint_space_bin_width_normalized":.1,"position_voxel_size_m":.05,"orientation_matrix_step":.25}
    result=coverage(q,positions,quaternions,limits,profile,[2])
    assert result["orientation_bin_occupancy"] == 1


@pytest.mark.parametrize("field", ["q","position_m","quaternion_wxyz","sigma_min"])
def test_nan_rejected(field, valid_arrays, schema, limits):
    changed={k:v.copy() for k,v in valid_arrays.items()}; changed[field].flat[0]=np.nan
    with pytest.raises(ValueError): validate_arrays(changed,schema,limits)


def test_condition_inf_only_for_exact_zero(valid_arrays,schema,limits):
    changed={k:v.copy() for k,v in valid_arrays.items()}; changed["condition"][0]=np.inf
    with pytest.raises(ValueError, match="Inf policy"): validate_arrays(changed,schema,limits)
