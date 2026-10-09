from dataclasses import replace
import numpy as np
import pytest
from neurokinematics.neural import c106r_diagnostic3 as r
from neurokinematics.kinematics.custom_fk import IndependentFK


@pytest.fixture(scope='module')
def rows():
    train,val = r.d.load_data(label_fk=False)
    return train.take(np.arange(32)),val.take(np.arange(32))


def test_relative_pose_recovers_target_with_independent_fk(rows):
    x=rows[0];rel=r.relative_pose(x)
    independent=IndependentFK(r.load_robot())
    for i,q in enumerate(x.q_current):
        t=independent.forward_kinematics(q)
        assert np.allclose(t[:3,3]+rel[i,:3],x.position[i],atol=1e-12,rtol=0)
        assert np.allclose(t[:3,:3]@r.quaternion_rotation(rel[i,3:]),r.quaternion_rotation(x.quaternion[i]),atol=1e-12,rtol=0)


def test_current_pose_yields_zero_displacement_and_identity_rotation(rows):
    x=rows[0];fk=r.PinocchioFK(r.load_robot())
    ts=[fk.reference_forward_kinematics(q) for q in x.q_current]
    same=replace(x,position=np.array([t[:3,3] for t in ts]),quaternion=np.array([r.canonical_quaternion(t[:3,:3]) for t in ts]))
    rel=r.relative_pose(same)
    assert np.allclose(rel[:,:3],0,atol=1e-15)
    assert np.allclose(rel[:,3:],[1,0,0,0],atol=1e-14)


def test_no_teacher_label_metadata_dependency_and_quaternion_sign_invariance(rows):
    x=rows[0];original=r.relative_pose(x)
    changed=replace(x,q_target=np.full_like(x.q_target,42),target_normalized=np.zeros_like(x.target_normalized),
                    family=np.full_like(x.family,'bad'),quaternion=-x.quaternion)
    assert np.array_equal(original,r.relative_pose(changed))


def test_train_only_fit_raw_identity_width_and_current_preservation(rows):
    train,val=rows;rel=r.relative_pose(train)
    norm=r.fit_normalization(train,rel)
    assert np.array_equal(r.features(train,'raw',norm),train.conditioned)
    f=r.features(train,'relative',norm,rel)
    assert f.shape==(32,13) and f.dtype==np.float32
    assert np.array_equal(f[:,-6:],train.conditioned[:,-6:])
    assert np.allclose(f[:,:3].mean(0),0,atol=1e-6)
    assert np.allclose(f[:,:3].std(0),1,atol=1e-6)
    with pytest.raises(ValueError):r.fit_normalization(val,r.relative_pose(val))
    with pytest.raises(ValueError):r.features(train,'unknown',norm)
