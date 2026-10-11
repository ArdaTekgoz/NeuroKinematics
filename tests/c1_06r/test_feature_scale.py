from dataclasses import replace
import numpy as np
import pytest
from neurokinematics.neural import c106r_feature_scale as z


@pytest.fixture(scope='module')
def sample():
    rows=z.load_source('directions').take(np.arange(512));pose=z.r.relative_pose(rows)
    return rows,pose,z.fit_local(rows,pose)


def test_standardization_invertibility_and_residual_current(sample):
    rows,pose,scaler=sample;x=z.features(rows,'LOCAL_Z',{},scaler,pose)
    assert np.allclose(x[:,:7].mean(0),0,atol=1e-6)
    assert np.allclose(x[:,:7].std(0),1,atol=1e-6)
    assert np.allclose(x[:,:7]*scaler['std']+scaler['mean'],pose,atol=1e-8,rtol=1e-6)
    assert np.array_equal(x[:,-6:],rows.conditioned[:,-6:])


def test_probe_validation_and_constant_features_cannot_fit(sample):
    rows,pose,_=sample
    with pytest.raises(ValueError):z.fit_local(replace(rows,split='validation'),pose)
    with pytest.raises(ValueError):z.fit_local(z.load_source('probe').take(np.arange(512)),pose)
    with pytest.raises(ValueError):z.fit_local(rows,np.zeros_like(pose))


def test_scaling_has_no_teacher_label_dependency(sample):
    rows,pose,scaler=sample
    changed=replace(rows,q_target=np.full_like(rows.q_target,42),target_normalized=np.full_like(rows.target_normalized,42))
    assert np.array_equal(z.features(rows,'LOCAL_Z',{},scaler),z.features(changed,'LOCAL_Z',{},scaler))
    assert z.fit_local(changed,z.r.relative_pose(changed))==scaler


def test_raw_control_is_unchanged_and_probe_does_not_refit(sample):
    rows,pose,scaler=sample;norm=z.d.read_json(z.d.ROOT/'experiments/C1-06R/diagnostic3/normalization.json')
    assert np.array_equal(z.features(rows,'RAW',norm,scaler,pose),z.r.features(rows,'relative',norm,pose))
    probe=z.load_source('probe').take(np.arange(512));p=z.r.relative_pose(probe)
    output=z.features(probe,'LOCAL_Z',norm,scaler,p)
    assert np.array_equal(output[:,:7],((p-scaler['mean'])/scaler['std']).astype(np.float32))
