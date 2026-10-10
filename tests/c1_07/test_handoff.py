from copy import deepcopy
import json
import numpy as np
import pytest
from neurokinematics.neural import c107 as c


def fixture():
    m=json.loads((c.ROOT/'experiments/C1-07/preparation/handoff-manifest.json').read_text())
    w=json.loads((c.ROOT/'experiments/C1-07/closure/witness.json').read_text())
    return m,w['requests'][0]


@pytest.mark.parametrize('field,value',[('position_m',[float('nan'),0,0]),('position_m',[1,2]),('quaternion_wxyz',[0,0,0,0]),('quaternion_wxyz',[-1,0,0,0]),('q_current_rad',[100]*6)])
def test_invalid_request_rejected(field,value):
    _,request=fixture();request[field]=value
    with pytest.raises((ValueError,TypeError)):c.feature_inputs([request])


@pytest.mark.parametrize('key',['normalization','source_config','checkpoint'])
def test_artifact_sha_mutant_rejected_before_inference(key,tmp_path):
    manifest,request=fixture();candidate=deepcopy(manifest['candidates'][0]);candidate[key]['sha256']='0'*64
    # A corrupt artifact must be rejected even in a checkout without real weights.
    if key=='checkpoint':
        corrupt=tmp_path/'corrupt.pt';corrupt.write_bytes(b'not a checkpoint')
        candidate[key]['path']=str(corrupt)
    with pytest.raises(ValueError,match='SHA mismatch'):c.predict(candidate,[request])


def test_wrong_robot_rejected():
    m,_=fixture();m['robot']['hashes']['wrong']='0'*64
    with pytest.raises(ValueError,match='robot'):c.verify_manifest(m)


def test_invalid_prediction_cannot_pass_and_witness_target_is_valid():
    _,request=fixture();robot=c.r.load_robot();q=np.asarray(request['q_current_rad'])
    invalid=c.evaluate(np.array([[100.]*6,[float('nan')]*6]),[request,request])
    assert all(not x['valid'] and not x['profile_a'] and not x['profile_b'] for x in invalid)
    t=c.IndependentFK(robot).forward_kinematics(q)
    from neurokinematics.data.factory import canonical_quaternion
    zero=dict(request,position_m=t[:3,3].tolist(),quaternion_wxyz=canonical_quaternion(t[:3,:3]).tolist())
    assert c.evaluate(q[None],[zero])[0]['profile_b']


def test_teacher_metadata_cannot_change_features():
    _,request=fixture();altered=dict(request,q_target=[999]*6,split='test',family='invented',mode='other',pair_id='different')
    a=c.feature_inputs([request]);b=c.feature_inputs([altered]);assert np.array_equal(a.conditioned,b.conditioned)
