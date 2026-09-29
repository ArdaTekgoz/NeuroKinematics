"""R02: real C1-02 interfaces using tiny C1-03 synthetic roots, no shard reads."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from neurokinematics.kinematics.model import ROOT
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.data.pairs import load_contract, make_base
from neurokinematics.data.pair_validation import validate_one, checked_input_projection
from neurokinematics.data.factory import canonical_quaternion

@pytest.fixture(params=range(3))
def pair(request,robot):
    samples=[json.loads(s) for s in (ROOT/'experiments/C1-03/samples.jsonl').read_text().splitlines()]
    source=next(r for r in samples if r['id']==f'grad-{request.param:04d}')
    q=np.array(source['q64']);ref=PinocchioFK(robot);t=ref.reference_forward_kinematics(q)
    quat=canonical_quaternion(t[:3,:3])
    if quat[0]<0:quat=-quat
    root=dict(sample_id=f'c103-fixture-{request.param}',group_id=f'c103-fixture-{request.param}',split=['train','validation','test'][request.param],family='main',q=q,position=t[:3,3],quaternion=quat)
    config,_=load_contract();row=make_base(root,'local',config,np.array(robot.limits))
    row.update(q_target=q.copy(),label_present=True,teacher_status='NOT_APPLICABLE',teacher_failure_class='NONE')
    return root,row,ref

def test_local_and_unlabelled_wide(pair,robot,public_fk):
    root,row,ref=pair
    validate_one(row,root,robot,ref)
    assert list(checked_input_projection(row))==['position_m','quaternion_wxyz','q_current']
    config,_=load_contract();wide=make_base(root,'wide',config,np.array(robot.limits))
    wide.update(q_target=None,label_present=False,teacher_status='FAILED',teacher_failure_class='NO_VALID_CANDIDATE')
    validate_one(wide,root,robot,ref)
    for x in [row,wide]:
        value=public_fk(torch.tensor(x['q_current']),robot_id=robot.robot_id,joint_names=robot.joint_names).numpy()
        assert np.linalg.norm(value-ref.reference_forward_kinematics(x['q_current']))<=1e-9

@pytest.mark.parametrize('fault',['pose','split','nan','limit','quat_order'])
def test_invalid_pair(pair,robot,fault):
    root,row,ref=pair;row=deepcopy(row)
    if fault=='pose':row['position_m'][0]+=.01
    elif fault=='split':row['split']='other'
    elif fault=='nan':row['q_current'][0]=np.nan
    elif fault=='limit':row['q_current'][0]=robot.limits[0][1]+.01
    else:row['quaternion_wxyz']=row['quaternion_wxyz'][[1,2,3,0]]
    with pytest.raises(ValueError):validate_one(row,root,robot,ref)

def test_target_leakage(pair,monkeypatch):
    import neurokinematics.data.pair_validation as validation
    config,schema=load_contract();config['normalization']['input_fields'].append('q_target')
    monkeypatch.setattr(validation,'load_contract',lambda:(config,schema))
    with pytest.raises(ValueError,match='leakage'):checked_input_projection(pair[1])
