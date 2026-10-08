"""Stage 2 harness tests using synthetic records, never final files."""
from collections import Counter
import copy
import json
import numpy as np
import pytest
from neurokinematics.neural import c106
from neurokinematics.neural import c106_runtime as rt
from neurokinematics.neural import c106_reporting as rp
from neurokinematics.data.factory import canonical_quaternion
from neurokinematics.core.contract import solver_config_hash


@pytest.fixture
def synthetic():
    v=c106.Validator(); q=(v.lower+v.upper)/2; t=v.fk.forward_kinematics(q)
    row=dict(query_id='f05-main-00000000',query_group_id='root-f05-main-00000000',subset='main',start_class='local',
             target_source='independent_frozen_fk',q_current=(q+.01).tolist(),q_target=q.tolist(),
             target_position_m=t[:3,3].tolist(),target_quaternion_wxyz=canonical_quaternion(t[:3,:3]).tolist())
    solver=dict(id='synthetic/default')
    cfg=dict(robot=dict(base_frame='base_link',tcp_frame='tool0',joint_order=list(v.robot.joint_names)))
    record={k:row[k] for k in ('query_id','query_group_id','subset','start_class','q_current','target_position_m','target_quaternion_wxyz')}
    record.update(query_list_sha256='qhash',dataset_manifest_sha256='dhash',solver_id=solver['id'],solver_config_sha256=solver_config_hash(solver),
                  frame='base_link',tcp='tool0',quaternion_order='wxyz',joint_order=list(v.robot.joint_names),collision='NOT_CHECKED',
                  deadline_profile_ms=10,measurement_pass_index=0,total_elapsed_ns=100,transport_elapsed_ns=80,validation_elapsed_ns=20)
    return v,row,solver,cfg,record


def test_synthetic_identity(synthetic):
    v,row,solver,cfg,record=synthetic
    count=Counter(); rt.check_query(row,0,count,v,{})
    assert count=={('main','local'):1}
    rt.check_baseline_binding(record,row,solver,cfg,'qhash','dhash')


@pytest.mark.parametrize('field,value',[('q_current',[0]*6),('target_position_m',[1,2,3]),('query_id','different'),
    ('query_group_id','different'),('query_list_sha256','wrong'),('dataset_manifest_sha256','wrong'),
    ('solver_config_sha256','wrong'),('frame','world'),('tcp','other'),('collision','PASS'),
    ('measurement_pass_index',5),('deadline_profile_ms',100),('total_elapsed_ns',99)])
def test_baseline_binding_rejects(synthetic,field,value):
    _,row,solver,cfg,record=synthetic; record[field]=value
    with pytest.raises(ValueError): rt.check_baseline_binding(record,row,solver,cfg,'qhash','dhash')


@pytest.mark.parametrize('field,value',[('target_source','teacher'),('start_class','wide'),('q_current',[999]*6),
    ('target_position_m',[9,9,9]),('query_id','wrong')])
def test_query_contract_rejects(synthetic,field,value):
    v,row,_,_,_=synthetic; row[field]=value
    with pytest.raises(ValueError): rt.check_query(row,0,Counter(),v,{})


def test_local_target_copy_rejected(synthetic):
    v,row,_,_,_=synthetic; row['q_current']=row['q_target']
    with pytest.raises(ValueError): rt.check_query(row,0,Counter(),v,{})


def test_report_full_denominator_and_success_time(synthetic):
    v,query,_,_,_=synthetic
    success=v.check(query['q_target'],query)
    fail=v.check([np.nan]*6,query)
    result=rp.breakdown([success,fail],np.array([[1,20_000_000]]*5))
    assert result['n']==2 and result['counts']['profile_a']==1 and result['rates']['profile_a']==.5
    assert result['geometry_missing']==1 and result['geometry_coverage']==.5
    assert result['all_time_ms']['n']==10 and result['successful_time_ms']['n']==5
    assert result['timeout_10ms']['rate']==.5
    empty=rp.breakdown([fail],np.array([[100]]*5))
    assert empty['successful_time_ms']['median'] is None
    assert empty['position_m_valid_only']['n']==0


def test_time_shape_rejected(synthetic):
    v,q,_,_,_=synthetic
    with pytest.raises(ValueError): rp.breakdown([v.check(q['q_target'],q)],[[1,2]])


def test_fractional_bootstrap_pairs_repeat_means():
    queries=[dict(query_id=f'{s}{i}',query_group_id=f'{s}{i}',subset=s) for s in c106.SUBSETS for i in range(4)]
    values=np.full((12,3,2),-.6)
    result=rp.fractional_bootstrap(values,queries,repeats=30)
    assert len(result)==2 and result[0]['unique_queries']==12
    assert result[0]['mean_over_fixed_seeds']['main']['difference']==pytest.approx(-.6)
    assert result[0]['mean_over_fixed_seeds']['hard_equal_weight']['ci95']==pytest.approx([-.6,-.6])
    queries[1]['query_group_id']=queries[0]['query_group_id']
    with pytest.raises(ValueError): rp.fractional_bootstrap(values,queries,repeats=10)


@pytest.mark.parametrize('fault',['hash','bytes','rows'])
def test_raw_manifest_corruption(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(rt,'ROOT',tmp_path)
    p=tmp_path/'fake.jsonl'; p.write_bytes(b'{}\n{}\n')
    item=dict(path='fake.jsonl',**rt.fingerprint(p,lines=True)); rt.check_file(item)
    item[fault if fault!='hash' else 'sha256']='0'*64 if fault=='hash' else item[fault]+1
    with pytest.raises(ValueError): rt.check_file(item)


def test_output_never_overwritten(tmp_path):
    p=tmp_path/'result.json'; rt.write(p,{'status':'first'})
    with pytest.raises(FileExistsError): rt.write(p,{'status':'second'})
    assert rt.read(p)['status']=='first'


def test_approval_required_before_opening(tmp_path,monkeypatch):
    p=tmp_path/'approval.json'; p.write_text(json.dumps({'user_message':'no approval'}))
    with pytest.raises(ValueError): rt.require_authorization(p)
