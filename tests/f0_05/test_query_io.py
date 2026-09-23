"""Small production query/JSONL path with a substituted smoke count.

Full 10000/1000/1000 generation is reserved for the acceptance runner.
"""

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from neurokinematics.benchmark import queries,runner
from neurokinematics.benchmark.contract import load_frozen,strict_json


@pytest.fixture
def small_contract(monkeypatch):
    frozen=deepcopy(load_frozen())
    frozen['config.json']['query_counts']={'main':2,'boundary':2,'singularity':2}
    frozen['config.json']['singularity_sigma_min_le']=10.
    monkeypatch.setattr(queries,'load_frozen',lambda:frozen)
    monkeypatch.setattr(runner,'load_frozen',lambda:frozen)
    monkeypatch.setattr(queries,'f04_exclusions',lambda root:(set(),set(),
        {'dataset_content_sha256':frozen['config.json']['f0_04_dataset_content_sha256']}))
    return frozen


def test_small_query_generation_determinism_and_result_integrity(tmp_path,small_contract):
    dataset=tmp_path/'unused-f04';a=tmp_path/'a.jsonl';b=tmp_path/'b.jsonl'
    first=queries.generate_queries(a,dataset);second=queries.generate_queries(b,dataset)
    assert a.read_bytes()==b.read_bytes() and first['query_list_sha256']==second['query_list_sha256']
    verified=queries.verify_queries(a,dataset,first)
    assert verified['counts']=={'main':2,'boundary':2,'singularity':2}
    assert all(v==1 for v in verified['starts'].values())
    results=tmp_path/'results.jsonl';sample=tmp_path/'sample.jsonl'
    measured=runner.benchmark(a,results,sample,first,dataset)
    assert measured['record_count']==60
    checked=runner.verify_results(a,results,dataset,first,expected_file_hash=measured['result_file_sha256'])
    assert checked['status']=='PASS' and checked['groups']==measured['groups']
    assert measured['warmup_calls']==12
    assert checked['groups']['all/10ms/B']['unique_queries']==6
    assert checked['groups']['all/10ms/B']['attempts']==30


@pytest.mark.parametrize('fault',['truncate','reorder','broken_json','train_q','wrong_pose','bad_start','extra_row'])
def test_small_query_corruption_rejected(tmp_path,small_contract,monkeypatch,fault):
    dataset=tmp_path/'unused-f04';path=tmp_path/'q.jsonl'
    manifest=queries.generate_queries(path,dataset)
    rows=path.read_bytes().splitlines(keepends=True)
    if fault=='truncate':rows=rows[:-1]
    elif fault=='reorder':rows[0],rows[1]=rows[1],rows[0]
    elif fault=='broken_json':rows[0]=b'{"q":NaN}\n'
    elif fault=='train_q':
        q=strict_json(rows[0].decode())['q_target'];key=queries.q_key(q)
        monkeypatch.setattr(queries,'f04_exclusions',lambda root:({key},set(),{'dataset_content_sha256':'x'}))
    elif fault=='wrong_pose':
        row=strict_json(rows[0].decode());row['target_position_m'][0]+=.1;rows[0]=queries.encode_query(row)
    elif fault=='bad_start':
        row=strict_json(rows[0].decode());row['q_current']=row['q_target'];rows[0]=queries.encode_query(row)
    else:rows.append(rows[0])
    path.write_bytes(b''.join(rows))
    with pytest.raises(ValueError):queries.verify_queries(path,dataset,manifest)


@pytest.mark.parametrize('fault',['truncate','reorder','broken_json','wrong_hash','fake_success','late_pass','missing_iteration'])
def test_small_result_corruption_rejected(tmp_path,small_contract,fault):
    dataset=tmp_path/'unused-f04';query=tmp_path/'q.jsonl';manifest=queries.generate_queries(query,dataset)
    results=tmp_path/'r.jsonl';measured=runner.benchmark(query,results,tmp_path/'sample.jsonl',manifest,dataset)
    rows=results.read_bytes().splitlines(keepends=True)
    if fault=='truncate':rows=rows[:-1]
    elif fault=='reorder':rows[0],rows[1]=rows[1],rows[0]
    elif fault=='broken_json':rows[0]=b'{"x":NaN}\n'
    else:
        row=strict_json(rows[0].decode())
        if fault=='wrong_hash':row['query_list_sha256']='b'*64
        elif fault=='fake_success':row['profile_b_geometry']=not row['profile_b_geometry']
        elif fault=='late_pass':
            row['total_elapsed_ns']=100_000_000
            row['validation_elapsed_ns']=100_000_000-row['solve_elapsed_ns']
            row['profile_b_deadline']=True
        elif fault=='missing_iteration':row['iterations']=None
        rows[0]=queries.encode_query(row)
    results.write_bytes(b''.join(rows))
    with pytest.raises(ValueError):runner.verify_results(query,results,dataset,manifest)


def test_aggregation_includes_failures_and_no_fake_zero():
    group=runner.GroupStats()
    template={'query_group_id':'one','measurement_pass_index':0,'profile_b_geometry':False,
              'profile_b_deadline':False,'solver_status':'TIMEOUT','joint_limits':'PASS',
              'position_error_m':None,'orientation_error_deg':None,'total_elapsed_ns':50_000_000,
              'iterations':None}
    group.add(template,'b')
    success={**template,'query_group_id':'two','measurement_pass_index':1,
             'profile_b_geometry':True,'profile_b_deadline':True,'solver_status':'SUCCESS',
             'total_elapsed_ns':1_000_000,'iterations':3,'position_error_m':0.,'orientation_error_deg':0.}
    group.add(success,'b')
    summary=group.summarize()
    assert summary['unique_queries']==2 and summary['attempts']==2
    assert summary['all_query_latency_ns']['max']==50_000_000
    assert summary['deadline_success_latency_ns']['max']==1_000_000
    assert summary['iteration_missing']==1 and summary['iterations']['count']==1
    assert summary['status_counts']=={'TIMEOUT':1,'SUCCESS':1}
