"""End-to-end fault injections into the production benchmark record path."""

from copy import deepcopy
import hashlib

import pytest

from neurokinematics.benchmark import queries,runner
from neurokinematics.benchmark.contract import load_frozen


@pytest.fixture
def smoke(monkeypatch,tmp_path):
    frozen=deepcopy(load_frozen());frozen['config.json']['query_counts']={'main':2,'boundary':2,'singularity':2}
    frozen['config.json']['singularity_sigma_min_le']=10.
    monkeypatch.setattr(queries,'load_frozen',lambda:frozen)
    monkeypatch.setattr(runner,'load_frozen',lambda:frozen)
    monkeypatch.setattr(queries,'f04_exclusions',lambda root:(set(),set(),
        {'dataset_content_sha256':frozen['config.json']['f0_04_dataset_content_sha256']}))
    query=tmp_path/'query.jsonl';manifest=queries.generate_queries(query,tmp_path)
    return tmp_path,query,manifest


@pytest.mark.parametrize('mutation',[
    'q_target_leak','wrong_query_hash','wrong_solver_hash','wrong_dataset_hash',
    'wrong_frame','wrong_tcp','wrong_joint_order','geometry_flag','timeout_pass',
    'late_pass','missing_iteration','position_error','orientation_error'])
def test_record_mutation_rejected_by_production_gate(smoke,monkeypatch,mutation,record_property):
    root,query,manifest=smoke
    original=runner._result_record
    def corrupted(*args):
        row=original(*args)
        if mutation=='q_target_leak':row['q_target']=args[0]['q_target']
        elif mutation=='wrong_query_hash':row['query_list_sha256']='f'*64
        elif mutation=='wrong_solver_hash':row['solver_config_sha256']='f'*64
        elif mutation=='wrong_dataset_hash':row['dataset_manifest_sha256']='f'*64
        elif mutation=='wrong_frame':row['frame']='world'
        elif mutation=='wrong_tcp':row['tcp']='flange'
        elif mutation=='wrong_joint_order':row['joint_order']=row['joint_order'][::-1]
        elif mutation=='geometry_flag':row['profile_b_geometry']=not row['profile_b_geometry']
        elif mutation=='timeout_pass':row.update(solver_status='TIMEOUT',profile_b_deadline=True)
        elif mutation=='late_pass':
            row['total_elapsed_ns']=100_000_000
            row['validation_elapsed_ns']=100_000_000-row['solve_elapsed_ns']
            row['profile_b_deadline']=True
        elif mutation=='missing_iteration':row['iterations']=None
        elif mutation=='position_error':row['position_error_m']=999.
        else:row['orientation_error_rad']=3.
        return row
    monkeypatch.setattr(runner,'_result_record',corrupted)
    result=root/'result.jsonl';runner.benchmark(query,result,root/'sample.jsonl',manifest,root)
    # Test the production validator, not a comparison of two arrays.
    with pytest.raises(ValueError):runner.verify_results(query,result,root,manifest)
    record_property('mutation',mutation)
    record_property('detected_by_acceptance_assertion',True)


@pytest.mark.parametrize('mutation',['row_missing','row_reordered','json_corrupted','query_train_duplicate'])
def test_stream_or_train_mutation_rejected(smoke,mutation,record_property,monkeypatch):
    root,query,manifest=smoke
    if mutation=='query_train_duplicate':
        row=queries.strict_json(query.read_bytes().splitlines()[0].decode())
        key=queries.q_key(row['q_target'])
        monkeypatch.setattr(queries,'f04_exclusions',lambda p:({key},set(),{'dataset_content_sha256':'x'}))
        with pytest.raises(ValueError):queries.verify_queries(query,root,manifest)
    else:
        result=root/'result.jsonl';runner.benchmark(query,result,root/'sample.jsonl',manifest,root)
        lines=result.read_bytes().splitlines(keepends=True)
        if mutation=='row_missing':lines=lines[:-1]
        elif mutation=='row_reordered':lines[0],lines[1]=lines[1],lines[0]
        else:lines[0]=b'{"x":NaN}\n'
        result.write_bytes(b''.join(lines))
        with pytest.raises(ValueError):runner.verify_results(query,result,root,manifest)
    record_property('mutation',mutation)
    record_property('detected_by_acceptance_assertion',True)


def test_failed_query_latency_exclusion_detected(smoke,monkeypatch,record_property):
    root,query,manifest=smoke
    original=runner.GroupStats.add
    def mutated(self,row,profile):
        if not row[f'profile_{profile}_deadline']:
            return
        return original(self,row,profile)
    with monkeypatch.context() as patch:
        patch.setattr(runner.GroupStats,'add',mutated)
        result=root/'result.jsonl';measured=runner.benchmark(query,result,root/'sample.jsonl',manifest,root)
    verified=runner.verify_results(query,result,root,manifest)
    assert measured['groups']!=verified['groups']
    # The summarize CLI's acceptance comparison rejects this altered aggregate.
    assert measured['groups']['all/10ms/B']['attempts']<verified['groups']['all/10ms/B']['attempts']
    record_property('mutation','failed_query_latency_exclusion')
    record_property('detected_by_acceptance_assertion',True)
