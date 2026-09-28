"""T-F09 negative tests use real smoke artifacts and production validators."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from neurokinematics.foundations_gate import (EXPECTED_START, audit_dataset,
    require_same, sha, validate_gate, verify_checksums, verify_handoff)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'experiments/F0-06'


@pytest.fixture
def state():
    checks = ('historical_integrity', 'locked_environment', 'regression', 'dataset_determinism',
              'query_determinism', 'split_audit', 'normalization', 'fk_limits', 'benchmark_schema',
              'handoff_inputs', 'commands_complete')
    return {**dict.fromkeys(checks,'PASS'), 'start_commit':EXPECTED_START, 'critical_errors':[],
            'linux_execution':'NOT_RUN','linux_claim':'NOT_VERIFIED','g0':'PASS / ACCEPTED',
            'foundations':'COMPLETE','core':'READY / NOT_STARTED'}


def test_complete_gate(state):
    validate_gate(state)


@pytest.mark.parametrize('key,value', [
    ('start_commit','0'*40), ('regression','FAIL'), ('historical_integrity','FAIL'),
    ('dataset_determinism','FAIL'), ('query_determinism','FAIL'), ('split_audit','FAIL'),
    ('normalization','FAIL'), ('benchmark_schema','FAIL'), ('handoff_inputs','FAIL'),
    ('linux_claim','VERIFIED'), ('g0','FAIL / REJECTED'), ('commands_complete','NOT_RUN'),
    ('critical_errors',['unresolved']), ('locked_environment','FAIL')])
def test_gate_mutation_rejected(state,key,value,record_property):
    state[key] = value
    with pytest.raises(ValueError):
        validate_gate(state)
    record_property('mutation',key)
    record_property('detected',True)


@pytest.mark.parametrize('mutation',['corrupt','missing','duplicate','path_escape'])
def test_checksum_mutation(tmp_path,mutation,record_property):
    evidence = tmp_path/'old-evidence.json'
    evidence.write_bytes((ROOT/'experiments/F0-05/config.json').read_bytes())
    sums = tmp_path/'SHA256SUMS'
    sums.write_text(f'{sha(evidence)}  old-evidence.json\n',encoding='utf-8')
    assert verify_checksums(tmp_path,sums) == 1
    if mutation == 'corrupt': evidence.write_bytes(evidence.read_bytes()+b' ')
    elif mutation == 'missing': evidence.unlink()
    elif mutation == 'duplicate': sums.write_text(sums.read_text()*2)
    else: sums.write_text(f'{sha(evidence)}  ../old-evidence.json\n')
    with pytest.raises(ValueError): verify_checksums(tmp_path,sums)
    record_property('mutation','evidence_'+mutation)
    record_property('detected',True)


@pytest.mark.parametrize('field',['dataset_content_sha256','query_sha256'])
def test_determinism_mutation(field,record_property):
    a = json.loads((EVIDENCE/'reproduction/a/summary.json').read_bytes())
    b = json.loads((EVIDENCE/'reproduction/b/summary.json').read_bytes())
    require_same(a[field],b[field],field)
    b[field] = '0'*64
    with pytest.raises(ValueError): require_same(a[field],b[field],field)
    record_property('mutation',field)
    record_property('detected',True)


def test_real_data_audit_and_train_normalization_mutation(tmp_path,record_property):
    run = EVIDENCE/'reproduction/a'
    manifest = json.loads((run/'dataset/dataset-manifest.json').read_bytes())
    assert audit_dataset(run/'dataset',manifest,EVIDENCE/'data-config.json',run/'data-evidence')['status'] == 'PASS'
    normalization = json.loads((run/'data-evidence/normalization.json').read_bytes())
    normalization['source'] = 'validation and train'
    (tmp_path/'normalization.json').write_text(json.dumps(normalization))
    with pytest.raises(ValueError,match='normalization'):
        audit_dataset(run/'dataset',manifest,EVIDENCE/'data-config.json',tmp_path)
    record_property('mutation','normalization_nontrain_evidence')
    record_property('detected',True)


def test_real_split_leakage(tmp_path,record_property):
    from neurokinematics.data.factory import (read_shard,write_deterministic_npz,
        canonical_array_hash,_dataset_hash)
    run = EVIDENCE/'reproduction/a'
    shutil.copytree(run/'dataset',tmp_path/'dataset')
    manifest = json.loads((tmp_path/'dataset/dataset-manifest.json').read_bytes())
    schema = json.loads((ROOT/'experiments/F0-04/schema.json').read_bytes())
    order = [f['name'] for f in schema['fields']]
    shard = manifest['shards']['main'][0]
    path = tmp_path/'dataset'/shard['path']
    arrays = read_shard(path,order)
    train = next(i for i,x in enumerate(arrays['split']) if x == b'train')
    test = next(i for i,x in enumerate(arrays['split']) if x != b'train')
    arrays['group_id'][test] = arrays['group_id'][train]
    write_deterministic_npz(path,arrays,order)
    shard['file_sha256'] = sha(path)
    shard['content_sha256'] = canonical_array_hash(arrays,order)
    manifest['dataset_content_sha256'] = _dataset_hash(manifest['shards'])
    (tmp_path/'dataset/dataset-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): audit_dataset(tmp_path/'dataset',manifest,EVIDENCE/'data-config.json',run/'data-evidence')
    record_property('mutation','real_split_leakage_rehashed')
    record_property('detected',True)


def test_real_benchmark_schema_mutation(record_property):
    from neurokinematics.benchmark.contract import validate_record
    from neurokinematics.benchmark.validation import CandidateValidator
    from neurokinematics.kinematics import load_robot
    record = json.loads((EVIDENCE/'reproduction/a/results.jsonl').read_bytes().splitlines()[0])
    keys = ('query_list_sha256','solver_config_sha256','dataset_manifest_sha256')
    expected = {k:record[k] for k in keys}
    validator = CandidateValidator(load_robot())
    validate_record(record,validator=validator,expected_hashes=expected)
    del record['solver_status']
    with pytest.raises(ValueError): validate_record(record,validator=validator,expected_hashes=expected)
    record_property('mutation','real_benchmark_missing_schema_field')
    record_property('detected',True)


def test_missing_handoff(record_property):
    inputs = json.loads((EVIDENCE/'handoff-inputs.json').read_bytes())
    verify_handoff(ROOT,inputs)
    del inputs['pixi.lock']
    with pytest.raises(ValueError): verify_handoff(ROOT,inputs)
    record_property('mutation','missing_core_handoff_input')
    record_property('detected',True)


def test_smoke_config_cannot_relax_rules(tmp_path):
    from neurokinematics.benchmark.contract import load_reproduction_config
    config = json.loads((EVIDENCE/'benchmark-config.json').read_bytes())
    config['profiles']['B']['position_m'] = 1
    path = tmp_path/'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError): load_reproduction_config(path)
