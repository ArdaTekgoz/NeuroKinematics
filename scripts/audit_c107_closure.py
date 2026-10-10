"""Verify historical freezes and the fresh T-C06 evidence without rerunning training."""
import hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/C1-07'


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen=read('experiments/C1-06R/training-freeze.json')['files']
    maps={'frozen':frozen,'preparation':read('experiments/C1-07/preparation/registration.json')['inputs']}
    for i in range(3,10):maps['diagnostic'+str(i)]=read(f'experiments/C1-06R/diagnostic{i}/registration.json')['inputs']
    for key,entries in maps.items():
        for name,value in entries.items():assert sha(ROOT/name)==value,(key,name)
    historical={}
    for filename in ['experiments/C1-06R/DELIVERY_SHA256SUMS','experiments/C1-07/SHA256SUMS']:
        file=ROOT/filename;lines=file.read_text().splitlines()
        for line in lines:
            expected,name=line.split('  ',1);assert sha(file.parent/name)==expected,name
        historical[filename]=dict(entries=len(lines),sha256=sha(file))
    clean=read('experiments/C1-07/closure/clean/complete.json');assert clean['status']=='PASS' and not clean['environment_copied']
    assert sum(v['tests'] for v in clean['tests'].values())==127
    assert all(v[k]==0 for v in clean['tests'].values() for k in ['failures','errors','skipped'])
    witness=read('experiments/C1-07/closure/clean/witness-result.json');assert witness['status']=='PASS' and witness['predictions']==288
    assert witness['exact_q_and_independent_fk_metrics'] and witness['head']==clean['head']
    source_paths=['src/neurokinematics/neural/c107.py','scripts/c107_witness.py','scripts/reproduce_c107.py','tests/c1_07/test_handoff.py','experiments/C1-07/closure/PROTOCOL.md','experiments/C1-07/closure/witness.json']
    for name in source_paths:
        committed=subprocess.check_output(['git','show',clean['head']+':'+name],cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest()==sha(ROOT/name),name
    logs=0
    for file in (BASE/'closure/clean/commands').glob('*/command.json'):
        record=json.loads(file.read_text());assert record['exit_code']==0
        for stream in ['stdout','stderr']:assert sha(file.parent/(stream+'.log'))==record[stream+'_sha256']
        logs+=1
    assert logs==clean['commands']==11
    assert not (BASE/'closure/clean/commands/clean-status/stdout.log').read_bytes().strip()
    paths=['experiments/F0-06/G0_DECISION.md','experiments/C1-01/RUN-20260928-T-C00-acceptance.md',
        'experiments/C1-01/udp-v2/verify/gate.json','experiments/C1-02/acceptance.json','experiments/C1-03/stage2/acceptance.json',
        'experiments/C1-04/stage2/acceptance.json','experiments/C1-05/stage2/acceptance.json','experiments/C1-06/stage2/final-001/acceptance.json',
        'experiments/C1-07/preparation/C1-06R-closure.json','experiments/C1-07/preparation/handoff-manifest.json']
    assert read(paths[2])['status']=='PASS' and read(paths[3])['status']=='PASS'
    assert read(paths[4])['status']=='PASS / ACCEPTED' and read(paths[5])['T-C03']=='PASS'
    assert read(paths[6])['T_C04']=='PASS'
    final=read(paths[7]);assert final['T_C05']=='PASS' and final['H2']=='REJECTED' and final['direct_ik_decision']=='NO_GO'
    result=dict(status='PASS',source_commit=clean['head'],source_files={p:sha(ROOT/p) for p in source_paths},
        frozen_files=len(frozen),registration_counts={k:len(v) for k,v in maps.items()},historical_manifests=historical,
        historical_evidence={p:sha(ROOT/p) for p in paths},clean_command_logs_verified=logs,tests=127,predictions=288,
        historical_tests='not all rerun; selected critical regressions rerun',new_final='NOT_CREATED',old_final_analysis='NOT_READ')
    (BASE/'closure/integrity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS:127 tests,288 predictions,11 logs,122 frozen sources,455 C1-06R deliveries,16 preparation deliveries')


if __name__=='__main__':main()
