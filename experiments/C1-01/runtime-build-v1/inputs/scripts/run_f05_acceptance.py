"""F0-05 fail-fast acceptance. The previous experiment directories are read-only."""

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_HEAD='84adc24a03f619d7079e4e3900b8f51ab2591ef5'


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(path):
    path=(ROOT/Path(path)).resolve()
    path.relative_to(ROOT)
    return path


def preflight(output,dataset_root):
    from neurokinematics.benchmark.contract import load_frozen
    from neurokinematics.data.factory import verify_dataset
    from neurokinematics.kinematics.model import load_robot
    frozen=load_frozen();cfg=frozen['config.json'];load_robot()
    approval=json.loads((ROOT/'experiments/F0-05/stage2-approval.json').read_bytes())
    if not approval['approved'] or approval['approved_stage']!='Aşama 2':
        raise ValueError('missing explicit Stage 2 approval')
    if approval['approved_frozen_artifacts']!=json.loads((ROOT/'experiments/F0-05/stage1-frozen-hashes.json').read_bytes())['artifacts']:
        raise ValueError('approved frozen hashes changed')
    for name,expected in cfg['immutable_file_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'immutable input changed: {name}')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    if branch!='main' or head!=remote or head!=EXPECTED_HEAD:
        raise ValueError(f'unexpected Git head: {branch} {head} {remote}')
    manifest=ROOT/'experiments/F0-04/dataset-manifest.json'
    dataset=verify_dataset(dataset_root,manifest)
    if dataset['dataset_content_sha256']!=cfg['f0_04_dataset_content_sha256']:
        raise ValueError('F0-04 dataset content mismatch')
    result={'status':'PASS','head':head,'branch':branch,'origin_main':remote,
            'approved_frozen_artifacts':approval['approved_frozen_artifacts'],
            'immutable_file_sha256':cfg['immutable_file_sha256'],
            'dataset_verification':dataset,'word_lock_file':'IGNORED_BY_USER_APPROVAL'}
    put(output/'preflight.json',result)
    return result


def junit_details(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    bad=[x.attrib.get('name') for x in cases if any(x.find(t) is not None for t in ('failure','error','skipped'))]
    if not cases or bad:raise ValueError(f'incomplete JUnit {path}: {bad}')
    return len(cases),cases


def capture_mutations(paths):
    found=[]
    for path in paths:
        _,cases=junit_details(path)
        for case in cases:
            props={p.get('name'):p.get('value') for p in case.findall('properties/property')}
            if 'mutation' in props:found.append(props)
    if len(found)<14 or any(p.get('detected_by_acceptance_assertion',p.get('detected'))!='True' for p in found):
        raise ValueError('mutation evidence incomplete')
    return {'status':'PASS','detected_count':len(found),'cases':found}


def write_evidence_hashes(output):
    selected=[p for p in output.iterdir() if p.is_file() and p.suffix!='.md' and p.name!='SHA256SUMS']
    for folder in ('src/neurokinematics/solvers','src/neurokinematics/benchmark','tests/f0_05'):
        selected.extend((ROOT/folder).glob('*.py'))
    selected.extend(ROOT/p for p in ('pixi.toml','pixi.lock','scripts/run_f05_acceptance.py'))
    lines=[f'{digest(p)}  {p.relative_to(ROOT).as_posix()}' for p in sorted(set(selected))]
    (output/'SHA256SUMS').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def verify_evidence(output,generated_root):
    from neurokinematics.benchmark.queries import verify_queries
    lines=(output/'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    if not lines:raise ValueError('empty evidence checksum list')
    for line in lines:
        expected,relative=line.split('  ',1)
        path=within(relative)
        if digest(path)!=expected:raise ValueError(f'evidence checksum mismatch: {relative}')
    manifest=json.loads((output/'query-manifest.json').read_bytes())
    query=generated_root/'run-a/query-list.jsonl'
    if digest(query)!=manifest['query_list_sha256']:
        raise ValueError('query list content mismatch')
    measured=json.loads((output/'solver-summary.json').read_bytes())
    result_path=generated_root/'run-a/query_results.jsonl'
    if digest(result_path)!=measured['result_file_sha256']:
        raise ValueError('result file content mismatch')
    return {'status':'PASS','verified_file_count':len(lines),
            'query_list_sha256':manifest['query_list_sha256'],
            'result_file_sha256':measured['result_file_sha256']}


def failure_cases(output,results):
    from neurokinematics.benchmark.contract import strict_json
    from neurokinematics.benchmark.proof import classify_outer_target,verify_outer_reach
    cases=[];counts=Counter()
    with results.open('rb') as stream:
        for raw in stream:
            row=strict_json(raw.decode('utf-8'))
            status=row['solver_status'];counts[status]+=1
            if status!='SUCCESS' and counts[status]<=5:
                cases.append({k:row[k] for k in ('query_id','subset','start_class','deadline_profile_ms',
                    'measurement_pass_index','solver_status','validation_status','termination_reason',
                    'position_error_m','orientation_error_deg','iterations','total_elapsed_ns')})
    status,outer=classify_outer_target([100.,0.,0.])
    if status!='PROVEN_UNREACHABLE' or outer is None or not verify_outer_reach(*outer):
        raise ValueError('analytic outer proof failed')
    return {'status':'RECORDED','counts':dict(counts),'first_five_per_failure_status':cases,
            'analytic_outside_target':{'solver_status':'NOT_USED_FOR_PROOF',
                                        'validation_status':status.value,'reachability':status.value,
                                        'certificate':outer[0],'certificate_sha256':outer[1]},
            'invalid_input':{'status':'INVALID_INPUT','test':'T-F08 JUnit'},
            'unresolved_not_proven':{'status':'UNRESOLVED','test':'T-F08 JUnit'}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('experiments/F0-05'))
    parser.add_argument('--generated-root',type=Path,default=Path('data/generated/F0-05/acceptance'))
    parser.add_argument('--dataset-root',type=Path,default=Path('data/generated/F0-04/run-a'))
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    output,generated,dataset=map(within,(args.output,args.generated_root,args.dataset_root))
    if output==generated or output==dataset or generated==dataset:
        raise ValueError('output, generated-root and dataset-root must differ')
    if args.verify_only:
        print(json.dumps(verify_evidence(output,generated),ensure_ascii=False));return 0
    output.mkdir(parents=True,exist_ok=True);generated.mkdir(parents=True,exist_ok=True)
    commands=[];environment={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1'}
    try:
        preflight(output,dataset)
        def run(label,command):
            print('$ '+' '.join(command),flush=True)
            started=datetime.now(timezone.utc).isoformat();lines=[]
            with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace',env=environment,bufsize=1) as process:
                for line in process.stdout:
                    print(line,end='',flush=True);lines.append(line)
                code=process.wait()
            commands.append({'label':label,'command':command,'started_utc':started,
                'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code,'output':''.join(lines)})
            put(output/'commands.json',commands)
            if code:raise ValueError(f'{label} failed with exit {code}')
        def junit(name):return f'--junitxml={output.relative_to(ROOT).as_posix()}/{name}-junit.xml'
        run('lock-check',['pixi','lock','--check'])
        run('install',['pixi','install','--locked'])
        for name,task in (('f00','test-f00'),('f01','test-f01'),('f02','test-f02'),
                          ('f03','test-f03'),('f04','test-f04')):
            run(name,['pixi','run','--locked',task,junit(name),'-o','junit_family=legacy'])
        run('f05-unit',['pixi','run','--locked','python','-m','pytest','-q',
            'tests/f0_05/test_math.py','tests/f0_05/test_contract.py','tests/f0_05/test_query_io.py',
            junit('f05-unit'),'-o','junit_family=legacy'])
        run('tf08',['pixi','run','--locked','python','-m','pytest','-q',
            'tests/f0_05/test_tf08.py',junit('tf08'),'-o','junit_family=legacy'])
        run('mutations',['pixi','run','--locked','python','-m','pytest','-q',
            'tests/f0_05/test_mutations.py','tests/f0_05/test_stage2_mutations.py',
            junit('mutation'),'-o','junit_family=legacy'])
        qa=generated/'run-a/query-list.jsonl';qb=generated/'run-b/query-list.jsonl'
        if qa.exists() or qb.exists():raise ValueError('query output exists; choose fresh --generated-root')
        rel=lambda p:p.relative_to(ROOT).as_posix()
        run('query-a',['pixi','run','--locked','generate-f05','--output',rel(qa),
                       '--dataset-root',rel(dataset),'--manifest-out',rel(output/'query-manifest.json')])
        run('query-b',['pixi','run','--locked','generate-f05','--output',rel(qb),
                       '--dataset-root',rel(dataset)])
        ma=json.loads((output/'query-manifest.json').read_bytes())
        if digest(qa)!=digest(qb) or digest(qa)!=ma['query_list_sha256']:
            raise ValueError('independent query generation differs')
        run('verify-queries',['pixi','run','--locked','verify-f05-queries','--input',rel(qa),
            '--dataset-root',rel(dataset),'--manifest',rel(output/'query-manifest.json')])
        from neurokinematics.benchmark.queries import verify_queries
        query_audit=verify_queries(qa,dataset,ma)
        put(output/'query-hashes.json',{'status':'PASS','run_a_sha256':digest(qa),
            'run_b_sha256':digest(qb),'content_match':True,'audit':query_audit})
        results=generated/'run-a/query_results.jsonl'
        run('benchmark',['pixi','run','--locked','benchmark-f05','--queries',rel(qa),
            '--dataset-root',rel(dataset),'--manifest',rel(output/'query-manifest.json'),
            '--results',rel(results),'--sample',rel(output/'query-results-sample.jsonl'),
            '--summary-out',rel(output/'solver-summary.json')])
        measured=json.loads((output/'solver-summary.json').read_bytes())
        run('verify-results',['pixi','run','--locked','verify-f05-results','--queries',rel(qa),
            '--results',rel(results),'--dataset-root',rel(dataset),
            '--manifest',rel(output/'query-manifest.json'),
            '--expected-sha256',measured['result_file_sha256'],
            '--summary-out',rel(output/'result-verification.json')])
        run('summarize',['pixi','run','--locked','summarize-f05',
            '--verified-summary',rel(output/'result-verification.json'),
            '--measurement-summary',rel(output/'solver-summary.json'),
            '--output',rel(output/'benchmark-summary.json')])
        verified=json.loads((output/'result-verification.json').read_bytes())
        groups=verified['groups']
        put(output/'subgroup-summary.json',{k:v for k,v in groups.items() if k.count('/')==3})
        put(output/'deadline-summary.json',{k:v for k,v in groups.items() if k.startswith('all/')})
        put(output/'failure-cases.json',failure_cases(output,results))
        put(output/'mutation-results.json',capture_mutations([output/'mutation-junit.xml']))
        put(output/'environment.json',measured['environment'])
        counts={name:junit_details(output/f'{name}-junit.xml')[0] for name in
            ('f00','f01','f02','f03','f04','f05-unit','tf08','mutation')}
        put(output/'acceptance.json',{'status':'PASS','test_counts':counts,
            'query_list_sha256':ma['query_list_sha256'],'result_file_sha256':verified['result_file_sha256'],
            'record_count':verified['record_count'],'command_count':len(commands),
            'finished_utc':datetime.now(timezone.utc).isoformat()})
        write_evidence_hashes(output)
        evidence=verify_evidence(output,generated)
        put(output/'evidence-verification.json',evidence)
        write_evidence_hashes(output)
        evidence=verify_evidence(output,generated)
        print('F0-05 acceptance PASS; evidence:',json.dumps(evidence),flush=True)
        return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError,ET.ParseError) as exc:
        put(output/'acceptance.json',{'status':'FAIL','reason':str(exc),
            'completed_commands':len(commands),'finished_utc':datetime.now(timezone.utc).isoformat()})
        print(f'F0-05 acceptance FAIL: {exc}',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
