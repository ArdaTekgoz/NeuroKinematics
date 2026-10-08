"""Authorized C1-06 final campaign. Stage-1 primitives remain immutable."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from neurokinematics.kinematics.model import ROOT, load_robot, validate_q
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error, singularity_metrics
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.data.factory import read_shard
from neurokinematics.neural import c104, c105, c106
from neurokinematics.neural.physics import normalized_head

BASE = ROOT / 'experiments/C1-06'
STAGE2 = BASE / 'stage2'
STAGE1_COMMIT = '075478545a30373db2a4ac64434f2db678df050e'
STAGE1_SHA = '12cea1ed3d4419162aaebc5beca5a190583c7f24376aadd9d28f9368f752f7c0'
SOURCES = ['src/neurokinematics/neural/c106_runtime.py',
           'src/neurokinematics/neural/c106_reporting.py', 'scripts/run_c106.py',
           'tests/c1_06/test_runtime.py']


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path, *, lines=False):
    h = hashlib.sha256(); size = count = 0
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block); size += len(block)
            if lines: count += block.count(b'\n')
    result = dict(sha256=h.hexdigest(), bytes=size)
    if lines: result['rows'] = count
    return result


def file_record(path, *, lines=False):
    return dict(path=Path(path).relative_to(ROOT).as_posix(), **fingerprint(path, lines=lines),
                accessible=True, storage='LOCAL_ONLY' if 'data/generated' in Path(path).as_posix() else 'GIT',
                remote_archive='NOT_CONFIRMED')


def check_file(record):
    path = (ROOT / record['path']).resolve()
    if not path.is_relative_to(ROOT.resolve()): raise ValueError('outside root')
    actual = fingerprint(path, lines='rows' in record)
    if any(record[k] != v for k, v in actual.items()): raise ValueError('file drift: ' + record['path'])


def config(path=BASE/'config.json'):
    return read(path)


def require_authorization(approval=STAGE2/'approval.json'):
    a = read(approval)
    if (a.get('user_message') != 'Onaylıyorum' or a.get('stage1_commit') != STAGE1_COMMIT or
            a.get('stage1_sha256sums_sha256') != STAGE1_SHA or c104.sha(BASE/'SHA256SUMS') != STAGE1_SHA):
        raise ValueError('approval/stage1 binding mismatch')
    for line in (BASE/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, rel = line.split('  ', 1)
        if c104.sha(ROOT/rel) != expected: raise ValueError('Stage1 drift: ' + rel)
    return a


def require_runtime():
    require_authorization()
    p = read(STAGE2/'preflight.json')
    if p['status'] != 'PASS': raise ValueError('preflight required')
    for record in p['source_files']: check_file(record)
    return p


def setup():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1': raise ValueError('thread environment: ' + name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def model_key(cp):
    return f"{cp['experiment']}-{cp['variant']}-seed-{cp['seed']}"


def load_model(cp, validator):
    check_file(cp)
    if cp['experiment'] == 'E-C01':
        model, meta = c104.load_checkpoint(ROOT/cp['path'], expected_variant='conditioned')
        if meta.get('seed', meta.get('training_seed')) != cp['seed']:
            raise ValueError('C104 checkpoint seed')
    else:
        model, meta = c105.load_checkpoint(ROOT/cp['path'])
        if (meta['variant'], meta['experiment'], meta['training_seed'], meta['selected_epoch']) != (
                cp['variant'], cp['experiment'], cp['seed'], cp['epoch']):
            raise ValueError('C105 checkpoint identity')

    def batch(x):
        logits = model(torch.from_numpy(np.asarray(x, dtype=np.float32)))
        z = logits if cp['experiment'] == 'E-C01' else normalized_head(logits, cp['variant'])
        return validator.lower + z.detach().numpy().astype(np.float64) * (validator.upper-validator.lower)

    def single(x):
        return batch(x[None, :])[0]
    return model, meta, single, batch


def preflight():
    require_authorization()
    validator = c106.Validator()
    audit = read(BASE/'input-hashes.json')
    for item in audit['files']: check_file(item)
    witnesses = []
    norm = read(ROOT/'experiments/C1-02/normalization.json')
    w4 = read(ROOT/'experiments/C1-04/stage2/fixed-validation-inference.json')
    w5 = read(ROOT/'experiments/C1-05/stage2/fixed-validation-witness.json')
    fk = PinocchioFK(validator.robot)
    with torch.inference_mode():
        for cp in config()['checkpoints']:
            model, metadata, single, batch = load_model(cp, validator)
            if cp['experiment'] == 'E-C01':
                expected = w4['expected'][f"{cp['seed']}/conditioned"]
                values = []
                for sample in w4['samples']:
                    query = dict(target_position_m=sample['position_m'], target_quaternion_wxyz=sample['quaternion_wxyz'], q_current=sample['q_current'])
                    values.append(single(c106.features(query, norm, validator.robot.limits)))
                q = np.asarray(values); old = np.asarray([v['q_raw_rad'] for v in expected])
                poses = [None if not e['in_limits'] else (e['fk_position_m'], e['fk_rotation']) for e in expected]
            else:
                old_record = next(r for r in w5['records'] if (r['seed'],r['variant'],r['experiment']) == (cp['seed'],cp['variant'],cp['experiment']))
                q = batch(np.asarray(w5['features'], dtype=np.float32)); old = np.asarray(old_record['q_rad'])
                poses = [None if t is None else (np.asarray(t)[:3,3],np.asarray(t)[:3,:3]) for t in old_record['poses']]
                # Single-query fast wrapper must equal the original function on identical shape.
                for x in np.asarray(w5['features'], dtype=np.float32):
                    original, _ = c105.infer(model, cp['variant'], x[None, :])
                    if not np.array_equal(single(x), original[0]): raise ValueError('single witness changed')
            if not np.array_equal(q, old): raise ValueError('fixed q witness changed: ' + model_key(cp))
            worst = 0.
            for value, pose in zip(q, poses):
                if pose is None: continue
                independent = validator.fk.forward_kinematics(value)
                reference = fk.reference_forward_kinematics(value)
                worst = max(worst,float(np.max(np.abs(independent-reference))))
                if not np.array_equal(reference[:3,3],pose[0]) or not np.array_equal(reference[:3,:3],pose[1]):
                    raise ValueError('historical FK witness changed')
            if worst > 1e-9: raise ValueError('independent witness FK')
            witnesses.append(dict(model=model_key(cp), samples=10, max_q_delta=0, max_independent_fk_delta=worst))
    # Unit suite has no final input dependency. Require its latest complete evidence.
    import xml.etree.ElementTree as ET
    root = ET.parse(STAGE2/'runtime-tests.xml').getroot()
    suites = list(root.iter('testsuite'))
    if not suites or any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped')):
        raise ValueError('runtime negative tests did not pass')
    write(STAGE2/'preflight.json', dict(status='PASS', time_utc=now(), witnesses=witnesses,
        synthetic_tests=sum(int(s.get('tests',0)) for s in suites),
        source_files=[file_record(ROOT/rel) for rel in SOURCES] + [file_record(STAGE2/'runtime-tests.xml')],
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
                         processor=platform.processor(),torch_threads=torch.get_num_threads(),
                         torch_interop_threads=torch.get_num_interop_threads(),gpu='NOT_USED'),
        legacy_c104_wrapper='old global .gitattributes snapshot conflicts with later evidence rules; fixed witnesses and substantive inputs verified directly',
        final_test='SEALED_NOT_RUN'))
    print(json.dumps(dict(status='PASS', checkpoints=len(witnesses), samples_each=10)), flush=True)


def check_query(row, index, counts, validator, frozen):
    subset = 'main' if index < 10000 else 'boundary' if index < 11000 else 'singularity'
    ordinal = index if subset == 'main' else index-10000 if subset == 'boundary' else index-11000
    if (row['query_id'],row['query_group_id'],row['subset'],row['start_class']) != (
            f'f05-{subset}-{ordinal:08d}',f'root-f05-{subset}-{ordinal:08d}',subset,'local' if ordinal%2==0 else 'wide'):
        raise ValueError('query order/identity/mode')
    target = validate_q(row['q_target'],validator.robot.joint_names,validator.robot.limits)
    current = validate_q(row['q_current'],validator.robot.joint_names,validator.robot.limits)
    if row['target_source'] != 'independent_frozen_fk': raise ValueError('source provenance')
    if row['start_class']=='local':
        if np.array_equal(target,current) or np.max(np.abs(target-current)) > .05+1e-14: raise ValueError('local provenance')
    elif np.sqrt(np.mean(((current-target)/(validator.upper-validator.lower))**2)) < .25:
        raise ValueError('wide provenance')
    pose = validator.fk.forward_kinematics(target)
    if np.linalg.norm(pose[:3,3]-row['target_position_m']) > 1e-9 or np.linalg.norm(pose[:3,:3]-quaternion_rotation(row['target_quaternion_wxyz'])) > 1e-9:
        raise ValueError('target FK')
    counts[(subset,row['start_class'])] += 1


def check_baseline_binding(record, query, solver, frozen, query_hash, dataset_hash):
    for name in ('query_id','query_group_id','subset','start_class','q_current','target_position_m','target_quaternion_wxyz'):
        if record[name] != query[name]: raise ValueError('baseline query binding: ' + name)
    if record['query_list_sha256'] != query_hash or record['dataset_manifest_sha256'] != dataset_hash:
        raise ValueError('baseline manifest binding')
    from neurokinematics.core.contract import solver_config_hash
    if record['solver_id'] != solver['id'] or record['solver_config_sha256'] != solver_config_hash(solver):
        raise ValueError('baseline solver binding')
    if (record['frame'],record['tcp'],record['quaternion_order'],record['joint_order']) != (
            frozen['robot']['base_frame'],frozen['robot']['tcp_frame'],'wxyz',frozen['robot']['joint_order']):
        raise ValueError('baseline robot/frame binding')
    if record['collision'] != 'NOT_CHECKED': raise ValueError('baseline collision claim')
    if record['deadline_profile_ms'] not in (10,50) or record['measurement_pass_index'] not in range(5):
        raise ValueError('baseline repeat/deadline')
    if record['total_elapsed_ns'] != record['transport_elapsed_ns'] + record['validation_elapsed_ns']:
        raise ValueError('baseline timing')


def campaign_paths(run):
    if not run.replace('-','').isalnum(): raise ValueError('run name')
    return STAGE2/run, ROOT/'data/generated/C1-06'/run


def identity(run):
    require_runtime()
    out, raw_root = campaign_paths(run)
    out.mkdir(parents=True, exist_ok=False); raw_root.mkdir(parents=True, exist_ok=False)
    write(out/'opening.json',dict(time_utc=now(),status='AUTHORIZED_ONE_WAY_OPENING',config_sha256=c104.sha(BASE/'config.json')))
    cfg=config(); validator=c106.Validator(); norm=read(ROOT/'experiments/C1-02/normalization.json')
    path=ROOT/cfg['query_file']
    if c104.sha(path)!=cfg['query_sha256']: raise ValueError('query bytes changed')
    rows=[json.loads(line) for line in path.open(encoding='utf-8')]
    if len(rows)!=12000: raise ValueError('query count')
    counts=Counter(); targets=set(); positions=set(); groups=set(); queries={}
    jac=IndependentJacobian(validator.robot); diagnostics=[]
    for i,row in enumerate(rows):
        check_query(row,i,counts,validator,cfg)
        qkey=np.asarray(row['q_target'],dtype='<f8').tobytes()
        pkey=np.r_[row['target_position_m'],row['target_quaternion_wxyz']].astype('<f8').tobytes()
        if qkey in targets or row['query_group_id'] in groups or row['query_id'] in queries: raise ValueError('duplicate root/query')
        targets.add(qkey); positions.add(pkey); groups.add(row['query_group_id']); queries[row['query_id']]=row
        q=np.asarray(row['q_target']); dist=float(np.min(np.minimum(q-validator.lower,validator.upper-q)/(validator.upper-validator.lower)))
        sigma=singularity_metrics(jac.jacobian(q),.9015)['sigma_min']
        if row['subset']=='boundary' and dist>=.02: raise ValueError('boundary definition')
        if row['subset']=='singularity' and sigma>0.00727741160967353: raise ValueError('singularity definition')
        c106.features(row,norm,validator.robot.limits)
        diagnostics.append(dict(query_id=row['query_id'],group_id=row['query_group_id'],subset=row['subset'],mode=row['start_class'],
            near_limit=dist<.02,normalized_limit_distance=dist,sigma_min=sigma,singular=sigma<=0.00727741160967353,
            position_shift=bool(np.any(np.abs((np.asarray(row['target_position_m'])-norm['position_mean_m'])/norm['position_std_m'])>3))))
    if dict(counts)!={(s,m):n//2 for s,n in cfg['counts'].items() for m in ('local','wide')}: raise ValueError('counts')
    # Train/validation only, checking provenance roots rather than chosen teacher labels.
    schema=read(ROOT/'experiments/C1-02/schema.json'); order=[f['name'] for f in schema['fields']]
    manifest=read(ROOT/'experiments/C1-02/dataset-manifest.json'); old_p=[]; old_quat=[]; train_rows=0
    for shard in manifest['shards']:
        if '-test-' in shard['path']: continue
        arrays=read_shard(ROOT/'data/generated/C1-02/v1'/shard['path'],order)
        for q,p,quat,g in zip(arrays['root_q_target'],arrays['position_m'],arrays['quaternion_wxyz'],arrays['group_id']):
            if q.astype('<f8').tobytes() in targets or np.r_[p,quat].astype('<f8').tobytes() in positions or g.decode() in groups:
                raise ValueError('training/validation source overlap')
        old_p.extend(arrays['position_m']); old_quat.extend(arrays['quaternion_wxyz']); train_rows+=len(arrays['pair_id'])
    tree=cKDTree(old_p); near=0
    for row in rows:
        for idx in tree.query_ball_point(row['target_position_m'],r=.002):
            near+=1
            if math.degrees(rotation_error(quaternion_rotation(row['target_quaternion_wxyz']),quaternion_rotation(old_quat[idx])))<=1:
                raise ValueError('near pose training/validation leakage')
    write(raw_root/'query-diagnostics.json',diagnostics)
    baseline_cfg=read(ROOT/'experiments/C1-01/baseline-config.json')
    baseline_manifest=read(BASE/'input-hashes.json')['baseline']; baseline_results=[]
    dataset_hash=c104.sha(ROOT/'experiments/F0-04/dataset-manifest.json')
    for entry in baseline_manifest:
        check_file(entry)
        solver=next(s for s in baseline_cfg['solvers'] if s['id']==entry['method'])
        seen=set(); differences=Counter(); worst_position=worst_angle=0.
        derived=raw_root/(entry['method'].replace('/','-')+'-verified.jsonl')
        with derived.open('x',encoding='utf-8',newline='\n') as stream, (ROOT/entry['path']).open(encoding='utf-8') as source:
            for i,line in enumerate(source):
                record=json.loads(line); query=queries[record['query_id']]
                check_baseline_binding(record,query,solver,baseline_cfg,cfg['query_sha256'],dataset_hash)
                expected=(10 if i<60000 else 50,(i%60000)//12000,rows[i%12000]['query_id'])
                key=(record['deadline_profile_ms'],record['measurement_pass_index'],record['query_id'])
                if key!=expected or key in seen: raise ValueError('baseline order/duplicate')
                seen.add(key)
                q=record['q_candidate']
                verdict=validator.check(q if q is not None else [],query)
                if any(verdict['profile_'+p]!=record['profile_'+p+'_geometry'] for p in ('a','b')):
                    raise ValueError('baseline independent geometry category changed')
                if verdict['in_limits'] != (record['joint_limits']=='PASS'): raise ValueError('baseline limits category changed')
                # Cross-platform numeric values retained separately; categorical agreement is exact.
                if verdict['position_error_m'] is not None:
                    worst_position=max(worst_position,abs(verdict['position_error_m']-record['position_error_m']))
                    worst_angle=max(worst_angle,abs(verdict['orientation_error_deg']-record['orientation_error_deg']))
                differences[record['common_status']]+=1
                row=dict(query_id=record['query_id'],group_id=record['query_group_id'],subset=record['subset'],mode=record['start_class'],
                    method=entry['method'],deadline_ms=key[0],pass_index=key[1],**verdict,
                    no_candidate=q is None,timeout=record['timed_out'],status=record['common_status'],
                    elapsed_ns=record['total_elapsed_ns'],profile_a_deadline=record['profile_a_deadline'],profile_b_deadline=record['profile_b_deadline'],
                    label_present='NOT_APPLICABLE',teacher_status='NOT_APPLICABLE')
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if (i+1)%24000==0: print(json.dumps(dict(stage='baseline-identity',method=entry['method'],rows=i+1)),flush=True)
        if len(seen)!=120000: raise ValueError('baseline missing observation')
        baseline_results.append(dict(method=entry['method'],source=entry,derived=file_record(derived,lines=True),
            statuses=dict(differences),max_cross_platform_position_delta_m=worst_position,
            max_cross_platform_orientation_delta_deg=worst_angle,geometry_category_agreement='EXACT'))
    write(out/'identity.json',dict(status='PASS',time_utc=now(),query_count=len(rows),unique_roots=len(groups),
        counts={f'{s}/{m}':n for (s,m),n in counts.items()},train_validation_rows_checked=train_rows,
        near_position_pairs_checked=near,leakage_count=0,C102_test='SEALED; prior accepted hash-bound split audit',
        query=file_record(path,lines=True),diagnostics=file_record(raw_root/'query-diagnostics.json'),baselines=baseline_results))
    print(json.dumps(dict(status='PASS',stage='identity',queries=12000,baseline_rows=600000)),flush=True)


def evaluate(run):
    require_runtime(); out,raw_root=campaign_paths(run)
    identity_record=read(out/'identity.json')
    if identity_record['status']!='PASS': raise ValueError('identity gate')
    check_file(identity_record['query']); check_file(identity_record['diagnostics'])
    cfg=config(); queries=[json.loads(line) for line in (ROOT/cfg['query_file']).open(encoding='utf-8')]
    norm=read(ROOT/'experiments/C1-02/normalization.json'); validator=c106.Validator()
    warm=np.asarray(read(ROOT/'experiments/C1-05/stage2/fixed-validation-witness.json')['features'],dtype=np.float32)
    results=[]; total_start=time.perf_counter(); completed=0; last=None
    write(out/'evaluation-start.json',dict(time_utc=now(),status='RUNNING',config_sha256=c104.sha(BASE/'config.json')))
    try:
        with torch.inference_mode():
            for cp in cfg['checkpoints']:
                key=model_key(cp); start=time.perf_counter(); _,_,predict,_=load_model(cp,validator)
                load_s=time.perf_counter()-start; start=time.perf_counter()
                for i in range(20): predict(warm[i%10])
                warm_s=time.perf_counter()-start; destination=raw_root/(key+'.jsonl')
                first=[]; start=time.perf_counter()
                with destination.open('x',encoding='utf-8',newline='\n') as stream:
                    for repeat in range(5):
                        for i,query in enumerate(queries):
                            last=dict(model=key,pass_index=repeat,query_id=query['query_id'])
                            tick=time.perf_counter_ns()
                            x=c106.features(query,norm,validator.robot.limits); q=predict(x)
                            verdict=validator.check(q,query); elapsed=time.perf_counter_ns()-tick
                            qraw=q.tolist() if np.isfinite(q).all() else [float(v) if np.isfinite(v) else None for v in q]
                            semantic=(qraw,verdict)
                            if repeat==0: first.append(semantic)
                            elif semantic!=first[i]: raise ValueError('non-deterministic repeated q/geometry')
                            record=dict(run=run,model=key,experiment=cp['experiment'],variant=cp['variant'],seed=cp['seed'],
                                checkpoint_sha256=cp['sha256'],query_id=query['query_id'],group_id=query['query_group_id'],
                                subset=query['subset'],mode=query['start_class'],target_source=query['target_source'],
                                query_list_sha256=cfg['query_sha256'],robot_hashes=validator.robot.hashes,
                                pass_index=repeat,q_raw_rad=qraw,shape_valid=q.shape==(6,),**verdict,
                                elapsed_ns=elapsed,timeout_10ms=elapsed>10000000,timeout_50ms=elapsed>50000000,
                                profile_a_deadline_10ms=verdict['profile_a'] and elapsed<=10000000,
                                profile_a_deadline_50ms=verdict['profile_a'] and elapsed<=50000000,
                                profile_b_deadline_10ms=verdict['profile_b'] and elapsed<=10000000,
                                profile_b_deadline_50ms=verdict['profile_b'] and elapsed<=50000000,
                                label_present='NOT_APPLICABLE',teacher_status='NOT_APPLICABLE')
                            stream.write(json.dumps(record,allow_nan=False)+'\n'); completed+=1
                        stream.flush()
                        print(json.dumps(dict(stage='evaluate',model=key,completed_pass=repeat+1,completed_total=completed)),flush=True)
                results.append(dict(model=key,checkpoint=cp,raw=file_record(destination,lines=True),model_load_s=load_s,
                                    warmup_s=warm_s,evaluation_wall_s=time.perf_counter()-start))
                write(out/(key+'-complete.json'),results[-1])
        write(out/'evaluation.json',dict(status='COMPLETE_UNVERIFIED',time_utc=now(),rows=completed,models=results,
                                       wall_s=time.perf_counter()-total_start))
    except BaseException as exc:
        write(out/'interruption.json',dict(status='INTERRUPTED',time_utc=now(),error=repr(exc),completed_rows=completed,last=last,
                                         partial_files=[file_record(p,lines=True) for p in raw_root.glob('*seed*.jsonl')]))
        raise
