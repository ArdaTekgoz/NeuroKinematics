"""F0-05 serial benchmark, result integrity and transparent subgroup statistics."""

from collections import Counter,defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import os
from time import perf_counter_ns

import numpy as np

from neurokinematics.kinematics.model import load_robot
from neurokinematics.solvers.dls import DLS, SolverStatus
from .contract import load_frozen, strict_json, validate_record
from .queries import encode_query, verify_queries
from .validation import CandidateValidator, deadline_success


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def read_queries(path, dataset_root, manifest):
    verify_queries(path,dataset_root,manifest)
    started=perf_counter_ns()
    with Path(path).open('rb') as stream:
        rows=[strict_json(raw.decode('utf-8')) for raw in stream]
    return rows,perf_counter_ns()-started


def _result_record(query,result,verdict,deadline_ms,pass_index,solve_ns,validation_ns,hashes,cfg):
    total=solve_ns+validation_ns
    return {'query_id':query['query_id'],'query_group_id':query['query_group_id'],
        'query_list_sha256':hashes['query_list_sha256'],'solver_name':'DLS','solver_version':'0.1.0',
        'solver_config_sha256':hashes['solver_config_sha256'],
        'dataset_manifest_sha256':hashes['dataset_manifest_sha256'],
        'target_source':query['target_source'],'subset':query['subset'],'start_class':query['start_class'],
        'q_current':query['q_current'],
        'q_candidate':None if result.q_candidate is None else result.q_candidate.tolist(),
        'target_position_m':query['target_position_m'],
        'target_quaternion_wxyz':query['target_quaternion_wxyz'],
        'position_error_m':verdict.position_error_m,'orientation_error_rad':verdict.orientation_error_rad,
        'orientation_error_deg':verdict.orientation_error_deg,
        'profile_a_geometry':verdict.profile_a_geometry,'profile_b_geometry':verdict.profile_b_geometry,
        'joint_limits':verdict.joint_limits,'collision':'NOT_CHECKED',
        'solver_status':result.status.value,'validation_status':verdict.status.value,
        'reachability':'KNOWN_REACHABLE','reachability_proof_sha256':None,
        'termination_reason':result.termination_reason,'deadline_profile_ms':deadline_ms,
        'profile_a_deadline':deadline_success(verdict.profile_a_geometry,result.status,total,deadline_ms*1_000_000),
        'profile_b_deadline':deadline_success(verdict.profile_b_geometry,result.status,total,deadline_ms*1_000_000),
        'iterations':result.iterations,
        'iteration_availability':'AVAILABLE' if result.iterations is not None else 'NOT_AVAILABLE',
        'first_profile_a_iteration':result.first_profile_a_iteration,
        'first_profile_b_iteration':result.first_profile_b_iteration,
        'solve_elapsed_ns':solve_ns,'validation_elapsed_ns':validation_ns,'total_elapsed_ns':total,
        'measurement_pass_index':pass_index,'thermal_state':'warm',
        'seeds':{key:cfg[key] for key in ('query_seed','local_start_seed','wide_start_seed',
                  'boundary_query_seed','singularity_query_seed')},
        'frame':'base_link','tcp':'tool0','quaternion_order':'wxyz',
        'joint_order':cfg['joint_order'],'input_error':None}


def benchmark(query_path,result_path,sample_path,manifest,dataset_root,*,progress=None):
    cfg=load_frozen()['config.json']; rows,loading_ns=read_queries(query_path,dataset_root,manifest)
    solver=DLS(); validator=CandidateValidator(solver.inputs)
    hashes={'query_list_sha256':manifest['query_list_sha256'],
            'solver_config_sha256':hashlib.sha256((Path(__file__).resolve().parents[3]/'experiments/F0-05/solver-config.json').read_bytes()).hexdigest(),
            'dataset_manifest_sha256':manifest['dataset_manifest_sha256']}
    result_path=Path(result_path);result_path.parent.mkdir(parents=True,exist_ok=True)
    sample_path=Path(sample_path);sample_path.parent.mkdir(parents=True,exist_ok=True)
    stats=defaultdict(GroupStats);status_counts=Counter();io_ns=0;count=0
    started=datetime.now(timezone.utc).isoformat();warmup_ns=0;warmup_calls=0
    with result_path.open('wb') as output,sample_path.open('wb') as sample:
        for deadline in cfg['deadline_profiles_ms']:
            t=perf_counter_ns()
            for row in rows[:cfg['warmup_queries_per_deadline']]:
                solver.solve(row['q_current'],row['target_position_m'],row['target_quaternion_wxyz'],deadline_ns=deadline*1_000_000)
                warmup_calls+=1
            warmup_ns+=perf_counter_ns()-t
            for pass_index in range(cfg['measurement_passes']):
                for index,row in enumerate(rows):
                    # q_target is intentionally never sent to DLS.
                    t0=perf_counter_ns()
                    result=solver.solve(row['q_current'],row['target_position_m'],row['target_quaternion_wxyz'],deadline_ns=deadline*1_000_000)
                    t1=perf_counter_ns()
                    verdict=validator.validate(result,row['target_position_m'],row['target_quaternion_wxyz'])
                    # Verdict is sampled once within the validation interval.
                    deadline_success(verdict.profile_b_geometry,result.status,perf_counter_ns()-t0,deadline*1_000_000)
                    t2=perf_counter_ns()
                    record=_result_record(row,result,verdict,deadline,pass_index,t1-t0,t2-t1,hashes,cfg)
                    payload=encode_query(record)
                    io_start=perf_counter_ns();output.write(payload)
                    if pass_index==0 and index in (0,1,2,5000,10000,11000):sample.write(payload)
                    io_ns+=perf_counter_ns()-io_start
                    status_counts[result.status.value]+=1
                    for profile in ('a','b'):
                        for key in (f"{row['subset']}/{row['start_class']}/{deadline}ms/{profile.upper()}",
                                    f"all/{deadline}ms/{profile.upper()}",
                                    f"{row['subset']}/{deadline}ms/{profile.upper()}"):
                            stats[key].add(record,profile)
                    count+=1
                if progress: progress(deadline,pass_index,count)
    expected=sum(cfg['query_counts'].values())*len(cfg['deadline_profiles_ms'])*cfg['measurement_passes']
    if count!=expected:raise ValueError(f'incomplete benchmark count {count}/{expected}')
    return {'status':'MEASURED_UNVERIFIED','started_utc':started,
            'finished_utc':datetime.now(timezone.utc).isoformat(),
            'query_list_sha256':manifest['query_list_sha256'],
            'result_file':result_path.as_posix(),'result_file_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'record_count':count,'independent_query_count':len(rows),'warmup_calls':warmup_calls,
            'warmup_elapsed_ns':warmup_ns,'query_loading_ns':loading_ns,'serialization_io_ns':io_ns,
            'status_counts':dict(status_counts),'groups':{k:v.summarize() for k,v in stats.items()},
            'environment':{'platform':platform.platform(),'python':platform.python_version(),
                'numpy':np.__version__,'logical_cpu_count':os.cpu_count(),'cpu':platform.processor(),
                'ram':'NOT_MEASURED','gpu':'NOT_USED','thread_count':'NOT_MEASURED',
                'timing_clock':'perf_counter_ns','execution':'serial CPU','linux':'NOT_RUN'}}


def percentiles(values):
    if not values:return {'count':0,'median':None,'p50':None,'p95':None,'p99':None,'max':None}
    a=np.asarray(values,dtype=np.float64)
    return {'count':len(a),'median':float(np.median(a)),'p50':float(np.percentile(a,50,method='linear')),
            'p95':float(np.percentile(a,95,method='linear')),'p99':float(np.percentile(a,99,method='linear')),
            'max':float(np.max(a))}


class GroupStats:
    def __init__(self):
        self.groups=set();self.p=[];self.r=[];self.all_latency=[];self.success_latency=[]
        self.iterations=[];self.geo=0;self.dead=0;self.status=Counter();self.limit_fail=0
        self.pass_counts=Counter()

    def add(self,row,profile):
        self.groups.add(row['query_group_id']);self.pass_counts[row['measurement_pass_index']]+=1
        geometry=row[f'profile_{profile}_geometry'];deadline=row[f'profile_{profile}_deadline']
        self.geo+=geometry;self.dead+=deadline;self.status[row['solver_status']]+=1
        self.limit_fail+=row['joint_limits']=='FAIL'
        if row['position_error_m'] is not None:self.p.append(row['position_error_m'])
        if row['orientation_error_deg'] is not None:self.r.append(row['orientation_error_deg'])
        self.all_latency.append(row['total_elapsed_ns'])
        if deadline:self.success_latency.append(row['total_elapsed_ns'])
        if row['iterations'] is not None:self.iterations.append(row['iterations'])

    def summarize(self):
        n=len(self.all_latency)
        return {'unique_queries':len(self.groups),'attempts':n,'measurement_pass_counts':dict(self.pass_counts),
            'geometry_success_count':self.geo,'geometry_success_rate':self.geo/n if n else None,
            'deadline_success_count':self.dead,'deadline_success_rate':self.dead/n if n else None,
            'status_counts':dict(self.status),'joint_limit_violations':self.limit_fail,
            'position_error_m':percentiles(self.p),'position_error_missing':n-len(self.p),
            'orientation_error_deg':percentiles(self.r),'orientation_error_missing':n-len(self.r),
            'all_query_latency_ns':percentiles(self.all_latency),
            'deadline_success_latency_ns':percentiles(self.success_latency),
            'iterations':percentiles(self.iterations),'iteration_missing':n-len(self.iterations),
            'warmup':'SEPARATE','cpu':'NOT_MEASURED','thread_count':'NOT_MEASURED','ram':'NOT_MEASURED'}


def verify_results(query_path,result_path,dataset_root,manifest,*,expected_file_hash=None):
    rows,_=read_queries(query_path,dataset_root,manifest)
    cfg=load_frozen()['config.json']; validator=CandidateValidator(load_robot())
    expected={'query_list_sha256':manifest['query_list_sha256'],
              'solver_config_sha256':hashlib.sha256((Path(__file__).resolve().parents[3]/'experiments/F0-05/solver-config.json').read_bytes()).hexdigest(),
              'dataset_manifest_sha256':manifest['dataset_manifest_sha256']}
    digest=hashlib.sha256();count=0;stats=defaultdict(GroupStats)
    with Path(result_path).open('rb') as stream:
        for deadline in cfg['deadline_profiles_ms']:
            for pass_index in range(cfg['measurement_passes']):
                for query in rows:
                    raw=stream.readline()
                    if not raw or not raw.endswith(b'\n') or b'\r' in raw:
                        raise ValueError('missing or truncated result JSONL')
                    digest.update(raw);record=strict_json(raw.decode('utf-8'))
                    if encode_query(record)!=raw:raise ValueError('noncanonical result JSONL')
                    if (record['query_id']!=query['query_id'] or record['query_group_id']!=query['query_group_id']
                        or record['subset']!=query['subset'] or record['start_class']!=query['start_class']
                        or record['q_current']!=query['q_current']
                        or record['target_position_m']!=query['target_position_m']
                        or record['target_quaternion_wxyz']!=query['target_quaternion_wxyz']
                        or record['deadline_profile_ms']!=deadline or record['measurement_pass_index']!=pass_index):
                        raise ValueError('result/query/order binding mismatch')
                    validate_record(record,validator=validator,expected_hashes=expected)
                    for profile in ('a','b'):
                        for key in (f"{query['subset']}/{query['start_class']}/{deadline}ms/{profile.upper()}",
                                    f"all/{deadline}ms/{profile.upper()}",
                                    f"{query['subset']}/{deadline}ms/{profile.upper()}"):
                            stats[key].add(record,profile)
                    count+=1
        if stream.readline():raise ValueError('extra result JSONL rows')
    expected_count=len(rows)*len(cfg['deadline_profiles_ms'])*cfg['measurement_passes']
    if count!=expected_count:raise ValueError('result row count mismatch')
    if expected_file_hash and digest.hexdigest()!=expected_file_hash:
        raise ValueError('result file SHA256 mismatch')
    return {'status':'PASS','record_count':count,'result_file_sha256':digest.hexdigest(),
            'groups':{k:v.summarize() for k,v in stats.items()}}
