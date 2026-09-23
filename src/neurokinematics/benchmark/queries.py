"""Frozen F0-05 query generation and independent F0-04 exclusion audit."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from neurokinematics.data.factory import canonical_quaternion, read_shard, verify_dataset
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.metrics import singularity_metrics, quaternion_rotation
from neurokinematics.kinematics.model import ROOT, load_robot, validate_q
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from .contract import load_frozen, strict_json, load_reproduction_config


def q_key(q):
    value = np.asarray(q, dtype='<f8').copy()
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError('invalid q for exact duplicate audit')
    value[value == 0] = 0
    return value.tobytes(order='C')


def f04_exclusions(dataset_root, *, dataset_manifest=None, data_config=None):
    path = Path(dataset_manifest) if dataset_manifest else ROOT/'experiments/F0-04/dataset-manifest.json'
    manifest = json.loads(path.read_bytes())
    verify_dataset(Path(dataset_root), path, **({"config_path": Path(data_config)} if data_config else {}))
    schema = json.loads((ROOT/'experiments/F0-04/schema.json').read_bytes())
    order = [field['name'] for field in schema['fields']]
    q, groups = set(), set()
    for subset in ('main','boundary','singularity'):
        for shard in manifest['shards'][subset]:
            arrays = read_shard(Path(dataset_root)/shard['path'], order)
            q.update(map(q_key, arrays['q']))
            groups.update(x.decode() for x in arrays['group_id'])
    return q, groups, manifest


def encode_query(row):
    return (json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')


def generate_queries(path, dataset_root, *, config_path=None, dataset_manifest=None, data_config=None):
    if any((config_path, dataset_manifest, data_config)) and not all((config_path, dataset_manifest, data_config)):
        raise ValueError('reproduction requires config, dataset manifest and data config together')
    frozen = load_frozen(); cfg = load_reproduction_config(config_path) if config_path else frozen['config.json']
    inputs = load_robot()
    old_q, old_groups, dataset = f04_exclusions(dataset_root, **({"dataset_manifest": dataset_manifest, "data_config": data_config} if dataset_manifest else {}))
    bounds = np.asarray(inputs.limits,dtype=np.float64)
    streams = {name:np.random.Generator(np.random.PCG64(cfg[key])) for name,key in
               [('main','query_seed'),('boundary','boundary_query_seed'),('singularity','singularity_query_seed'),
                ('local','local_start_seed'),('wide','wide_start_seed')]}
    pin, independent, jac = PinocchioFK(inputs), IndependentFK(inputs), IndependentJacobian(inputs)
    used_q, used_groups, used_ids = set(),set(),set()
    counters = {}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(); total=0
    with path.open('wb') as output:
        for subset in ('main','boundary','singularity'):
            stats=Counter(); count=cfg['query_counts'][subset]
            for index in range(count):
                for ordinal in range(1,cfg['target_candidate_limits'][subset]+1):
                    stats['target_candidates']+=1
                    target=streams[subset].uniform(bounds[:,0],bounds[:,1])
                    if subset=='boundary':
                        u=streams[subset].random()
                        joint=(index//2)%len(bounds); side=index%2
                        target[joint]=bounds[joint,side]+(u*.02 if side==0 else -u*.02)*(bounds[joint,1]-bounds[joint,0])
                        dist=np.minimum((target-bounds[:,0])/(bounds[:,1]-bounds[:,0]),
                                        (bounds[:,1]-target)/(bounds[:,1]-bounds[:,0]))
                        if not np.min(dist)<cfg['boundary_normalized_limit_distance_lt']:
                            stats['boundary_rejected']+=1; continue
                    if subset=='singularity':
                        sigma=singularity_metrics(jac.jacobian(target),cfg['characteristic_length_m'])['sigma_min']
                        if sigma>cfg['singularity_sigma_min_le']:
                            stats['singularity_rejected']+=1; continue
                    key=q_key(target)
                    if key in old_q or key in used_q:
                        stats['duplicate_rejected']+=1; continue
                    break
                else: raise ValueError(f'{subset} target candidate cap reached at {index}')
                start_class='local' if index%2==0 else 'wide'
                for attempts in range(1,cfg['start_candidate_limit_per_query']+1):
                    stats[f'{start_class}_candidates']+=1
                    if start_class=='local':
                        current=target+streams['local'].uniform(*cfg['local_perturbation_uniform_rad'],size=6)
                        if np.any(current<bounds[:,0]) or np.any(current>bounds[:,1]):
                            stats['local_limit_rejected']+=1; continue
                        if q_key(current)==key:
                            stats['local_equal_rejected']+=1; continue
                    else:
                        current=streams['wide'].uniform(bounds[:,0],bounds[:,1])
                        distance=float(np.sqrt(np.mean(((current-target)/(bounds[:,1]-bounds[:,0]))**2)))
                        if distance<cfg['wide_min_normalized_rms_distance']:
                            stats['wide_near_rejected']+=1; continue
                    break
                else: raise ValueError(f'{subset} {start_class} candidate cap reached at {index}')
                validate_q(current,inputs.joint_names,inputs.limits)
                reference=pin.reference_forward_kinematics(target)
                check=independent.forward_kinematics(target)
                if (np.linalg.norm(reference[:3,3]-check[:3,3])>1e-9
                    or np.linalg.norm(reference[:3,:3]-check[:3,:3],ord='fro')>1e-9):
                    raise ValueError('independent target FK mismatch')
                quat=canonical_quaternion(reference[:3,:3])
                qid=f'f05-{subset}-{index:08d}'; gid=f'root-f05-{subset}-{index:08d}'
                if qid in used_ids or gid in used_groups or gid in old_groups:
                    raise ValueError('query ID or group collision')
                row={'query_id':qid,'query_group_id':gid,'subset':subset,
                     'target_source':'independent_frozen_fk','start_class':start_class,
                     'q_target':target.tolist(),'q_current':current.tolist(),
                     'target_position_m':reference[:3,3].tolist(),'target_quaternion_wxyz':quat.tolist(),
                     'target_candidate_index':ordinal,'start_candidate_count':attempts}
                payload=encode_query(row); output.write(payload); digest.update(payload)
                used_q.add(key); used_ids.add(qid); used_groups.add(gid)
                stats['accepted']+=1; stats[f'{start_class}_accepted']+=1; total+=1
            counters[subset]=dict(stats)
    return {'schema_version':'1.0.0','status':'PASS','file':path.as_posix(),
            'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'content_sha256':digest.hexdigest(),'query_list_sha256':digest.hexdigest(),
            'record_count':total,'subsets':counters,'f04_exact_q_duplicates':0,
            'f04_group_intersections':0,'cross_subset_q_duplicates':0,
            'cross_subset_id_duplicates':0,'cross_subset_group_intersections':0,
            'dataset_manifest_sha256':hashlib.sha256((Path(dataset_manifest) if dataset_manifest else ROOT/'experiments/F0-04/dataset-manifest.json').read_bytes()).hexdigest(),
            'dataset_content_sha256':dataset['dataset_content_sha256'],
            'config_sha256':hashlib.sha256((Path(config_path) if config_path else ROOT/'experiments/F0-05/config.json').read_bytes()).hexdigest(),
            'reproduction_command':f'pixi run --locked generate-f05 --output {path.as_posix()} --dataset-root {Path(dataset_root).as_posix()}'}


def verify_queries(path,dataset_root,expected_manifest=None, *, config_path=None, dataset_manifest=None, data_config=None):
    if any((config_path, dataset_manifest, data_config)) and not all((config_path, dataset_manifest, data_config)):
        raise ValueError('reproduction requires config, dataset manifest and data config together')
    if config_path and expected_manifest:
        if (expected_manifest['config_sha256'] != hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
                or expected_manifest['dataset_manifest_sha256'] != hashlib.sha256(Path(dataset_manifest).read_bytes()).hexdigest()):
            raise ValueError('reproduction config/manifest binding mismatch')
    cfg=load_reproduction_config(config_path) if config_path else load_frozen()['config.json']; inputs=load_robot()
    old_q,old_groups,_=f04_exclusions(dataset_root, **({"dataset_manifest": dataset_manifest, "data_config": data_config} if dataset_manifest else {}))
    pin=PinocchioFK(inputs); bounds=np.asarray(inputs.limits)
    seen_q,seen_id,seen_group=set(),set(),set(); counts=Counter(); starts=Counter()
    path=Path(path); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.endswith(b'\n') or b'\r' in raw:
                raise ValueError('query JSONL newline violation')
            row=strict_json(raw.decode('utf-8'))
            if encode_query(row)!=raw: raise ValueError('query canonical encoding or ordering violation')
            subset=row['subset']; index=counts[subset]
            if subset not in cfg['query_counts'] or index>=cfg['query_counts'][subset]:
                raise ValueError('invalid subset/count')
            if subset!=('main' if sum(counts.values())<cfg['query_counts']['main'] else
                       'boundary' if sum(counts.values())<cfg['query_counts']['main']+cfg['query_counts']['boundary'] else 'singularity'):
                raise ValueError('query subset/order violation')
            if set(row)!=set(cfg_contract_query_fields()): raise ValueError('query fields violation')
            q=validate_q(row['q_target'],inputs.joint_names,inputs.limits)
            current=validate_q(row['q_current'],inputs.joint_names,inputs.limits)
            key=q_key(q); qid=f'f05-{subset}-{index:08d}'; gid=f'root-f05-{subset}-{index:08d}'
            if key in old_q or key in seen_q or row['query_id']!=qid or qid in seen_id or row['query_group_id']!=gid or gid in seen_group or gid in old_groups:
                raise ValueError('query duplicate/identity violation')
            expected_class='local' if index%2==0 else 'wide'
            if row['start_class']!=expected_class or row['target_source']!='independent_frozen_fk':
                raise ValueError('query start/source violation')
            if not isinstance(row['target_candidate_index'],int) or not 1<=row['target_candidate_index']<=cfg['target_candidate_limits'][subset] or not isinstance(row['start_candidate_count'],int) or not 1<=row['start_candidate_count']<=cfg['start_candidate_limit_per_query']:
                raise ValueError('invalid candidate counts')
            if expected_class=='local':
                d=current-q
                if q_key(current)==key or np.any(d<cfg['local_perturbation_uniform_rad'][0]-1e-14) or np.any(d>cfg['local_perturbation_uniform_rad'][1]+1e-14):
                    raise ValueError('local start violation')
            else:
                if np.sqrt(np.mean(((current-q)/(bounds[:,1]-bounds[:,0]))**2))<cfg['wide_min_normalized_rms_distance']:
                    raise ValueError('wide start violation')
            if subset=='boundary':
                d=np.minimum((q-bounds[:,0])/(bounds[:,1]-bounds[:,0]),(bounds[:,1]-q)/(bounds[:,1]-bounds[:,0]))
                if not np.min(d)<cfg['boundary_normalized_limit_distance_lt']: raise ValueError('boundary violation')
            elif subset=='singularity':
                jac=IndependentJacobian(inputs).jacobian(q)
                if singularity_metrics(jac,cfg['characteristic_length_m'])['sigma_min']>cfg['singularity_sigma_min_le']:
                    raise ValueError('singularity violation')
            pose=pin.reference_forward_kinematics(q)
            quat=canonical_quaternion(pose[:3,:3])
            if not np.allclose(row['target_position_m'],pose[:3,3],rtol=0,atol=1e-12) or np.linalg.norm(quaternion_rotation(row['target_quaternion_wxyz'])-pose[:3,:3],ord='fro')>1e-12 or not np.allclose(row['target_quaternion_wxyz'],quat,rtol=0,atol=1e-12):
                raise ValueError('target pose mismatch')
            seen_q.add(key);seen_id.add(qid);seen_group.add(gid);counts[subset]+=1;starts[(subset,expected_class)]+=1
    if dict(counts)!=cfg['query_counts'] or any(starts[(subset,kind)]!=count//2 for subset,count in cfg['query_counts'].items() for kind in ('local','wide')):
        raise ValueError('query count or 50/50 ratio violation')
    actual=digest.hexdigest()
    if expected_manifest and (actual!=expected_manifest['query_list_sha256'] or len(seen_q)!=expected_manifest['record_count']):
        raise ValueError('query manifest hash/count mismatch')
    return {'status':'PASS','query_list_sha256':actual,'counts':dict(counts),
            'starts':{f'{k[0]}_{k[1]}':v for k,v in starts.items()},'f04_q_duplicates':0,
            'f04_group_intersections':0,'cross_subset_duplicates':0}


def cfg_contract_query_fields():
    return load_frozen()['benchmark-contract.json']['query_generation']['query_fields']
