"""Research-only portable inference and T-C06 witness; no IK refinement."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import torch
from . import c105
from . import c106r_diagnostic3 as r
from .c106r_diagnostic4 import DiagnosticMLP
from ..kinematics.custom_fk import IndependentFK

ROOT=c105.ROOT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_file(entry,root=ROOT):
    p=Path(root)/entry['path']
    if sha(p)!=entry['sha256']:raise ValueError('artifact SHA mismatch: '+entry['path'])
    return p


def feature_inputs(requests):
    p=np.asarray([v['position_m'] for v in requests],dtype=np.float64)
    quat=np.asarray([v['quaternion_wxyz'] for v in requests],dtype=np.float64)
    qc=np.asarray([v['q_current_rad'] for v in requests],dtype=np.float64)
    if p.shape!=(len(requests),3) or quat.shape!=(len(requests),4) or qc.shape!=(len(requests),6):raise ValueError('invalid request shape')
    if not all(np.isfinite(v).all() for v in (p,quat,qc)):raise ValueError('nonfinite request')
    if not np.allclose(np.linalg.norm(quat,axis=1),1,atol=1e-10,rtol=0):raise ValueError('unit quaternion required')
    for q in quat:
        nonzero=q[np.abs(q)>1e-15]
        if len(nonzero) and nonzero[0]<0:raise ValueError('canonical wxyz required')
    robot=r.load_robot();lo,hi=np.asarray(robot.limits).T
    if np.any(qc<lo) or np.any(qc>hi):raise ValueError('current outside limits')
    norm=json.loads((ROOT/'experiments/C1-02/normalization.json').read_text())
    conditioned=np.c_[(p-norm['position_mean_m'])/norm['position_std_m'],quat,(qc-lo)/(hi-lo)].astype(np.float32)
    return SimpleNamespace(position=p,quaternion=quat,q_current=qc,conditioned=conditioned)


def evaluate(q,requests):
    robot=r.load_robot();lo,hi=np.asarray(robot.limits).T;fk=IndependentFK(robot);out=[]
    for value,request in zip(q,requests):
        valid=value.shape==(6,) and np.isfinite(value).all() and (value>=lo).all() and (value<=hi).all()
        if valid:
            t=fk.forward_kinematics(value);p=float(np.linalg.norm(t[:3,3]-request['position_m']))
            rr=t[:3,:3]@r.quaternion_rotation(np.asarray(request['quaternion_wxyz'])).T
            v=np.array([rr[2,1]-rr[1,2],rr[0,2]-rr[2,0],rr[1,0]-rr[0,1]])/2
            angle=float(np.degrees(np.arctan2(np.linalg.norm(v),(np.trace(rr)-1)/2)))
        else:p=angle=float('inf')
        out.append(dict(q_rad=value.tolist(),valid=bool(valid),profile_a=bool(p<=.002 and angle<=1),
            profile_b=bool(p<=.001 and angle<=.5),position_m=p if valid else None,orientation_deg=angle if valid else None))
    return out


def predict(candidate,requests,artifact_root=ROOT):
    for field in ('normalization','source_config'):checked_file(candidate[field])
    path=checked_file(candidate['checkpoint'],artifact_root);rows=feature_inputs(requests)
    if candidate['family']=='FK_TANH':
        model,meta=c105.load_checkpoint(path)
        if meta['variant']!='FK_TANH' or meta['training_seed']!=candidate['seed']:raise ValueError('checkpoint identity mismatch')
        q,_=c105.infer(model,'FK_TANH',rows.conditioned)
    elif candidate['family']=='LOCAL_RAW':
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['contract']['arm']!='RAW' or payload['contract']['seed']!=candidate['seed']:raise ValueError('checkpoint identity mismatch')
        model=DiagnosticMLP(512);model.load_state_dict(payload['model_state_dict']);model.eval()
        norm=json.loads(checked_file(candidate['normalization']).read_text());features=r.features(rows,'relative',norm)
        with torch.no_grad():
            x=torch.tensor(features);q=r.decode_joints(model(x)+x[:,-6:]).numpy()
    else:raise ValueError('unknown model family')
    return evaluate(q,requests)


def verify_manifest(manifest):
    if manifest['robot']['hashes']!=r.load_robot().hashes:raise ValueError('robot identity mismatch')
    for name,digest in manifest['inputs'].items():
        if not name.startswith('data/') and sha(ROOT/name)!=digest:raise ValueError('source drift: '+name)
