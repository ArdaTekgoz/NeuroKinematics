"""Paired raw/relative-pose short diagnosis with fixed sample/update budgets."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from . import c106r_diagnostic2 as d
from .c106r_precision import decode_joints
from neurokinematics.kinematics.model import load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.metrics import quaternion_rotation
from neurokinematics.data.factory import canonical_quaternion

BASE = d.ROOT / 'experiments/C1-06R/diagnostic3'
CONFIG = BASE / 'config.json'


def relative_pose(rows):
    """Uses only target pose and current q, never labels or split metadata."""
    fk = PinocchioFK(load_robot())
    result = np.empty((len(rows.q_current), 7), dtype=np.float64)
    for i, q in enumerate(rows.q_current):
        current = fk.reference_forward_kinematics(q)
        result[i, :3] = rows.position[i] - current[:3, 3]
        result[i, 3:] = canonical_quaternion(current[:3, :3].T @ quaternion_rotation(rows.quaternion[i]))
    return result


def fit_normalization(rows, pose):
    if rows.split != 'train' or pose.shape != (len(rows.pair_id), 7):
        raise ValueError('fit requires matching train inputs')
    std = pose[:, :3].std(0)
    if not np.isfinite(pose).all() or (std <= 0).any():
        raise ValueError('invalid relative pose statistics')
    return dict(source='train_only', count=len(rows.pair_id),
                mean=pose[:, :3].mean(0).tolist(), std=std.tolist())


def features(rows, mode, norm, pose=None):
    if mode == 'raw':
        return rows.conditioned.copy()
    if mode != 'relative' or norm['source'] != 'train_only':
        raise ValueError('invalid feature contract')
    pose = relative_pose(rows) if pose is None else pose
    value = np.concatenate(((pose[:, :3]-norm['mean'])/norm['std'], pose[:, 3:], rows.conditioned[:, -6:]), axis=1).astype(np.float32)
    if value.shape != rows.conditioned.shape or not np.isfinite(value).all():
        raise ValueError('invalid features')
    return value


def evaluate(model, head, rows, values):
    if len(values) != len(rows.pair_id):
        raise ValueError('full input denominator required')
    model.eval()
    q = []
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            x = torch.tensor(values[start:start+1024], device='cuda')
            q.append(decode_joints(d.predict(model, x, head)).cpu().numpy())
    return d.geometric_metrics(np.concatenate(q), rows, details=True)


def summary(metric, rows):
    result = d.summarize(metric)
    records = metric['rows']
    groups = {m: rows.mode == m for m in ('local', 'wide')}
    groups.update({f: rows.family == f for f in ('main', 'boundary', 'singularity')})
    result['strata'] = {}
    for name, mask in groups.items():
        part = [r for r, use in zip(records, mask) if use]
        if not part:
            continue
        item = dict(n=len(part), profile_a=sum(r['profile_a'] for r in part), profile_b=sum(r['profile_b'] for r in part),
                    invalid=sum(not r['valid'] for r in part))
        for key in ('position_m', 'orientation_deg'):
            ordered = sorted(r[key] if r['valid'] else math.inf for r in part)
            item[key] = {label: ordered[math.ceil(p*len(ordered))-1] if math.isfinite(ordered[math.ceil(p*len(ordered))-1]) else None
                         for label,p in (('median',.5),('p95',.95))}
        result['strata'][name] = item
    return result


def run():
    d.configure()
    frozen = d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files']
    d.guard_hashes(frozen)
    cfg = d.read_json(CONFIG)
    for artifact in ('registration.json', 'results.json'):
        if (BASE/artifact).exists():
            raise FileExistsError('preserve prior attempt: '+artifact)
    train, val = d.load_data(label_fk=True)
    rp_train, rp_val = relative_pose(train), relative_pose(val)
    norm = fit_normalization(train, rp_train)
    d.write_json(BASE/'normalization.json', norm)
    source_paths = [Path(__file__), Path(d.__file__), d.ROOT/'src/neurokinematics/neural/c106r_precision.py',
                    d.ROOT/'src/neurokinematics/data/factory.py', CONFIG, BASE/'normalization.json']
    inputs = {str(p.relative_to(d.ROOT)):d.sha(p) for p in source_paths}
    d.write_json(BASE/'registration.json', dict(status='REGISTERED_BEFORE_TRAINING', inputs=inputs,
                 inherited_freeze_sha256=d.sha(d.ROOT/'experiments/C1-06R/training-freeze.json'), final_test='NOT_CREATED'))
    caches = {mode:(features(train,mode,norm,rp_train),features(val,mode,norm,rp_val)) for mode in cfg['features']}
    indices = {p:i for i,p in enumerate(train.pair_id)}
    cells, initial = [], None
    for n in cfg['sizes']:
        subsets, shared = d.matched_rows(train,n)
        for mixture in cfg['mixtures']:
            rows = subsets[mixture]
            idx = np.array([indices[p] for p in rows.pair_id])
            for mode in cfg['features']:
                x_values, v_values = caches[mode][0][idx], caches[mode][1]
                for head in cfg['heads']:
                    name = f'n{n}-{mixture}-{mode}-{head}'
                    output = d.ROOT/'data/generated/C1-06R/diagnostic3'/name
                    output.mkdir(parents=True,exist_ok=False)
                    tick = time.perf_counter()
                    model = d.build_model(cfg['seed'])
                    initial = d.state_hash(model) if initial is None else initial
                    assert d.state_hash(model) == initial
                    x, y = torch.tensor(x_values,device='cuda'), torch.tensor(rows.target_normalized,device='cuda')
                    opt = torch.optim.AdamW(model.parameters(),**cfg['optimizer'])
                    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['steps'],**cfg['schedule'])
                    history = []
                    for step in range(cfg['steps']+1):
                        if step:
                            model.train()
                            opt.zero_grad(set_to_none=True)
                            loss = ((d.predict(model,x,head)-y)**2).sum(-1).mean()
                            if not torch.isfinite(loss):
                                raise ValueError('nonfinite loss')
                            loss.backward()
                            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                                raise ValueError('nonfinite or missing gradient')
                            opt.step()
                            sch.step()
                        if step % cfg['log_every'] == 0:
                            with torch.no_grad():
                                loss_value = float(((d.predict(model,x,head)-y)**2).sum(-1).mean())
                            history.append(dict(step=step,q_loss=loss_value,**summary(evaluate(model,head,rows,x_values),rows)))
                    measured = dict(train=evaluate(model,head,rows,x_values),validation=evaluate(model,head,val,v_values))
                    weight = output/'last.pt'
                    contract = dict(seed=cfg['seed'],n=n,mixture=mixture,features=mode,head=head,inputs=inputs,
                                    torch=str(torch.__version__),cuda=str(torch.version.cuda),pair_ids=rows.pair_id.tolist())
                    torch.save(dict(contract=contract,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),weight)
                    restored = d.build_model(cfg['seed'])
                    payload = torch.load(weight,weights_only=True)
                    assert payload['contract'] == contract
                    restored.load_state_dict(payload['model_state_dict'])
                    assert evaluate(restored,head,val,v_values) == measured['validation']
                    d.write_json(output/'predictions.json',measured)
                    result = dict(name=name,n=n,mixture=mixture,features=mode,head=head,seed=cfg['seed'],steps=cfg['steps'],
                         initial_hash=initial,history=history,train=summary(measured['train'],rows),validation=summary(measured['validation'],val),
                         shared_local=summary(evaluate(model,head,shared,caches[mode][0][[indices[p] for p in shared.pair_id]]),shared),
                         checkpoint=dict(path=str(weight.relative_to(d.ROOT)),sha256=d.sha(weight)),
                         raw=dict(path=str((output/'predictions.json').relative_to(d.ROOT)),sha256=d.sha(output/'predictions.json')),
                         reload='EXACT_MATCH',wall_s=time.perf_counter()-tick)
                    d.write_json(BASE/(name+'.json'),result)
                    cells.append(result)
                    print(f"{name}: train A {result['train']['profile_a']}/{n}; validation A {result['validation']['profile_a']}/3600",flush=True)
    d.guard_hashes(inputs)
    d.write_json(BASE/'results.json',dict(status='COMPLETE_DIAGNOSIS',cells=cells,inputs=inputs,total_updates=len(cells)*cfg['steps'],
                  long_training='NOT_RUN',final_test='NOT_CREATED',single_seed=True))


if __name__ == '__main__':
    run()
