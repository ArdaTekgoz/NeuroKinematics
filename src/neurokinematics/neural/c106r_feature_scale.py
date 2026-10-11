"""Paired local-only feature scaling with immutable directional data."""
from pathlib import Path
import time
import numpy as np
import torch
from . import c106r_directions as f
from .c104 import Rows, validate_rows

d, r, e = f.d, f.r, f.e
BASE = d.ROOT/'experiments/C1-06R/diagnostic7'
RAW = d.ROOT/'data/generated/C1-06R/diagnostic7'
CONFIG = BASE/'config.json'


def fit_local(rows, pose):
    if rows.split != 'train' or not all(':train:' in p for p in rows.pair_id):
        raise ValueError('only registered training directions may fit scaling')
    if pose.shape != (len(rows.pair_id), 7) or not np.isfinite(pose).all():
        raise ValueError('bad relative pose')
    std = pose.std(0)
    if np.any(std <= 0):raise ValueError('constant feature')
    return dict(source='train_directions_only', count=len(rows.pair_id),
                pair_ids=rows.pair_id.tolist(),mean=pose.mean(0).tolist(),std=std.tolist())


def features(rows, mode, norm, local, pose=None):
    pose = r.relative_pose(rows) if pose is None else pose
    if mode == 'RAW':return r.features(rows,'relative',norm,pose)
    if mode != 'LOCAL_Z' or local['source'] != 'train_directions_only':raise ValueError('bad scaler contract')
    scaled = (pose-np.asarray(local['mean']))/np.asarray(local['std'])
    value = np.concatenate((scaled,rows.conditioned[:,-6:]),axis=1).astype(np.float32)
    if value.shape != rows.conditioned.shape or not np.isfinite(value).all():raise ValueError('invalid features')
    return value


def load_source(name):
    manifest=d.read_json(f.BASE/'preflight.json')
    path=f.RAW/(name+'.npz')
    entry=next(x for x in manifest['artifacts'] if d.ROOT/x['path']==path)
    assert d.sha(path)==entry['sha256']
    with np.load(path,allow_pickle=False) as z:
        rows=Rows(str(z['split']),**{k:z[k].copy() for k in Rows.__dataclass_fields__ if k!='split'})
    validate_rows(rows)
    return rows


def run():
    d.configure()
    if (BASE/'registration.json').exists() or RAW.exists():raise FileExistsError('preserve prior attempt')
    cfg=d.read_json(CONFIG);frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    for name in ('diagnostic3','diagnostic4','diagnostic5','diagnostic6'):
        d.guard_hashes(d.read_json(d.ROOT/f'experiments/C1-06R/{name}/registration.json')['inputs'])
    prior=d.read_json(f.BASE/'results.json')
    paths=[Path(__file__),CONFIG,d.ROOT/'tests/c1_06r/test_feature_scale.py',
        d.ROOT/'tests/c1_06r/test_validation_criteria.py',d.ROOT/'docs/adr/ADR-022-c106r-local-feature-scale.md',
        d.ROOT/cfg['normalization'],f.BASE/'results.json',f.BASE/'preflight.json',f.BASE/'audit.json']
    inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in paths}
    inputs.update({x['path']:x['sha256'] for x in d.read_json(f.BASE/'preflight.json')['artifacts']})
    d.guard_hashes(inputs)
    d.write_json(BASE/'registration.json',dict(status='REGISTERED_BEFORE_SCALER_FIT_AND_TRAINING',inputs=inputs))
    RAW.mkdir(parents=True)
    train,val=d.load_data(label_fk=True);directions,probe=load_source('directions'),load_source('probe')
    roots=d.matched_rows(train,512)[0]['local'];f.check_separation(roots,directions,probe,val)
    norm=d.read_json(d.ROOT/cfg['normalization'])
    vp=r.relative_pose(val);cells=[];initial=None
    d.write_json(BASE/'preflight.json',dict(status='PASS',geometry='REUSED_EXACT_BYTE_DIAGNOSTIC6',
        source_preflight=f.artifact(f.BASE/'preflight.json'),source_audit=f.artifact(f.BASE/'audit.json'),frozen=len(frozen)))
    for n in cfg['sizes']:
        datasets=dict(train=directions.take(np.arange(n*8)),original=roots.take(np.arange(n)),
                      same_root_probe=probe.take(np.arange(n*8)),validation=val)
        poses={k:r.relative_pose(rows) if k!='validation' else vp for k,rows in datasets.items()}
        local=fit_local(datasets['train'],poses['train']);scaler=BASE/f'n{n}-scaler.json';d.write_json(scaler,local)
        for arm in cfg['arms']:
            tick=time.perf_counter();name=f'n{n}-{arm}';out=RAW/name;out.mkdir()
            values={k:features(rows,arm,norm,local,poses[k]) for k,rows in datasets.items()}
            model=e.build_model(cfg['seed'],cfg['width']);digest=d.state_hash(model)
            initial=digest if initial is None else initial;assert digest==initial
            x=torch.tensor(values['train'],device='cuda');y=torch.tensor(datasets['train'].target_normalized,device='cuda')
            opt=torch.optim.AdamW(model.parameters(),**cfg['optimizer'])
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['steps'],**cfg['schedule']);history=[]
            for step in range(1,cfg['steps']+1):
                model.train();opt.zero_grad(set_to_none=True)
                loss=((d.predict(model,x,'residual')-y)**2).sum(-1).mean();assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step();sch.step()
                if step%1000==0:
                    history.append(dict(step=step,pre_update_q_loss=float(loss.detach())))
                    print(f'{name}: {step}/{cfg["steps"]}',flush=True)
            measured={k:r.evaluate(model,'residual',rows,values[k]) for k,rows in datasets.items()}
            control='NOT_APPLICABLE'
            if arm=='RAW':
                old=next(c for c in prior['cells'] if c['name']==f'n{n}-DIRECTIONS')
                for k in ('checkpoint','raw'):assert d.sha(d.ROOT/old[k]['path'])==old[k]['sha256']
                payload=torch.load(d.ROOT/old['checkpoint']['path'],weights_only=True)
                assert all(torch.equal(v.cpu(),payload['model_state_dict'][k]) for k,v in model.state_dict().items())
                assert measured==d.read_json(d.ROOT/old['raw']['path'])
                control='EXACT_TENSORS_AND_ALL_METRICS'
            contract=dict(inputs=inputs,name=name,scaler=f.artifact(scaler),seed=cfg['seed'],
                torch=str(torch.__version__),cuda=str(torch.version.cuda),train_pair_ids=datasets['train'].pair_id.tolist())
            weight=out/'last.pt';torch.save(dict(contract=contract,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),weight)
            saved=torch.load(weight,weights_only=True);assert saved['contract']==contract
            restored=e.build_model(cfg['seed'],cfg['width']);restored.load_state_dict(saved['model_state_dict'])
            for k,rows in datasets.items():assert r.evaluate(restored,'residual',rows,values[k])==measured[k]
            d.write_json(out/'predictions.json',measured)
            cell=dict(name=name,n_roots=n,arm=arm,steps=cfg['steps'],batch_size=8*n,exposures=8*n*cfg['steps'],
                initial_hash=initial,history=history,scaler=f.artifact(scaler),control_replay=control,
                metrics={k:r.summary(measured[k],rows) for k,rows in datasets.items()},
                checkpoint=f.artifact(weight),raw=f.artifact(out/'predictions.json'),reload='EXACT_ALL_SETS',wall_s=time.perf_counter()-tick)
            d.write_json(BASE/(name+'.json'),cell);cells.append(cell)
            print(f'{name}: train A{measured["train"]["profile_a"]}/{n*8}; probe A{measured["same_root_probe"]["profile_a"]}/{n*8}; validation A{measured["validation"]["profile_a"]}/3600',flush=True)
    d.guard_hashes(inputs);d.guard_hashes(frozen)
    d.write_json(BASE/'results.json',dict(status='COMPLETE_DIAGNOSIS',inputs=inputs,cells=cells,total_updates=20000,
        old_final_raw='NOT_READ',final_test='NOT_CREATED',single_seed=True))


if __name__=='__main__':run()
