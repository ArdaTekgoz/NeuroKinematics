"""Paired objective-only fine-tuning after frozen-checkpoint diagnosis."""
from pathlib import Path
import math
import time
import torch
import numpy as np
from . import c106r_diagnostic5 as s

d,r,e=s.d,s.r,s.e
BASE=s.BASE/'loss-followup'
CONFIG=BASE/'config.json'


def pose_objective(transform, position, rotation):
    p=((transform[...,:3,3]-position)**2).sum(-1)/(.002**2)
    a=((transform[...,:3,:3]-rotation)**2).sum((-1,-2))/(8*math.sin(math.radians(1)/2)**2)
    return (p+a).mean()


def run():
    d.configure()
    if (BASE/'registration.json').exists():raise FileExistsError('preserve prior run')
    cfg=d.read_json(CONFIG);parent=d.read_json(d.ROOT/cfg['parent'])
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    prior=d.read_json(s.BASE/'results.json');d.guard_hashes(prior['inputs'])
    d.guard_hashes(d.read_json(e.BASE/'results.json')['inputs'])
    assert d.sha(d.ROOT/parent['checkpoint']['path'])==parent['checkpoint']['sha256']
    payload=torch.load(d.ROOT/parent['checkpoint']['path'],weights_only=True)
    paths=[Path(__file__),CONFIG,d.ROOT/'tests/c1_06r/test_pose_followup.py',
           d.ROOT/'docs/adr/ADR-020-c106r-profile-scaled-pose-followup.md',d.ROOT/cfg['parent'],s.BASE/'results.json']
    inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in paths}
    d.write_json(BASE/'registration.json',dict(status='REGISTERED_BEFORE_TRAINING',inputs=inputs,parent_checkpoint=parent['checkpoint']))
    train,val=d.load_data(label_fk=True);rows=d.matched_rows(train,2048)[0]['local']
    assert rows.pair_id.tolist()==payload['contract']['pair_ids']
    norm=d.read_json(d.ROOT/'experiments/C1-06R/diagnostic3/normalization.json')
    xv,vv=r.features(rows,'relative',norm),r.features(val,'relative',norm)
    x,y=torch.tensor(xv,device='cuda'),torch.tensor(rows.target_normalized,device='cuda')
    p=torch.tensor(rows.position,device='cuda');rot=torch.tensor(np.asarray([r.quaternion_rotation(q) for q in rows.quaternion]),device='cuda')
    fk=s.TrainingFK.from_frozen(domain=s.DOMAIN)
    cells=[];initial=None
    for arm in cfg['arms']:
        tick=time.perf_counter();output=d.ROOT/'data/generated/C1-06R/diagnostic5/loss-followup'/arm
        output.mkdir(parents=True,exist_ok=False)
        model=e.build_model(cfg['seed'],cfg['width']);model.load_state_dict(payload['model_state_dict'])
        digest=d.state_hash(model);initial=digest if initial is None else initial;assert digest==initial
        opt=torch.optim.AdamW(model.parameters(),**cfg['optimizer'])
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['steps'],**cfg['schedule'])
        history=[]
        for step in range(cfg['steps']+1):
            if step:
                model.train();opt.zero_grad(set_to_none=True)
                z=d.predict(model,x,'residual')
                if arm=='Q':loss=((z-y)**2).sum(-1).mean()
                else:
                    q=r.decode_joints(z);t=fk(q,robot_id=fk.robot_id,joint_names=fk.joint_names)
                    loss=pose_objective(t,p,rot)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                opt.step();sch.step()
            if step%1000==0:
                metric=r.evaluate(model,'residual',rows,xv)
                history.append(dict(step=step,**r.summary(metric,rows)))
                print(f"{arm}:step{step},train A{metric['profile_a']}/2048",flush=True)
        measured=dict(train=r.evaluate(model,'residual',rows,xv),validation=r.evaluate(model,'residual',val,vv))
        weight=output/'last.pt';contract=dict(arm=arm,inputs=inputs,torch=str(torch.__version__),cuda=str(torch.version.cuda),
            parent_checkpoint_sha256=parent['checkpoint']['sha256'],pair_ids=rows.pair_id.tolist())
        torch.save(dict(contract=contract,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),weight)
        saved=torch.load(weight,weights_only=True);assert saved['contract']==contract
        restored=e.build_model(cfg['seed'],cfg['width']);restored.load_state_dict(saved['model_state_dict'])
        assert r.evaluate(restored,'residual',rows,xv)==measured['train']
        assert r.evaluate(restored,'residual',val,vv)==measured['validation']
        raw=output/'predictions.json';d.write_json(raw,measured)
        cell=dict(name=arm,n=2048,steps=cfg['steps'],initial_hash=initial,history=history,
            train=r.summary(measured['train'],rows),validation=r.summary(measured['validation'],val),
            checkpoint=dict(path=str(weight.relative_to(d.ROOT)),sha256=d.sha(weight)),
            raw=dict(path=str(raw.relative_to(d.ROOT)),sha256=d.sha(raw)),reload='EXACT_MATCH_TRAIN_VALIDATION',wall_s=time.perf_counter()-tick)
        d.write_json(BASE/(arm+'.json'),cell);cells.append(cell)
        print(f"{arm}:validation A{cell['validation']['profile_a']}/3600",flush=True)
    d.guard_hashes(inputs);d.guard_hashes(frozen)
    d.write_json(BASE/'results.json',dict(status='COMPLETE_DIAGNOSIS',cells=cells,inputs=inputs,
        total_updates=10000,single_seed=True,final_test='NOT_CREATED',old_final_raw='NOT_READ'))


if __name__=='__main__':run()
