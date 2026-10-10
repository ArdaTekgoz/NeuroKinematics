"""Preregistered three-seed centered residual control; no numerical refinement."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from . import c106r_local_response as a

d,r,e,z,s=a.d,a.r,a.e,a.z,a.s
BASE=d.ROOT/'experiments/C1-06R/diagnostic9'
RAW=d.ROOT/'data/generated/C1-06R/diagnostic9'
CONFIG=BASE/'config.json'


class Centered(torch.nn.Module):
    def __init__(self, net, norm):
        super().__init__()
        self.net=net
        param=next(net.parameters())
        zero=np.r_[-np.asarray(norm['mean'])/norm['std'],1.,0.,0.,0.]
        self.register_buffer('zero_pose',torch.tensor(zero,dtype=param.dtype,device=param.device))

    def zero_input(self,x):
        return torch.cat((self.zero_pose.expand(x.shape[0],-1),x[:,-6:]),dim=1)

    def forward(self,x):
        return self.net(x)-self.net(self.zero_input(x))


def datasets():
    train,val=d.load_data(label_fk=True)
    roots=d.matched_rows(train,512)[0]['local']
    directions,probe=z.load_source('directions'),z.load_source('probe')
    z.f.check_separation(roots,directions,probe,val)
    return dict(train=directions,original=roots,same_root_probe=probe,validation=val,
        zero_train_current=a.zero_queries(roots,'current'),
        zero_validation_current=a.zero_queries(val.take(np.flatnonzero(val.mode=='local')),'current'))


def anchor_data(data):
    robot=r.load_robot();bounds=np.asarray(robot.limits);groups={};checks={}
    for key,rows in (('train',data['original'].take(np.arange(64))),
                     ('validation',data['validation'].take(np.flatnonzero((data['validation'].mode=='local')&(data['validation'].family=='main'))))):
        margin=np.minimum(np.minimum(rows.q_current-bounds[:,0],bounds[:,1]-rows.q_current),
                          np.minimum(rows.q_target-bounds[:,0],bounds[:,1]-rows.q_target)).min(1)
        zero=a.zero_queries(rows.take(np.flatnonzero(margin>.002)[:32]),'current')
        assert len(zero.pair_id)==32
        _,_,_,checks[key]=s.preflight(zero,d.read_json(d.ROOT/'experiments/C1-06R/diagnostic5/config.json'),robot)
        groups[key]=zero
    return groups,checks


def derivative(model,groups,norm,out):
    robot=r.load_robot();bounds=np.asarray(robot.limits)
    jac=s.IndependentJacobian(robot);fk=s.IndependentFK(robot);result={}
    shadow=deepcopy(model).double()
    for name,zero in groups.items():
        j=np.asarray([jac.jacobian(q) for q in zero.q_current])
        tangent=np.asarray([a.feature_tangent(jj,fk.forward_kinematics(q)[:3,:3],'RAW',norm,None) for jj,q in zip(j,zero.q_current)])
        q,b=a.input_derivative(model,a.feature_values(zero,'RAW',norm,None))
        k=b[:,:,:7]@tangent
        valid=np.isfinite(q).all(1)&(q>=bounds[:,0]).all(1)&(q<=bounds[:,1]).all(1)
        jp=np.full_like(j,np.nan);errors=[]
        for i in np.flatnonzero(valid):
            jp[i]=jac.jacobian(q[i]);errors.append(a.response_error(j[i],k[i],jp[i]))
        _,b64=a.input_derivative(shadow,a.feature_values(zero,'RAW',norm,None,np.float64))
        k64=b64[:,:,:7]@tangent
        shifted=a.perturb_queries(zero,1e-5)
        fd=a.fd_matrix(a.predict(shadow,a.feature_values(shifted,'RAW',norm,None,np.float64)),32,1e-5)
        assert np.allclose(fd,k64,atol=1e-5,rtol=.001)
        path=out/(name+'-derivative.npz')
        np.savez_compressed(path,anchor_q=zero.q_current,q_zero=q,j_anchor=j,j_prediction=jp,
            k=k,k64=k64,fd64=fd,valid=valid,tangent=tangent)
        result[name]=dict(n=32,valid=int(valid.sum()),actual_task_relative_error_valid_only=a.stats(errors),
            shadow_fd_error=a.stats(np.linalg.norm(fd-k64,axis=(1,2))),raw=z.f.artifact(path))
    return result


def run():
    d.configure();tick=time.perf_counter()
    if (BASE/'registration.json').exists() or RAW.exists():raise FileExistsError('preserve prior attempt')
    cfg=d.read_json(CONFIG);frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    for name in range(3,9):d.guard_hashes(d.read_json(d.ROOT/f'experiments/C1-06R/diagnostic{name}/registration.json')['inputs'])
    paths=[Path(__file__),CONFIG,d.ROOT/'tests/c1_06r/test_centered.py',
        d.ROOT/'docs/adr/ADR-024-c106r-centered-residual.md',d.ROOT/cfg['normalization'],
        z.BASE/'results.json',z.f.BASE/'preflight.json',a.BASE/'preflight.json']
    inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in paths}
    inputs.update({x['path']:x['sha256'] for x in d.read_json(z.f.BASE/'preflight.json')['artifacts']})
    prior=next(c for c in d.read_json(z.BASE/'results.json')['cells'] if c['name']=='n512-RAW')
    for key in ('checkpoint','raw'):inputs[prior[key]['path']]=prior[key]['sha256']
    d.guard_hashes(inputs)
    d.write_json(BASE/'registration.json',dict(status='REGISTERED_BEFORE_TRAINING',inputs=inputs))
    RAW.mkdir(parents=True)
    data=datasets();norm=d.read_json(d.ROOT/cfg['normalization']);groups,checks=anchor_data(data)
    previous=d.read_json(a.BASE/'preflight.json')['selection']
    for name,rows in groups.items():assert rows.pair_id.tolist()==previous[name+'-current']
    values={key:r.features(rows,'relative',norm) for key,rows in data.items()}
    d.write_json(BASE/'preflight.json',dict(status='PASS',geometry=checks,
        selection={k:v.pair_id.tolist() for k,v in groups.items()},source_geometry='EXACT_DIAGNOSTIC6_DATA',frozen=len(frozen)))
    cells=[]
    for seed in cfg['seeds']:
        paired_initial=None
        for arm in cfg['arms']:
            start=time.perf_counter();name=f's{seed}-{arm}';out=RAW/name;out.mkdir()
            net=e.build_model(seed,cfg['width']);initial=d.state_hash(net)
            paired_initial=initial if paired_initial is None else paired_initial;assert initial==paired_initial
            model=net if arm=='RAW' else Centered(net,norm)
            x=torch.tensor(values['train'],device='cuda');y=torch.tensor(data['train'].target_normalized,device='cuda')
            opt=torch.optim.AdamW(model.parameters(),**cfg['optimizer'])
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['steps'],**cfg['schedule']);history=[]
            torch.cuda.synchronize();train_start=time.perf_counter()
            for step in range(1,cfg['steps']+1):
                model.train();opt.zero_grad(set_to_none=True)
                loss=((d.predict(model,x,'residual')-y)**2).sum(-1).mean();assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step();sch.step()
                if step%1000==0:
                    history.append(dict(step=step,pre_update_q_loss=float(loss.detach())))
                    print(f'{name}: {step}/{cfg["steps"]}',flush=True)
            torch.cuda.synchronize();train_s=time.perf_counter()-train_start
            measured={key:r.evaluate(model,'residual',rows,values[key]) for key,rows in data.items()}
            control='NOT_APPLICABLE'
            if arm=='RAW' and seed==cfg['seeds'][0]:
                saved=torch.load(d.ROOT/prior['checkpoint']['path'],weights_only=True)
                assert all(torch.equal(v.cpu(),saved['model_state_dict'][k]) for k,v in net.state_dict().items())
                old=d.read_json(d.ROOT/prior['raw']['path'])
                assert {k:measured[k] for k in old}==old
                control='EXACT_TENSORS_AND_FOUR_METRIC_SETS'
            contract=dict(name=name,arm=arm,seed=seed,inputs=inputs,normalization=cfg['normalization'],
                torch=str(torch.__version__),cuda=str(torch.version.cuda),train_pair_ids=data['train'].pair_id.tolist())
            weight=out/'last.pt';torch.save(dict(contract=contract,model_state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()}),weight)
            saved=torch.load(weight,weights_only=True);assert saved['contract']==contract
            restored=e.build_model(seed,cfg['width']);restored.load_state_dict(saved['model_state_dict'])
            if arm=='CENTERED':restored=Centered(restored,norm)
            for key,rows in data.items():assert r.evaluate(restored,'residual',rows,values[key])==measured[key]
            derivatives=derivative(model,groups,norm,out)
            d.write_json(out/'predictions.json',measured)
            cell=dict(name=name,arm=arm,seed=seed,steps=cfg['steps'],batch_size=4096,exposures=4096*cfg['steps'],
                initial_hash=initial,parameters=sum(p.numel() for p in model.parameters()),history=history,
                metrics={k:r.summary(measured[k],rows) for k,rows in data.items()},derivatives=derivatives,
                checkpoint=z.f.artifact(weight),raw=z.f.artifact(out/'predictions.json'),reload='EXACT_ALL_SIX_SETS',
                control_replay=control,training_s=train_s,wall_s=time.perf_counter()-start)
            d.write_json(BASE/(name+'.json'),cell);cells.append(cell)
            print(f'{name}: train A{measured["train"]["profile_a"]}/4096; probe A{measured["same_root_probe"]["profile_a"]}/4096; validation A{measured["validation"]["profile_a"]}/3600',flush=True)
    d.guard_hashes(inputs);d.guard_hashes(frozen)
    d.write_json(BASE/'results.json',dict(status='COMPLETE_DIAGNOSIS',inputs=inputs,cells=cells,total_updates=30000,
        total_exposures=122880000,wall_s=time.perf_counter()-tick,old_final_raw='NOT_READ',final_test='NOT_CREATED'))


if __name__=='__main__':run()
