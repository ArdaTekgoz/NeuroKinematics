"""Fixed-weight zero-displacement and local differential response diagnosis."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
from . import c106r_feature_scale as z
from .c104 import contracts

d,r,e,s=z.d,z.r,z.e,z.f.s
BASE=d.ROOT/'experiments/C1-06R/diagnostic8'
RAW=d.ROOT/'data/generated/C1-06R/diagnostic8'
CONFIG=BASE/'config.json'


def queries(base,current,target,suffix):
    robot=r.load_robot();bounds=np.asarray(robot.limits);fk=s.IndependentFK(robot)
    ts=np.asarray([fk.forward_kinematics(q) for q in target])
    p=ts[:,:3,3];quat=np.asarray([r.canonical_quaternion(t[:3,:3]) for t in ts])
    norm=contracts()[2]
    pos=(p-norm['position_mean_m'])/norm['position_std_m']
    qc=(current-bounds[:,0])/(bounds[:,1]-bounds[:,0])
    return replace(base,pair_id=np.asarray([str(a)+':'+suffix+':'+str(i) for i,a in enumerate(base.pair_id)]),
        q_current=current.copy(),q_target=target.copy(),position=p,quaternion=quat,
        label_present=np.ones(len(target),dtype=bool),pose_only=np.c_[pos,quat].astype(np.float32),
        conditioned=np.c_[pos,quat,qc].astype(np.float32),
        target_normalized=((target-bounds[:,0])/(bounds[:,1]-bounds[:,0])).astype(np.float32))


def zero_queries(base,kind):
    current=base.q_target if kind=='root' else base.q_current
    return queries(base,current,current,'zero-'+kind)


def perturb_queries(zero,h):
    n=len(zero.pair_id);idx=np.repeat(np.arange(n),12)
    deltas=np.repeat(np.eye(6),2,axis=0)*np.tile([1,-1],6)[:,None]*h
    current=zero.q_current[idx];target=current+np.tile(deltas,(n,1))
    return queries(zero.take(idx),current,target,'h'+str(h))


def feature_values(rows,mode,norm,local,dtype=np.float32):
    pose=r.relative_pose(rows)
    first=np.c_[(pose[:,:3]-norm['mean'])/norm['std'],pose[:,3:]] if mode=='RAW' else (pose-local['mean'])/local['std']
    return np.c_[first,rows.conditioned[:,-6:]].astype(dtype)


def feature_tangent(jacobian,rotation,mode,norm,local):
    h=np.zeros((7,6));h[:3]=jacobian[:3];h[4:]=.5*rotation.T@jacobian[3:]
    scales=np.r_[norm['std'],np.ones(4)] if mode=='RAW' else np.asarray(local['std'])
    return h/scales[:,None]


def predict(model,values):
    dtype=next(model.parameters()).dtype
    with torch.no_grad():
        return r.decode_joints(d.predict(model,torch.tensor(values,device='cuda',dtype=dtype),'residual')).cpu().numpy()


def input_derivative(model,values):
    x=torch.tensor(values,device='cuda',dtype=next(model.parameters()).dtype,requires_grad=True)
    q=r.decode_joints(d.predict(model,x,'residual'))
    cols=[torch.autograd.grad(q[:,j].sum(),x,retain_graph=j<5)[0].detach().cpu().numpy() for j in range(6)]
    return q.detach().cpu().numpy(),np.stack(cols,axis=1)


def fd_matrix(predictions,n,h):
    values=predictions.reshape(n,6,2,6)
    return ((values[:,:,0,:]-values[:,:,1,:])/(2*h)).transpose(0,2,1)


def response_error(jacobian,k,at_prediction=None):
    left=jacobian if at_prediction is None else at_prediction
    desired=s.normalize_jacobian(jacobian,.9015)
    actual=s.normalize_jacobian(left,.9015)@k
    return float(np.linalg.norm(actual-desired)/np.linalg.norm(desired))


def stats(values):return s.stats(values)


def run():
    d.configure();tick=time.perf_counter()
    if (BASE/'registration.json').exists() or RAW.exists():raise FileExistsError('preserve prior attempt')
    cfg=d.read_json(CONFIG);prior=d.read_json(d.ROOT/cfg['source'])
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    for name in ('diagnostic3','diagnostic4','diagnostic5','diagnostic6','diagnostic7'):
        d.guard_hashes(d.read_json(d.ROOT/f'experiments/C1-06R/{name}/registration.json')['inputs'])
    paths=[Path(__file__),CONFIG,d.ROOT/'tests/c1_06r/test_local_response.py',
        d.ROOT/'docs/adr/ADR-023-c106r-local-response-audit.md',d.ROOT/cfg['source'],
        d.ROOT/cfg['normalization'],d.ROOT/cfg['geometry_contract']]
    inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in paths}
    parents=[next(c for c in prior['cells'] if c['name']==name) for name in cfg['models']]
    for c in parents:
        for key in ('checkpoint','scaler'):inputs[c[key]['path']]=c[key]['sha256']
    d.guard_hashes(inputs)
    d.write_json(BASE/'registration.json',dict(status='REGISTERED_BEFORE_MEASUREMENT',inputs=inputs,optimizer_steps=0))
    RAW.mkdir(parents=True)
    train,val=d.load_data(label_fk=True);roots=d.matched_rows(train,512)[0]['local']
    val_local=val.take(np.flatnonzero(val.mode=='local'))
    candidates=dict(train=roots.take(np.arange(64)),validation=val_local.take(np.flatnonzero(val_local.family=='main')))
    bounds=np.asarray(r.load_robot().limits);anchor_groups={};geometry={};jacobians={};rotations={}
    independent=s.IndependentJacobian(r.load_robot());fk=s.IndependentFK(r.load_robot())
    for population,rows in candidates.items():
        margin=np.minimum(np.minimum(rows.q_current-bounds[:,0],bounds[:,1]-rows.q_current),
                          np.minimum(rows.q_target-bounds[:,0],bounds[:,1]-rows.q_target)).min(1)
        chosen=rows.take(np.flatnonzero(margin>cfg['interior_margin_rad'])[:cfg['anchors_per_group']])
        assert len(chosen.pair_id)==cfg['anchors_per_group']
        for kind in cfg['anchor_types']:
            key=population+'-'+kind;zero=zero_queries(chosen,kind);anchor_groups[key]=zero
            _,_,_,check=s.preflight(zero,d.read_json(d.ROOT/cfg['geometry_contract']),r.load_robot())
            geometry[key]=check
            jacobians[key]=np.asarray([independent.jacobian(q) for q in zero.q_current])
            rotations[key]=np.asarray([fk.forward_kinematics(q)[:3,:3] for q in zero.q_current])
    d.write_json(BASE/'preflight.json',dict(status='PASS',groups=geometry,
        selection={k:v.pair_id.tolist() for k,v in anchor_groups.items()},derivative_anchor_count=128))
    print('PASS: independent FK/Jacobian/FD on128 anchors; four checkpoints fixed',flush=True)
    norm=d.read_json(d.ROOT/cfg['normalization']);cells=[]
    for parent in parents:
        name=parent['name'];mode=parent['arm'];out=RAW/name;out.mkdir()
        model=e.build_model(2026100901,512)
        model.load_state_dict(torch.load(d.ROOT/parent['checkpoint']['path'],weights_only=True)['model_state_dict']);model.eval()
        initial=d.state_hash(model);shadow=deepcopy(model).double();local=d.read_json(d.ROOT/parent['scaler']['path'])
        zero_results={};zero_raw={};derivative_results={}
        for population,base in (('train',roots.take(np.arange(parent['n_roots']))),('validation',val_local)):
            for kind in cfg['anchor_types']:
                key=population+'-'+kind;rows=zero_queries(base,kind)
                values=feature_values(rows,mode,norm,local)
                assert np.array_equal(values,z.features(rows,mode,norm,local))
                q=predict(model,values);metric=d.geometric_metrics(q,rows,details=True)
                witness=d.geometric_metrics(rows.q_current,rows)
                assert witness['profile_a']==witness['profile_b']==len(rows.pair_id)
                zero_raw[key]=dict(anchors=rows.q_current.tolist(),metric=metric)
                zero_results[key]=dict(**r.summary(metric,rows),q_bias_l2_rad=stats(np.linalg.norm(q-rows.q_current,axis=1)),
                    unchanged_current_a=witness['profile_a'])
        d.write_json(out/'zero.json',zero_raw)
        for key,zero in anchor_groups.items():
            n=len(zero.pair_id);j=jacobians[key]
            tangent=np.asarray([feature_tangent(a,b,mode,norm,local) for a,b in zip(j,rotations[key])])
            q0,b=input_derivative(model,feature_values(zero,mode,norm,local));k=np.einsum('noi,nij->noj',b[:,:,:7],tangent)
            q64,b64=input_derivative(shadow,feature_values(zero,mode,norm,local,np.float64));k64=np.einsum('noi,nij->noj',b64[:,:,:7],tangent)
            valid=(q0>=bounds[:,0]).all(1)&(q0<=bounds[:,1]).all(1)&np.isfinite(q0).all(1)
            at_prediction=np.full_like(j,np.nan);actual_errors=[]
            for i in np.flatnonzero(valid):
                at_prediction[i]=independent.jacobian(q0[i]);actual_errors.append(response_error(j[i],k[i],at_prediction[i]))
            anchor_errors=np.asarray([response_error(a,b) for a,b in zip(j,k)])
            cond=np.asarray([np.linalg.cond(s.normalize_jacobian(a,.9015)) for a in j])
            pieces=dict(anchor_q=zero.q_current,q_zero=q0,k_float32=k,k_shadow64=k64,j_anchor=j,
                j_prediction=at_prediction,valid_zero=valid,condition=cond,tangent=tangent)
            fd_results=[]
            for h in cfg['fd_steps_rad']:
                rows=perturb_queries(zero,h);values=feature_values(rows,mode,norm,local)
                q=predict(model,values);fd=fd_matrix(q,n,h)
                metric=d.geometric_metrics(q,rows,details=True)
                baseline=d.geometric_metrics(rows.q_current,rows)
                error=np.linalg.norm(fd-k,axis=(1,2))/np.maximum(1,np.linalg.norm(k,axis=(1,2)))
                fd_results.append(dict(h_rad=h,fp32_fd_vs_autograd=stats(error),small_target=r.summary(metric,rows),
                    unchanged_current=d.summarize(baseline)))
                pieces['fd_'+str(h)]=fd;pieces['pred_'+str(h)]=q;pieces['target_'+str(h)]=rows.q_target
                if h==cfg['shadow_fd_h_rad']:
                    qh=predict(shadow,feature_values(rows,mode,norm,local,np.float64));fd64=fd_matrix(qh,n,h)
                    assert np.allclose(fd64,k64,atol=cfg['shadow_gradient_atol'],rtol=cfg['shadow_gradient_rtol'])
                    shadow_error=stats(np.linalg.norm(fd64-k64,axis=(1,2)))
                    pieces['fd_shadow64']=fd64
            np.savez_compressed(out/(key+'.npz'),**pieces)
            derivative_results[key]=dict(n=n,valid_prediction_jacobians=int(valid.sum()),condition=stats(cond),
                joint_k_minus_identity=stats(np.linalg.norm(k-np.eye(6),axis=(1,2))/np.sqrt(6)),
                diagonal_gain=stats(np.diagonal(k,axis1=1,axis2=2)),
                anchor_task_relative_error=stats(anchor_errors),actual_task_relative_error_valid_only=stats(actual_errors),
                lower_condition_half_actual_error=stats([response_error(j[i],k[i],at_prediction[i]) for i in np.flatnonzero(valid&(cond<=np.median(cond)))]),
                shadow_fd_error=shadow_error,fd=fd_results,raw=z.f.artifact(out/(key+'.npz')))
        assert d.state_hash(model)==initial
        cell=dict(name=name,source_checkpoint=parent['checkpoint'],weights_unchanged=True,zero=zero_results,
                  zero_raw=z.f.artifact(out/'zero.json'),derivatives=derivative_results)
        d.write_json(BASE/(name+'.json'),cell);cells.append(cell)
        print(name+': '+', '.join(f'{key} zero A{v["profile_a"]}/{v["n"]}' for key,v in zero_results.items()),flush=True)
    d.guard_hashes(inputs);d.guard_hashes(frozen)
    d.write_json(BASE/'results.json',dict(status='COMPLETE_DIAGNOSIS',inputs=inputs,cells=cells,wall_s=time.perf_counter()-tick,
        optimizer_steps=0,old_final_raw='NOT_READ',final_test='NOT_CREATED'))


if __name__=='__main__':run()
