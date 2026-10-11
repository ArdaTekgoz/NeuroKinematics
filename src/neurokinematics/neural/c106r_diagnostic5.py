"""Frozen-checkpoint, train-only geometry and gradient diagnosis."""
from pathlib import Path
import math
import numpy as np
import torch
from . import c106r_diagnostic4 as e
from .training_fk import TrainingFK, DOMAIN
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.finite_difference import CentralDifference, log_so3
from neurokinematics.kinematics.metrics import normalized_difference, normalize_jacobian

d, r = e.d, e.r
BASE = d.ROOT/'experiments/C1-06R/diagnostic5'
CONFIG = BASE/'config.json'


def stats(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    good = x[np.isfinite(x)]
    out = dict(n=len(x), finite=len(good))
    out.update({k:float(v) for k,v in zip(('min','median','p95','max'),np.quantile(good,[0,.5,.95,1]))} if len(good) else {})
    return out


def correlation(a,b):
    a,b = np.asarray(a),np.asarray(b)
    mask = np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[mask],b[mask])[0,1]) if mask.sum()>1 and a[mask].std()>0 and b[mask].std()>0 else None


def pose_delta(predicted, teacher):
    return np.r_[predicted[:3,3]-teacher[:3,3], log_so3(predicted[:3,:3]@teacher[:3,:3].T)]


def cosine(a,b):
    denominator = torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float(torch.dot(a,b)/denominator) if denominator>0 else None


def gradient_probe(model, x, y, position, rotation):
    z = d.predict(model,x,'residual')
    q = r.decode_joints(z)
    fk = TrainingFK.from_frozen(domain=DOMAIN)
    t = fk(q,robot_id=fk.robot_id,joint_names=fk.joint_names)
    terms = dict(Q=((z-y)**2).sum(-1).mean(),
                 P=(((t[:,:3,3]-position)/.9015)**2).sum(-1).mean(),
                 R=((t[:,:3,:3]-rotation)**2).sum((-1,-2)).mean()/8)
    params = tuple(model.parameters())
    vectors, outputs, layers = {}, {}, {}
    for name,loss in terms.items():
        g = torch.autograd.grad(loss, (*params,z), retain_graph=True)
        assert all(torch.isfinite(v).all() for v in g)
        vectors[name] = torch.cat([v.flatten() for v in g[:-1]])
        outputs[name] = g[-1].detach()
        layers[name] = {k:dict(norm=float(torch.linalg.vector_norm(v)), zeros=int((v==0).sum()), elements=v.numel())
                        for (k,_),v in zip(model.named_parameters(),g[:-1])}
    parameter_cosines = {a+'_'+b:cosine(vectors[a],vectors[b]) for a,b in (('Q','P'),('Q','R'),('P','R'))}
    per_row = {}
    for a,b in (('Q','P'),('Q','R'),('P','R')):
        den = torch.linalg.vector_norm(outputs[a],dim=1)*torch.linalg.vector_norm(outputs[b],dim=1)
        value = torch.where(den>0,(outputs[a]*outputs[b]).sum(1)/den,torch.nan)
        per_row[a+'_'+b] = value.cpu().numpy()
    return dict(losses={k:float(v.detach()) for k,v in terms.items()},
                parameter_norms={k:float(torch.linalg.vector_norm(v)) for k,v in vectors.items()},
                parameter_cosines=parameter_cosines,layer_gradients=layers),per_row


def preflight(rows, cfg, robot):
    pin, independent, fd, other_fk = PinocchioJacobian(robot),IndependentJacobian(robot),CentralDifference(robot),IndependentFK(robot)
    jc, jt, ts = [],[],[]
    max_fk_p=max_fk_r=max_j=0.
    for current,teacher in zip(rows.q_current,rows.q_target):
        matrices=[]
        for q in (current,teacher):
            t=pin.reference_forward_kinematics(q)
            cross=other_fk.forward_kinematics(q)
            max_fk_p=max(max_fk_p,float(np.linalg.norm(t[:3,3]-cross[:3,3])))
            max_fk_r=max(max_fk_r,float(np.linalg.norm(t[:3,:3]-cross[:3,:3])))
            a,b=pin.jacobian(q),independent.jacobian(q)
            max_j=max(max_j,normalized_difference(a,b,cfg['jacobian']['characteristic_length_m']))
            matrices.append(a)
        jc.append(matrices[0]);jt.append(matrices[1]);ts.append(t)
    assert max_fk_p <= cfg['fk']['position_m'] and max_fk_r <= cfg['fk']['rotation_frobenius']
    assert max_j <= cfg['jacobian']['normalized_error_threshold']
    limits=np.asarray(robot.limits)
    margin=np.minimum(rows.q_target-limits[:,0],limits[:,1]-rows.q_target).min(1)
    selected=np.flatnonzero(margin>2e-5)[:cfg['jacobian']['fd_count']]
    assert len(selected)==cfg['jacobian']['fd_count']
    errors=[normalized_difference(jt[i],fd.jacobian(rows.q_target[i],cfg['jacobian']['fd_h_rad']),.9015) for i in selected]
    assert max(errors)<=cfg['jacobian']['normalized_error_threshold']
    ts=np.asarray(ts)
    label_p=float(np.linalg.norm(ts[:,:3,3]-rows.position,axis=1).max())
    label_r=float(np.linalg.norm(ts[:,:3,:3]-np.asarray([r.quaternion_rotation(q) for q in rows.quaternion]),axis=(1,2)).max())
    assert label_p<=cfg['fk']['position_m'] and label_r<=cfg['fk']['rotation_frobenius']
    return np.asarray(jc),np.asarray(jt),ts,dict(status='PASS',fk_rows=2*len(rows.pair_id),jacobian_rows=2*len(rows.pair_id),
        independent_fk_max_position_m=max_fk_p,independent_fk_max_rotation_frobenius=max_fk_r,
        geometric_vs_pin_max_normalized=max_j,fd=dict(n=len(selected),max_normalized=max(errors),pair_ids=rows.pair_id[selected].tolist()),
        teacher_vs_target_max_position_m=label_p,teacher_vs_target_max_rotation_frobenius=label_r)


def run():
    d.configure()
    if (BASE/'registration.json').exists():raise FileExistsError('preserve prior attempt')
    cfg=d.read_json(CONFIG)
    previous=d.read_json(d.ROOT/cfg['checkpoint_source'])
    d.guard_hashes(previous['inputs'])
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files']
    d.guard_hashes(frozen)
    files=[Path(__file__),CONFIG,d.ROOT/'tests/c1_06r/test_diagnostic5.py',
           d.ROOT/'docs/adr/ADR-019-c106r-train-geometry-diagnosis.md',d.ROOT/cfg['checkpoint_source']]
    files.extend(d.ROOT/'src/neurokinematics/kinematics'/n for n in ('jacobian.py','pinocchio_jacobian.py','finite_difference.py','metrics.py'))
    files.extend([d.ROOT/'src/neurokinematics/neural/training_fk.py',d.ROOT/'experiments/F0-03/config.json',d.ROOT/'experiments/C1-03/config.json'])
    inputs={str(p.relative_to(d.ROOT)):d.sha(p) for p in files}
    d.write_json(BASE/'registration.json',dict(status='REGISTERED_BEFORE_MEASUREMENT',inputs=inputs,training='NONE'))
    train,_=d.load_data(label_fk=True)
    rows=d.matched_rows(train,cfg['n'])[0]['local']
    robot=r.load_robot();bounds=np.asarray(robot.limits);span=bounds[:,1]-bounds[:,0]
    jc,jt,teacher_t,checks=preflight(rows,cfg,robot)
    d.write_json(BASE/'preflight.json',checks)
    print('PASS:4096 FK/Jacobian checks,32 finite differences,2048 teacher poses',flush=True)
    relative=r.relative_pose(rows)
    x_values=r.features(rows,'relative',d.read_json(d.ROOT/cfg['normalization']),relative)
    condition=np.asarray([np.linalg.cond(normalize_jacobian(j,.9015)) for j in jt])
    margin=np.minimum((rows.q_target-bounds[:,0])/span,(bounds[:,1]-rows.q_target)/span).min(1)
    correction=np.linalg.norm(rows.q_target-rows.q_current,axis=1)
    grouping=dict(condition=condition,limit_margin_normalized=margin,correction_l2_rad=correction)
    feature_stats=dict(mean=x_values.mean(0).tolist(),std=x_values.std(0).tolist(),
        relative_position_std_m=relative[:,:3].std(0).tolist(),relative_quaternion_w=stats(relative[:,3]),
        relative_rotation_deg=stats(np.rad2deg(2*np.arccos(np.clip(relative[:,3],-1,1)))),
        max_local_joint_correction_rad=float(np.abs(rows.q_target-rows.q_current).max()),
        exact_duplicate_inputs=len(x_values)-len(np.unique(x_values,axis=0)))
    x=torch.tensor(x_values,device='cuda');y=torch.tensor(rows.target_normalized,device='cuda')
    p=torch.tensor(rows.position,device='cuda');rot=torch.tensor(np.asarray([r.quaternion_rotation(q) for q in rows.quaternion]),device='cuda')
    cells=[]
    for name in cfg['models']:
        cell=next(c for c in previous['cells'] if c['name']==name)
        for k in ('raw','checkpoint'):assert d.sha(d.ROOT/cell[k]['path'])==cell[k]['sha256']
        payload=torch.load(d.ROOT/cell['checkpoint']['path'],weights_only=True)
        assert payload['contract']['pair_ids']==rows.pair_id.tolist()
        model=e.build_model(cfg['seed'],cell['width']);model.load_state_dict(payload['model_state_dict']);model.eval()
        initial=d.state_hash(model)
        recorded=d.read_json(d.ROOT/cell['raw']['path'])['train']
        assert r.evaluate(model,'residual',rows,x_values)==recorded
        full,grows=gradient_probe(model,x,y,p,rot)
        batches=[gradient_probe(model,x[i:i+128],y[i:i+128],p[i:i+128],rot[i:i+128])[0] for i in range(0,len(x),128)]
        activations={}
        with torch.no_grad():
            value=x
            for i,layer in enumerate(model.layers):
                value=layer(value)
                if isinstance(layer,torch.nn.SiLU):
                    assert torch.isfinite(value).all()
                    activations[str(i)]=dict(rms=float(value.square().mean().sqrt()),unit_std=stats(value.std(0).cpu()),zero_elements=int((value==0).sum()),elements=value.numel())
        pin=PinocchioJacobian(robot);ind=IndependentJacobian(robot)
        records=[];pred_jmax=0.
        for i,m in enumerate(recorded['rows']):
            q=np.asarray(m['q_rad']);dq=q-rows.q_target[i]
            out=dict(pair_id=m['pair_id'],valid=m['valid'],profile_a=m['profile_a'],q_error_rad=dq.tolist(),
                q_loss=float(np.sum((dq/span)**2)),condition=float(condition[i]),limit_margin=float(margin[i]),correction_l2_rad=float(correction[i]),
                output_gradient_cosines={k:float(v[i]) if np.isfinite(v[i]) else None for k,v in grows.items()})
            if m['valid']:
                pred_t=pin.reference_forward_kinematics(q);actual=pose_delta(pred_t,teacher_t[i]);linear=jt[i]@dq
                jp=pin.jacobian(q);pred_jmax=max(pred_jmax,normalized_difference(jp,ind.jacobian(q),.9015))
                lp=jp@dq
                out.update(position_m=float(np.linalg.norm(actual[:3])),orientation_deg=float(np.rad2deg(np.linalg.norm(actual[3:]))),
                    severity_a=max(np.linalg.norm(actual[:3])/.002,np.linalg.norm(actual[3:])/math.radians(1)),
                    teacher_taylor_position_residual_m=float(np.linalg.norm(actual[:3]-linear[:3])),
                    teacher_taylor_rotation_residual_deg=float(np.rad2deg(np.linalg.norm(actual[3:]-linear[3:]))),
                    prediction_taylor_position_residual_m=float(np.linalg.norm(actual[:3]-lp[:3])),
                    prediction_taylor_rotation_residual_deg=float(np.rad2deg(np.linalg.norm(actual[3:]-lp[3:]))),
                    linear_position_m=float(np.linalg.norm(linear[:3])),linear_orientation_deg=float(np.rad2deg(np.linalg.norm(linear[3:]))),
                    per_joint_linear_position_m=np.linalg.norm(jt[i,:3]*dq[None,:],axis=0).tolist(),
                    per_joint_linear_rotation_deg=np.rad2deg(np.linalg.norm(jt[i,3:]*dq[None,:],axis=0)).tolist())
            records.append(out)
        assert pred_jmax<=cfg['jacobian']['normalized_error_threshold']
        good=[m for m in records if m['valid']]
        groups={}
        for group,values in grouping.items():
            order=np.argsort(values,kind='stable');parts=[]
            for idx in np.array_split(order,4):
                part=[records[i] for i in idx];valid=[m for m in part if m['valid']]
                parts.append(dict(n=len(part),range=[float(values[idx].min()),float(values[idx].max())],valid=len(valid),
                    profile_a=sum(m['profile_a'] for m in part),q_loss=stats([m['q_loss'] for m in part]),
                    valid_only_severity_a=stats([m['severity_a'] for m in valid])))
            groups[group]=parts
        raw=d.ROOT/'data/generated/C1-06R/diagnostic5'/f'{name}-rows.json'
        if raw.exists():raise FileExistsError(raw)
        d.write_json(raw,dict(rows=records))
        metrics={k:stats([m[k] for m in good]) for k in ('position_m','orientation_deg','severity_a',
            'teacher_taylor_position_residual_m','teacher_taylor_rotation_residual_deg',
            'prediction_taylor_position_residual_m','prediction_taylor_rotation_residual_deg')}
        correlations={k:correlation([m[a] for m in good],[m[b] for m in good]) for k,a,b in (
            ('Q_vs_Aseverity','q_loss','severity_a'),('condition_vs_Aseverity','condition','severity_a'),
            ('linear_vs_actual_position','linear_position_m','position_m'),('linear_vs_actual_orientation','linear_orientation_deg','orientation_deg'))}
        gradient_summary={k:dict(stats=stats(v),negative=int((v<0).sum())) for k,v in grows.items()}
        assert d.state_hash(model)==initial
        result=dict(name=name,n=len(records),valid=len(good),profile_a=cell['train']['profile_a'],
            weights_unchanged=True,replay='EXACT_MATCH',prediction_jacobian_max_normalized=pred_jmax,
            valid_only=metrics,valid_only_correlations=correlations,groups=groups,gradients_full=full,gradients_batches=batches,
            output_gradient_cosines=gradient_summary,activations=activations,
            per_joint_linear_position_mean_m=np.mean([m['per_joint_linear_position_m'] for m in good],axis=0).tolist(),
            per_joint_linear_rotation_mean_deg=np.mean([m['per_joint_linear_rotation_deg'] for m in good],axis=0).tolist(),
            raw=dict(path=str(raw.relative_to(d.ROOT)),sha256=d.sha(raw)))
        d.write_json(BASE/f'{name}.json',result);cells.append(result)
        print(f'{name}: fixed checkpoint analyzed on2048 train rows; no optimizer step',flush=True)
    d.guard_hashes(inputs);d.guard_hashes(frozen)
    d.write_json(BASE/'results.json',dict(status='COMPLETE_DIAGNOSIS',preflight=checks,feature_statistics=feature_stats,
        condition=stats(condition),cells=cells,inputs=inputs,training='NONE',validation_inference='NOT_RUN',final_test='NOT_CREATED'))


if __name__=='__main__':run()
