"""C1-05 paired experiments with frozen inputs, losses, budget and selection."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian
from neurokinematics.kinematics.custom_fk import IndependentFK
from neurokinematics.kinematics.metrics import quaternion_rotation, rotation_error, singularity_metrics
from .c104 import MLP, Rows, setup, load_data, read_json, write_json, sha, _rss_bytes, reject_shifted_labels
from .physics import PhysicsLoss, normalized_head, quaternion_matrix, combined

BASE=ROOT/'experiments/C1-05'
STAGE2=BASE/'stage2'
CONFIG=BASE/'config.json'
SOURCES=['src/neurokinematics/neural/'+s for s in ['c105.py','physics.py','training_fk.py']]
PAIRS={'E-C03':('Q','FK'),'E-C04':('FK','FK_LIMIT'),'E-C05':('FK','FK_TANH')}


def source_hashes():
    return {p:sha(ROOT/p) for p in SOURCES}


def tensor_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def configuration(variant):
    c=read_json(CONFIG)
    return next(a for a in c['matrix'] if a['id']==variant)


def prepared(rows):
    return dict(x=torch.from_numpy(rows.conditioned),y=torch.from_numpy(rows.target_normalized),
                p=torch.tensor(rows.position,dtype=torch.float32),
                R=quaternion_matrix(torch.tensor(rows.quaternion,dtype=torch.float32)))


def objective(model,variant,data,idx,physics):
    logits=model(data['x'][idx]); z=normalized_head(logits,variant)
    terms,q,t=physics.components(z,data['y'][idx],data['p'][idx],data['R'][idx])
    return combined(terms,configuration(variant)),terms,z,q,t,logits


def component_metrics(model,variant,rows,data,physics):
    idx=np.flatnonzero(rows.label_present); values={k:[] for k in ('q','p','R','lim')}
    with torch.no_grad():
        for start in range(0,len(idx),1024):
            _,parts,_,_,_,_=objective(model,variant,data,idx[start:start+1024],physics)
            for key in values: values[key].append(parts[key].double().numpy())
    values={k:np.concatenate(v) for k,v in values.items()}
    if any(not np.isfinite(v).all() for v in values.values()): raise ValueError('nonfinite metrics')
    overall={k:float(v.sum(dtype=np.float64)/len(idx)) for k,v in values.items()}
    return dict(labeled=len(idx),components=overall,by_mode={mode:dict(n=int(np.sum(rows.mode[idx]==mode)),
        components={k:float(v[rows.mode[idx]==mode].mean()) for k,v in values.items()}) for mode in ('local','wide') if np.any(rows.mode[idx]==mode)})


def gradient_record(model,parts,z,logits,arm,physics):
    params=list(model.named_parameters()); record={}
    for key,scalar in [(k,v.mean()) for k,v in parts.items()]+[('total',combined(parts,arm))]:
        gradients=torch.autograd.grad(scalar,[z]+[v for _,v in params],retain_graph=True,allow_unused=False)
        if any(not torch.isfinite(g).all() for g in gradients): raise ValueError('nonfinite component gradient')
        # Chain rule converts normalized-output gradient to physical-q gradient.
        physical=gradients[0]/(physics.upper-physics.lower).to(z)
        record[key]=dict(normalized_output_norm=float(torch.linalg.vector_norm(gradients[0])),
                         raw_q_output_norm=float(torch.linalg.vector_norm(physical)),
                         layers={name:float(torch.linalg.vector_norm(g)) for (name,_),g in zip(params,gradients[1:])})
    if arm['id']=='FK_TANH':
        v=torch.tanh(logits.detach()); dq=(physics.upper-physics.lower).to(v)/2*(1-v*v)
        record['tanh']=dict(saturated_fraction_per_joint=(v.abs()>=.99).float().mean(0).tolist(),
                            dq_dz_min_per_joint=dq.min(0).values.tolist(),dq_dz_mean_per_joint=dq.mean(0).tolist())
    return record


def save_checkpoint(path,model,optimizer,variant,seed,epoch,val_loss,experiment):
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(schema='c105-v1',variant=variant,experiment=experiment,training_seed=seed,
        selected_epoch=epoch,validation_q_loss=val_loss,config_sha256=sha(CONFIG),
        input_manifest_sha256=sha(BASE/'input-hashes.json'),source_hashes=source_hashes(),
        c102_manifest_sha256=sha(ROOT/'experiments/C1-02/dataset-manifest.json'),
        normalization_sha256=sha(ROOT/'experiments/C1-02/normalization.json'),robot_hashes=load_robot().hashes,
        architecture='13-256-256-256-6_SiLU',output_head=configuration(variant)['head'],
        identity=read_json(CONFIG)['identity'],model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict())
    temp=path.with_suffix('.tmp'); torch.save(metadata,temp); temp.replace(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),accessible=True,epoch=epoch,storage='LOCAL_ONLY')


def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    expected=dict(schema='c105-v1',config_sha256=sha(CONFIG),input_manifest_sha256=sha(BASE/'input-hashes.json'),
        source_hashes=source_hashes(),c102_manifest_sha256=sha(ROOT/'experiments/C1-02/dataset-manifest.json'),
        normalization_sha256=sha(ROOT/'experiments/C1-02/normalization.json'),robot_hashes=load_robot().hashes,
        architecture='13-256-256-256-6_SiLU',identity=read_json(CONFIG)['identity'])
    for key,value in expected.items():
        if payload.get(key)!=value: raise ValueError('checkpoint metadata drift: '+key)
    variant=payload['variant']; arm=configuration(variant)
    if payload['output_head']!=arm['head'] or payload['training_seed'] not in read_json(CONFIG)['training']['seeds']:
        raise ValueError('checkpoint head/seed drift')
    model=MLP('conditioned'); model.load_state_dict(payload['model_state_dict'],strict=True); model.eval()
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()): raise ValueError('nonfinite checkpoint')
    return model,{k:v for k,v in payload.items() if k not in ('model_state_dict','optimizer_state_dict')}


def infer(model,variant,features):
    model.eval(); predictions=[]
    with torch.no_grad():
        for start in range(0,len(features),1024):
            logits=model(torch.from_numpy(features[start:start+1024]))
            predictions.append(normalized_head(logits,variant).numpy())
    z=np.concatenate(predictions)
    lo,hi=np.asarray(load_robot().limits).T
    return lo+z.astype(np.float64)*(hi-lo),z


def quantiles(values,*,invalid=0):
    values=np.asarray(values,dtype=float)
    if not invalid:
        return dict(n=len(values),**{key:float(np.percentile(values,p)) if len(values) else None for key,p in [('median',50),('p95',95),('p99',99)]})
    ordered=np.sort(np.r_[values,np.full(invalid,np.inf)])
    result={'n':len(ordered)}
    for key,p in [('median',.5),('p95',.95),('p99',.99)]:
        value=float(ordered[math.ceil(p*len(ordered))-1])
        result[key]=value if math.isfinite(value) else None; result[key+'_unbounded']=not math.isfinite(value)
    return result


def full_quantiles(values,total):
    # Empirical nearest-rank for the full denominator, including when invalid=0.
    ordered=np.sort(np.r_[values,np.full(total-len(values),np.inf)])
    result={'n':total}
    for key,p in [('median',.5),('p95',.95),('p99',.99)]:
        value=float(ordered[math.ceil(p*total)-1]) if total else float('inf')
        result[key]=value if math.isfinite(value) else None; result[key+'_unbounded']=not math.isfinite(value)
    return result


def summarize_rows(rows):
    groups=defaultdict(list)
    for r in rows:
        for key in ['overall','mode/'+r['mode'],'family/'+r['family'],'label/'+str(r['label_present']),
                    f"family/{r['family']}/mode/{r['mode']}/label/{r['label_present']}"]:
            groups[key].append(r)
    result={}
    for key,records in groups.items():
        n=len(records); valid=sum(r['in_limits'] for r in records)
        group=dict(n=n,labeled=sum(r['label_present'] for r in records),valid_raw=valid,
            nonfinite=sum(not r['finite'] for r in records),out_of_limits=sum(r['finite'] and not r['in_limits'] for r in records),
            profile_a=sum(r['profile_a'] for r in records),profile_b=sum(r['profile_b'] for r in records),geometry_coverage=valid/n)
        for profile in ('a','b'): group['profile_'+profile+'_rate']=group['profile_'+profile]/n
        for metric in ('position_m','orientation_deg'):
            v=[r[metric] for r in records if r[metric] is not None]
            group[metric+'_valid_only']=quantiles(v); group[metric+'_full']=full_quantiles(v,n)
        for metric in ('q_loss','q_mae_rad','q_l2_rad','sigma_min'):
            v=[r[metric] for r in records if r[metric] is not None]
            group[metric]=dict(quantiles(v),mean=float(np.mean(v)) if v else None)
        result[key]=group
    return result


def evaluate(model,variant,rows,path,*,checkpoint=None):
    if rows.split!='validation': raise ValueError('evaluation test seal')
    if path.exists(): raise ValueError('retain previous evaluation')
    raw,z=infer(model,variant,rows.conditioned)
    inputs=load_robot(); lo,hi=np.asarray(inputs.limits).T
    fk=PinocchioFK(inputs); cross=IndependentFK(inputs); jac=PinocchioJacobian(inputs)
    records=[]; max_cross=0.
    for i,q in enumerate(raw):
        finite=bool(np.isfinite(q).all()); valid=bool(finite and np.all(q>=lo) and np.all(q<=hi))
        labeled=bool(rows.label_present[i]); qlist=q.tolist() if finite else None
        record=dict(pair_id=str(rows.pair_id[i]),family=str(rows.family[i]),mode=str(rows.mode[i]),label_present=labeled,
            finite=finite,in_limits=valid,q_raw_rad=qlist,q_fk_input_rad=qlist if valid else None,q_evaluation_rad=qlist,
            fk_status='EVALUATED' if valid else 'REJECTED_RAW_INVALID',position_m=None,orientation_deg=None,
            q_loss=None,q_mae_rad=None,q_l2_rad=None,sigma_min=None,profile_a=False,profile_b=False,collision='NOT_CHECKED')
        if finite and labeled:
            difference=q-rows.q_target[i]
            record.update(q_loss=float(np.sum((z[i]-rows.target_normalized[i])**2)),q_mae_rad=float(np.mean(np.abs(difference))),q_l2_rad=float(np.linalg.norm(difference)))
        if valid:
            t=fk.reference_forward_kinematics(q)
            pe=float(np.linalg.norm(t[:3,3]-rows.position[i])); re=math.degrees(rotation_error(t[:3,:3],quaternion_rotation(rows.quaternion[i])))
            record.update(position_m=pe,orientation_deg=re,profile_a=pe<=.002 and re<=1.,profile_b=pe<=.001 and re<=.5,
                          sigma_min=singularity_metrics(jac.jacobian(q),.9015)['sigma_min'])
            if len([r for r in records if r['in_limits']])<8:
                delta=float(np.linalg.norm(t-cross.forward_kinematics(q)))
                max_cross=max(max_cross,delta)
                if delta>1e-9: raise ValueError('independent FK spot drift')
        records.append(record)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        for r in records: stream.write(json.dumps(r,allow_nan=False)+'\n')
    summary=dict(status='MEASURED',variant=variant,rows=len(records),checkpoint=checkpoint,
                 raw_sha256=sha(path),raw_bytes=path.stat().st_size,breakdowns=summarize_rows(records),max_independent_fk_spot=max_cross)
    write_json(path.with_suffix('.summary.json'),summary)
    return summary


def train_pair(experiment,seed,attempt,*,pilot=False):
    config=read_json(CONFIG)
    if seed not in config['training']['seeds'] or experiment not in PAIRS: raise ValueError('unregistered experiment/seed')
    train,validation=load_data(label_fk=True)
    if pilot:
        ids=read_json(BASE/'pilot-pair-ids.json')
        train=train.take(np.flatnonzero(np.isin(train.pair_id,ids['train'])))
        validation=validation.take(np.flatnonzero(np.isin(validation.pair_id,ids['validation'])))
        if len(train.pair_id)!=192 or len(validation.pair_id)!=96: raise ValueError('pilot IDs drift')
    expected=(192,96) if pilot else (15204,3249)
    if (int(train.label_present.sum()),int(validation.label_present.sum()))!=expected: raise ValueError('inventory drift')
    variants=PAIRS[experiment]; setup(seed)
    output=STAGE2/('pilot' if pilot else experiment)/f'seed-{seed}'/attempt
    output.mkdir(parents=True,exist_ok=False)
    weight_root=ROOT/'data/generated/C1-05'/('pilot' if pilot else 'v1')/experiment/f'seed-{seed}'/attempt
    models={}; optimizers={}; initial_hash={}
    for variant in variants:
        torch.manual_seed(seed); model=MLP('conditioned'); models[variant]=model
        initial_hash[variant]=tensor_hash(model)
        optimizers[variant]=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    if len(set(initial_hash.values()))!=1: raise ValueError('unpaired initialization')
    td,vd=prepared(train),prepared(validation); physics=PhysicsLoss()
    indices=np.flatnonzero(train.label_present); rng=np.random.Generator(np.random.PCG64(seed))
    best={v:math.inf for v in variants}; best_epoch={v:0 for v in variants}; stale={v:0 for v in variants}; best_paths={}
    batch=192 if pilot else 1024; epochs=20 if pilot else 200
    logs={v:(output/(v+'-epochs.jsonl')).open('w',encoding='utf-8',newline='\n') for v in variants}
    pilot_log=(output/'raw-pilot.jsonl').open('w',encoding='utf-8',newline='\n') if pilot else None
    start=time.monotonic(); peak=_rss_bytes(); step=0; arm_wall={v:0. for v in variants}
    initial={v:dict(train=component_metrics(models[v],v,train,td,physics),validation=component_metrics(models[v],v,validation,vd,physics)) for v in variants}
    write_json(output/'start.json',dict(seed=seed,experiment=experiment,variants=variants,initial_state_sha256=initial_hash,initial_metrics=initial,
        source_hashes=source_hashes(),config_sha256=sha(CONFIG),training_pair_order_sha256=hashlib.sha256(('\n'.join(train.pair_id[indices])+'\n').encode()).hexdigest()))
    try:
        for epoch in range(1,epochs+1):
            order=indices[rng.permutation(len(indices))]
            order_sha=hashlib.sha256(('\n'.join(train.pair_id[order])+'\n').encode()).hexdigest()
            diagnostics={}; combined_norms={v:[] for v in variants}; update_norms={v:[] for v in variants}
            for offset in range(0,len(indices),batch):
                selected=order[offset:offset+batch]
                for variant in variants:
                    tick=time.monotonic(); model=models[variant]; opt=optimizers[variant]; opt.zero_grad(set_to_none=True)
                    total,parts,z,q,t,logits=objective(model,variant,td,selected,physics)
                    if not torch.isfinite(total): raise ValueError('nonfinite loss')
                    if offset==0:
                        diagnostics[variant]=gradient_record(model,parts,z,logits,configuration(variant),physics)
                        # π proximity uses a diagnostic geodesic angle; LR remains chordal.
                        cosine=((td['R'][selected].transpose(-1,-2)@t[:,:3,:3]).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
                        near=torch.acos(cosine.detach().clamp(-1,1))*180/torch.pi>=175
                        diagnostics[variant]['near_pi']=dict(n=int(near.sum()),total=len(near),LR_mean=float(parts['R'][near].detach().mean()) if near.any() else None)
                        if near.any():
                            gnear=torch.autograd.grad(parts['R'][near].mean(),z,retain_graph=True)[0]
                            diagnostics[variant]['near_pi']['LR_raw_q_gradient_norm']=float(torch.linalg.vector_norm(gnear/(physics.upper-physics.lower).to(z)))
                        if variant=='FK_TANH':
                            mask=train.family[selected]=='boundary'
                            if mask.any():
                                tanh=torch.tanh(logits[mask].detach())
                                diagnostics[variant]['tanh']['boundary']=dict(n=int(mask.sum()),saturated_fraction_per_joint=(tanh.abs()>=.99).float().mean(0).tolist(),
                                    dq_dz_mean_per_joint=(((physics.upper-physics.lower).to(tanh)/2)*(1-tanh*tanh)).mean(0).tolist())
                        out=((q<physics.lower.to(q))|(q>physics.upper.to(q))).any(-1)
                        diagnostics[variant]['raw']=dict(rows=len(q),out_of_limits=int(out.sum()),nonfinite=int((~torch.isfinite(q)).sum()))
                    before=[p.detach().clone() for p in model.parameters()]
                    total.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('bad combined gradients')
                    combined_norms[variant].append(math.sqrt(sum(float((p.grad.double()**2).sum()) for p in model.parameters())))
                    opt.step()
                    if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError('nonfinite parameter')
                    update_norms[variant].append(math.sqrt(sum(float(((p.detach()-b).double()**2).sum()) for p,b in zip(model.parameters(),before))))
                    arm_wall[variant]+=time.monotonic()-tick
                step+=1
            for variant in variants:
                tick=time.monotonic(); model=models[variant]
                tr=component_metrics(model,variant,train,td,physics); va=component_metrics(model,variant,validation,vd,physics)
                value=va['components']['q']
                if value<best[variant]:
                    best[variant]=value; best_epoch[variant]=epoch; stale[variant]=0
                    best_paths[variant]=save_checkpoint(weight_root/variant/'best.pt',model,optimizers[variant],variant,seed,epoch,value,experiment)
                else: stale[variant]+=1
                record=dict(epoch=epoch,optimizer_steps=step,variant=variant,seed=seed,train=tr,validation=va,
                    permutation_sha256=order_sha,gradient_probe='first effective batch before update; all batches combined norm below',gradients=diagnostics[variant],
                    combined_gradient_norms=combined_norms[variant],parameter_update_norms=update_norms[variant],best_epoch=best_epoch[variant],best_q_loss=best[variant])
                arm=configuration(variant)
                for metrics in (tr,va):
                    metrics['weighted_components']={k:metrics['components'][k]*weight for k,weight in [('q',1),('p',arm['lambda_p']),('R',arm['lambda_R']),('lim',arm['lambda_lim'])]}
                logs[variant].write(json.dumps(record,allow_nan=False)+'\n'); logs[variant].flush()
                if pilot:
                    for rows,data in [(train,td),(validation,vd)]:
                        with torch.no_grad():
                            _,parts,_,q,_,logits=objective(model,variant,data,np.arange(len(rows.pair_id)),physics)
                        for i,ident in enumerate(rows.pair_id):
                            qr=q[i].tolist()
                            pilot_log.write(json.dumps(dict(epoch=epoch,variant=variant,split=rows.split,pair_id=str(ident),q_raw_rad=qr,q_fk_input_rad=qr,q_evaluation_rad=qr,
                                losses={k:float(v[i]) for k,v in parts.items()}),allow_nan=False)+'\n')
                    pilot_log.flush()
                arm_wall[variant]+=time.monotonic()-tick
            peak=max(peak,_rss_bytes())
            if peak>4*1024**3: raise RuntimeError('RAM cap')
            if pilot and time.monotonic()-start>1800: raise RuntimeError('pilot wall cap')
            if any(t>7200 for t in arm_wall.values()): raise RuntimeError('per-model wall cap')
            if not pilot and all(stale[v]>=20 for v in variants): break
        last={v:save_checkpoint(weight_root/v/'last.pt',models[v],optimizers[v],v,seed,epoch,
              component_metrics(models[v],v,validation,vd,physics)['components']['q'],experiment) for v in variants}
        evaluations={}
        for variant in variants:
            model,metadata=load_checkpoint(Path(best_paths[variant]['path']))
            evaluations[variant]=evaluate(model,variant,validation,output/(variant+'-validation.jsonl'),checkpoint=best_paths[variant])
        result=dict(status='COMPLETE',experiment=experiment,seed=seed,pilot=pilot,epochs=epoch,steps_per_model=step,variants=variants,
                    best_epoch=best_epoch,best_validation_q_loss=best,best_checkpoints=best_paths,last_checkpoints=last,
                    initial_state_sha256=initial_hash,source_hashes=source_hashes(),config_sha256=sha(CONFIG),
                    wall_s=time.monotonic()-start,per_arm_training_wall_s=arm_wall,peak_rss_bytes=peak,
                    evaluation={v:e['breakdowns']['overall'] for v,e in evaluations.items()},train_labeled=len(indices),validation_rows=len(validation.pair_id))
        write_json(output/'summary.json',result)
        return result
    except Exception as exc:
        write_json(output/'failure.json',dict(status='FAIL',error=repr(exc),completed_steps=step,wall_s=time.monotonic()-start,source_hashes=source_hashes()))
        raise
    finally:
        for stream in logs.values(): stream.close()
        if pilot_log: pilot_log.close()
