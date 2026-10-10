"""Freeze research-only candidates and check their current-environment inference."""
from pathlib import Path
import json
import math
import platform
import subprocess
import time
import numpy as np
import torch
from neurokinematics.neural import c105
from neurokinematics.neural import c106r_centered as c

d,r=c.d,c.r
BASE=d.ROOT/'experiments/C1-07/preparation'
RAW=d.ROOT/'data/generated/C1-07/preparation'


def artifact(path):
    path=Path(path)
    if not path.is_absolute():path=d.ROOT/path
    return dict(path=path.relative_to(d.ROOT).as_posix(),sha256=d.sha(path),bytes=path.stat().st_size)


def check(entry):
    path=d.ROOT/entry['path']
    assert d.sha(path)==entry['sha256'],entry['path']
    return path


def main():
    start=time.perf_counter();torch.set_num_threads(1)
    if (BASE/'registration.json').exists() or RAW.exists():raise FileExistsError('preserve prior attempt')
    cfg=d.read_json(BASE/'config.json');old=d.read_json(d.ROOT/cfg['primary_source'])
    recent=d.read_json(d.ROOT/cfg['secondary_source']);frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files']
    d.guard_hashes(frozen)
    for n in range(3,10):d.guard_hashes(d.read_json(d.ROOT/f'experiments/C1-06R/diagnostic{n}/registration.json')['inputs'])
    prior_manifest=d.ROOT/'experiments/C1-06R/DELIVERY_SHA256SUMS';lines=prior_manifest.read_text().splitlines()
    for line in lines:
        digest,name=line.split('  ',1);assert d.sha(prior_manifest.parent/name)==digest,name
    assert old['model_family']=='FK_TANH' and old['H2']=='REJECTED'
    acceptance=d.read_json(check(old['acceptance']));assert acceptance['T_C05']=='PASS' and acceptance['direct_ik_decision']=='NO_GO'
    latest=d.read_json(d.ROOT/'experiments/C1-06R/diagnostic9/audit.json')
    assert latest['status']=='PASS' and latest['continuation']['pass_gate'] is False
    candidates=[]
    for seed in cfg['primary_seeds']:
        cp=next(x for x in old['checkpoints'] if x['seed']==seed);check(cp)
        candidates.append(dict(id=f'FK_TANH-{seed}',family='FK_TANH',role='PRIMARY_RESEARCH_CANDIDATE',seed=seed,
            checkpoint=artifact(cp['path']),selected_epoch=cp['epoch'],normalization=artifact(old['normalization']['path']),
            source_config=artifact('experiments/C1-05/config.json'),training_distribution='C1-02 labeled local+wide; all families',
            selection='Historical C1-05 validation Q-loss best, fixed by C1-06 handoff before this preparation'))
    for seed in cfg['secondary_seeds']:
        cell=next(x for x in recent['cells'] if x['seed']==seed and x['arm']=='RAW');check(cell['checkpoint'])
        candidates.append(dict(id=f'LOCAL_RAW-{seed}',family='LOCAL_RAW',role=cfg['secondary_role'],seed=seed,
            checkpoint=artifact(cell['checkpoint']['path']),selected_step=5000,
            normalization=artifact('experiments/C1-06R/diagnostic3/normalization.json'),
            source_config=artifact('experiments/C1-06R/diagnostic9/config.json'),training_distribution='512 main train roots; 4096 local directions only',
            selection='Terminal checkpoint, all three control seeds; no seed ranking or new training'))
    paths=[Path(__file__),BASE/'config.json',d.ROOT/'docs/adr/ADR-025-core-research-stop-and-hybrid-handoff.md',
        d.ROOT/cfg['primary_source'],d.ROOT/cfg['secondary_source'],check(old['acceptance']),
        d.ROOT/'experiments/C1-06R/diagnostic9/audit.json',prior_manifest,d.ROOT/'pixi.lock',
        d.ROOT/'experiments/C1-02/dataset-manifest.json',d.ROOT/'experiments/C1-03/stage2/acceptance.json',
        d.ROOT/'experiments/C1-06R/requirements-win-cu128.lock',d.ROOT/'experiments/C1-03/stage2/runtime-supplement.lock',
        d.ROOT/'scripts/c107_command.py']
    inputs={p.relative_to(d.ROOT).as_posix():d.sha(p) for p in paths}
    for x in candidates:
        for key in ('checkpoint','normalization','source_config'):inputs[x[key]['path']]=x[key]['sha256']
    inputs.update(c105.load_robot().hashes)
    d.write_json(BASE/'registration.json',dict(status='FIXED_BEFORE_PREPARATION_INFERENCE',inputs=inputs,candidate_ids=[x['id'] for x in candidates]))
    RAW.mkdir(parents=True)
    _,val=d.load_data(label_fk=True);robot=r.load_robot();independent=c.s.IndependentFK(robot)
    reference=r.PinocchioFK(robot);bounds=np.asarray(robot.limits);norm=d.read_json(d.ROOT/'experiments/C1-06R/diagnostic3/normalization.json')
    rel=r.features(val,'relative',norm);maxp=maxrot=0.;checked=0;records=[]
    for candidate in candidates:
        path=check(candidate['checkpoint'])
        if candidate['family']=='FK_TANH':
            model,metadata=c105.load_checkpoint(path);assert metadata['training_seed']==candidate['seed'] and metadata['variant']=='FK_TANH'
            q,_=c105.infer(model,'FK_TANH',val.conditioned)
            candidate['architecture']='13-256-256-256-6 SiLU; tanh normalized absolute output'
            candidate['output_contract']='z=(tanh(logits)+1)/2; q=lower+float64(z)*(upper-lower), historical c105.infer'
            candidate['input_contract']='absolute target position train-zscore; canonical target quaternion wxyz; normalized current joints'
        else:
            payload=torch.load(path,map_location='cpu',weights_only=True)
            assert payload['contract']['seed']==candidate['seed'] and payload['contract']['arm']=='RAW'
            model=c.e.build_model(candidate['seed'],512,device='cpu');model.load_state_dict(payload['model_state_dict']);model.eval()
            outputs=[]
            with torch.no_grad():
                for first in range(0,len(rel),1024):
                    x=torch.tensor(rel[first:first+1024]);outputs.append(r.decode_joints(d.predict(model,x,'residual')).numpy())
            q=np.concatenate(outputs)
            candidate['architecture']='13-512-512-512-6 SiLU; unbounded normalized residual output'
            candidate['output_contract']='z=current_normalized+MLP(x); endpoint-exact float64 c106r_precision.decode_joints; no clamp'
            candidate['input_contract']='base-frame target-current position train-zscore; canonical quaternion of Rcurrent.T@Rtarget; normalized current joints'
        assert q.shape==(3600,6) and np.isfinite(q).all()
        metric=d.geometric_metrics(q,val,details=True)
        for i,value in enumerate(q):
            valid=bool((value>=bounds[:,0]).all() and (value<=bounds[:,1]).all());assert valid==metric['rows'][i]['valid']
            if valid:
                t=independent.forward_kinematics(value);tref=reference.reference_forward_kinematics(value)
                pe=float(np.linalg.norm(t[:3,3]-tref[:3,3]));re=float(np.linalg.norm(t[:3,:3]-tref[:3,:3]))
                maxp=max(maxp,pe);maxrot=max(maxrot,re)
                assert pe<=cfg['fk_crosscheck_m'] and re<=cfg['fk_crosscheck_rotation_frobenius']
                p=float(np.linalg.norm(t[:3,3]-val.position[i]));rr=t[:3,:3]@r.quaternion_rotation(val.quaternion[i]).T
                v=np.array([rr[2,1]-rr[1,2],rr[0,2]-rr[2,0],rr[1,0]-rr[0,1]])/2
                angle=math.degrees(math.atan2(float(np.linalg.norm(v)),float((np.trace(rr)-1)/2)))
            else:p=angle=math.inf
            assert metric['rows'][i]['profile_a']==(p<=.002 and angle<=1)
            assert metric['rows'][i]['profile_b']==(p<=.001 and angle<=.5)
            checked+=1
        raw=RAW/(candidate['id']+'.json');d.write_json(raw,metric)
        candidate.update(parameters=sum(x.numel() for x in model.parameters()),storage='LOCAL_ONLY',remote_archive='NOT_CONFIRMED',
            direct_ik='NO_GO',hybrid_advantage='NOT_MEASURED',production='NOT_APPROVED',collision='NOT_CHECKED',
            validation=r.summary(metric,val),raw=artifact(raw))
        records.append(dict(id=candidate['id'],weights_only_load='PASS',validation_rows=3600,
            profile_a=metric['profile_a'],profile_b=metric['profile_b'],out_of_limits=metric['out_of_limits']))
        print(candidate['id'],records[-1],flush=True)
    assert checked==21600
    d.guard_hashes(inputs);d.guard_hashes(frozen)
    manifest=dict(schema='c107-research-handoff-preparation-v1',status='CANDIDATES_PINNED_NOT_G1_ACCEPTED',
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=inputs,
        robot=dict(robot_id=robot.robot_id,base=robot.base,tip=robot.tip,tcp=robot.tcp,joint_names=robot.joint_names,
            limits_rad=robot.limits,hashes=robot.hashes,position_unit='m',joint_unit='rad',quaternion='canonical wxyz'),
        candidates=candidates,excluded=dict(CENTERED='Continuation gate failed; validation and derivatives worse',LOCAL_Z='Worse same-root and new-root generalization'),
        T_C06='NOT_RUN_IN_FRESH_ENVIRONMENT',G1='OPEN',Hybrid='NOT_STARTED')
    d.write_json(BASE/'handoff-manifest.json',manifest)
    d.write_json(BASE/'C1-06R-closure.json',dict(status='CLOSED_WITH_UNMET_PRODUCT_TARGET',scope='Research stop, not all original plan deliverables passed',
        authority='User stage2 request; ADR-025',C1_06_T_C05='PASS_UNCHANGED',C1_06_H2='REJECTED_UNCHANGED',
        direct_ik='NO_GO',validation_product_target='NOT_MET',new_final_product_measurement='NOT_RUN',
        H2_R='NOT_EVALUATED_ON_NEW_INDEPENDENT_FINAL',R5='NOT_CREATED_NOT_RUN_AFTER_VALIDATION_STOP',
        current_mlp_micro_experiments='STOP',new_training='NOT_RUN',old_final_raw='NOT_READ',
        next_task='C1-07 final model card, clean-environment T-C06 and G1 decision',G1='OPEN'))
    d.write_json(BASE/'preparation-audit.json',dict(status='PASS',source_sha256=d.sha(Path(__file__)),frozen_files=len(frozen),
        prior_delivery_entries=len(lines),historical_registrations='3_TO_9_PASS',candidates=records,
        validation_predictions_independently_checked=checked,max_fk_position_difference_m=maxp,max_fk_rotation_difference_frobenius=maxrot,
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),inference_device='cpu',threads=1),
        wall_s=time.perf_counter()-start,clean_environment='NOT_RUN',new_training='NOT_RUN',old_final_raw='NOT_READ',
        independent_final='NOT_CREATED',hybrid_refinement='NOT_RUN',T_C06='PENDING',G1='OPEN'))
    print('PASS: preparation complete; G1 remains OPEN',flush=True)


if __name__=='__main__':main()
