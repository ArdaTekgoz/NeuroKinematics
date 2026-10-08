"""Reconstruct C1-05 selection, pairing, inventory and comparisons from raw evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from neurokinematics.neural import c105
from neurokinematics.neural.c104 import load_data, read_json, write_json, sha


def require(condition,message):
    if not condition: raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def audit_run(path,train,val):
    summary=read_json(path/'summary.json'); start=read_json(path/'start.json')
    seed=summary['seed']; variants=summary['variants']; epoch_count=summary['epochs']
    require(summary['status']=='COMPLETE' and 1<=epoch_count<=200,'run completeness')
    require(summary['steps_per_model']==15*epoch_count,'step budget')
    require(summary['source_hashes']==c105.source_hashes(),'source drift')
    require(summary['config_sha256']==sha(c105.CONFIG),'config drift')
    require(len(set(start['initial_state_sha256'].values()))==1,'initialization not paired')
    idx=np.flatnonzero(train.label_present); rng=np.random.Generator(np.random.PCG64(seed)); orders=[]
    for epoch in range(epoch_count):
        order=idx[rng.permutation(len(idx))]
        orders.append(hashlib.sha256(('\n'.join(train.pair_id[order])+'\n').encode()).hexdigest())
    outputs={}; checkpoint_records=[]
    for variant in variants:
        logs=rows(path/(variant+'-epochs.jsonl'))
        require(len(logs)==epoch_count,'epoch log count')
        best=min(range(len(logs)),key=lambda i:logs[i]['validation']['components']['q'])
        require(best+1==summary['best_epoch'][variant],'not lowest q-loss earliest epoch')
        for i,r in enumerate(logs):
            require(r['epoch']==i+1 and r['optimizer_steps']==15*(i+1),'epoch/step drift')
            require(r['permutation_sha256']==orders[i],'unpaired or incorrect order')
            require(r['train']['labeled']==15204 and r['validation']['labeled']==3249,'masked inventory drift')
            require(len(r['combined_gradient_norms'])==15 and all(math.isfinite(x) for x in r['combined_gradient_norms']),'gradient records')
        for kind in ('best_checkpoints','last_checkpoints'):
            cp=summary[kind][variant]; file=Path(cp['path'])
            require(file.is_file() and file.stat().st_size==cp['bytes'] and sha(file)==cp['sha256'],'checkpoint bytes/SHA')
            model,metadata=c105.load_checkpoint(file)
            require(metadata['selected_epoch']==(best+1 if kind=='best_checkpoints' else epoch_count),'checkpoint epoch')
            checkpoint_records.append(dict(experiment=summary['experiment'],seed=seed,variant=variant,kind=kind,**cp))
        cp=summary['best_checkpoints'][variant]; model,metadata=c105.load_checkpoint(Path(cp['path']))
        raw,z=c105.infer(model,variant,val.conditioned)
        result_path=path/(variant+'-validation.jsonl'); result=rows(result_path)
        sidecar=read_json(result_path.with_suffix('.summary.json'))
        require(sha(result_path)==sidecar['raw_sha256'],'raw SHA')
        require([r['pair_id'] for r in result]==val.pair_id.tolist(),'full ordered validation inventory')
        require(sum(not r['label_present'] for r in result)==351,'missing wide rows dropped')
        lo,hi=np.asarray(c105.load_robot().limits).T
        for i,r in enumerate(result):
            require(r['label_present']==bool(val.label_present[i]),'label mask drift')
            require(r['q_raw_rad']==raw[i].tolist(),'checkpoint inference changed')
            valid=bool(np.isfinite(raw[i]).all() and np.all(raw[i]>=lo) and np.all(raw[i]<=hi))
            require(valid==r['in_limits'],'raw limit validity')
            require(r['q_evaluation_rad']==r['q_raw_rad'],'projected evaluation')
            require(r['q_fk_input_rad']==(r['q_raw_rad'] if valid else None),'FK raw policy')
            require(r['profile_a']==bool(valid and r['position_m']<=.002 and r['orientation_deg']<=1.),'Profile A threshold/payda')
            require(r['profile_b']==bool(valid and r['position_m']<=.001 and r['orientation_deg']<=.5),'Profile B threshold')
            require((r['q_loss'] is None)==(not r['label_present']),'q label loss mask')
        require(c105.summarize_rows(result)==sidecar['breakdowns'],'summary differs from raw rows')
        outputs[variant]=dict(overall=sidecar['breakdowns']['overall'],raw=result)
    return summary,outputs,checkpoint_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1); train,val=load_data(label_fk=True)
    runs=[]; checkpoints=[]; comparisons=[]; historical=[]; totals=Counter()
    paths=sorted(c105.STAGE2.glob('E-C*/seed-*/*/summary.json'))
    configs=set(); seeds=default_seeds=[2026100201,2026100202,2026100203]
    expected=['E-C03','E-C04']+([] if (c105.STAGE2/'E-C05-SKIP.json').exists() else ['E-C05'])
    for experiment in expected:
        require(sorted(read_json(p)['seed'] for p in paths if read_json(p)['experiment']==experiment)==seeds,'three-seed comparison missing')
    for file in paths:
        summary,outputs,cps=audit_run(file.parent,train,val); checkpoints.extend(cps)
        control,intervention=summary['variants']; configs.update(summary['variants'])
        co,iv=outputs[control]['overall'],outputs[intervention]['overall']
        compare=dict(experiment=summary['experiment'],seed=summary['seed'],control=control,intervention=intervention,
            control_profile_a=co['profile_a'],intervention_profile_a=iv['profile_a'],profile_a_absolute_delta=iv['profile_a']-co['profile_a'],
            profile_a_percentage_points=(iv['profile_a']-co['profile_a'])/36,
            out_of_limits_delta=iv['out_of_limits']-co['out_of_limits'])
        for metric in ('position_m','orientation_deg'):
            for kind in ('valid_only','full'):
                before=co[metric+'_'+kind]['median'];after=iv[metric+'_'+kind]['median']
                compare[metric+'_'+kind+'_median_delta']=after-before if after is not None and before is not None else None
        compare['paired_success_gains']=sum(not r['profile_a'] and s['profile_a'] for r,s in zip(outputs[control]['raw'],outputs[intervention]['raw']))
        compare['paired_success_losses']=sum(r['profile_a'] and not s['profile_a'] for r,s in zip(outputs[control]['raw'],outputs[intervention]['raw']))
        comparisons.append(compare)
        historical_path=c105.ROOT/'experiments/C1-04/stage2'/f"seed-{summary['seed']}-conditioned-validation.jsonl"
        hist=rows(historical_path); require([r['pair_id'] for r in hist]==val.pair_id.tolist(),'historical pairing')
        if summary['experiment']=='E-C03':
            old=np.array([r['q_raw_rad'] for r in hist]); new=np.array([r['q_raw_rad'] for r in outputs['Q']['raw']])
            historical.append(dict(seed=summary['seed'],control_max_q_delta_rad=float(np.max(np.abs(old-new))),
                old_profile_a=0,new_control_profile_a=co['profile_a'],new_FK_profile_a=iv['profile_a'],
                old_valid_only_position_median=float(np.median([r['position_error_m'] for r in hist if r['in_limits']])),
                new_FK_valid_only_position_median=iv['position_m_valid_only']['median'],
                old_valid_only_orientation_median=float(np.median([r['orientation_error_deg'] for r in hist if r['in_limits']])),
                new_FK_valid_only_orientation_median=iv['orientation_deg_valid_only']['median']))
        totals['model_seed_runs']+=2;totals['optimizer_steps']+=2*summary['steps_per_model'];totals['validation_rows']+=7200;totals['wall_s']+=summary['wall_s']
        runs.append({k:v for k,v in summary.items() if k not in ('best_checkpoints','last_checkpoints')})
    require(len(configs)<=4 and totals['model_seed_runs']<=18 and totals['optimizer_steps']<=54000,'registered search/budget')
    variability={}
    for experiment in expected:
        group=[r for r in comparisons if r['experiment']==experiment]
        variability[experiment]={key:dict(mean=float(np.mean([r[key] for r in group])),std_sample=float(np.std([r[key] for r in group],ddof=1)),
            minimum=min(r[key] for r in group),maximum=max(r[key] for r in group)) for key in ['profile_a_absolute_delta','position_m_valid_only_median_delta','orientation_deg_valid_only_median_delta','out_of_limits_delta']}
    write_json(a.output,dict(status='PASS_EVIDENCE_AUDIT',totals=dict(totals),configurations=sorted(configs),runs=runs,
        paired_comparisons=comparisons,seed_variability=variability,historical_conditioned_comparisons=historical,checkpoints=checkpoints,
        scope='Validation only; T-C04 final acceptance also requires clean witness and final source/test audit',test_and_benchmark='SEALED_NOT_RUN'))
    print(json.dumps(dict(status='PASS_EVIDENCE_AUDIT',totals=dict(totals),paired_comparisons=comparisons)))


if __name__=='__main__':main()
