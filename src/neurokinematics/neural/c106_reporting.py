"""Full-denominator C1-06 reports and independent raw-evidence acceptance audit."""
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np

from neurokinematics.neural import c106
from neurokinematics.neural.c106_runtime import (ROOT,BASE,STAGE2,STAGE1_COMMIT,read,write,now,config,check_file,
    file_record,fingerprint,require_runtime,campaign_paths,model_key)
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK


def quantiles(values):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all(): raise ValueError('nonfinite reported distribution')
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a,95)) if len(a) else None,p99=float(np.percentile(a,99)) if len(a) else None)


def breakdown(records, elapsed, *, repetitions=5):
    """Records are unique neural queries or historical attempts; elapsed aligned by column."""
    n=len(records)
    if np.asarray(elapsed).shape != (repetitions,n): raise ValueError('time denominator mismatch')
    if not n: return dict(n=0)
    def count(field): return sum(bool(r[field]) for r in records)
    counts=dict(profile_a=count('profile_a'),profile_b=count('profile_b'),
                nonfinite=sum(not r['finite'] and not r.get('no_candidate',False) and r.get('shape_valid',True) for r in records),
                joint_limit=sum(r['finite'] and not r['in_limits'] for r in records),
                invalid_shape=sum(not r.get('shape_valid',True) for r in records),
                no_candidate=sum(r.get('no_candidate',False) for r in records),
                unresolved=sum(not r['profile_a'] for r in records))
    successful=np.array([r['profile_a'] for r in records],dtype=bool)
    geometry=[r for r in records if r['position_error_m'] is not None]
    result=dict(n=n,counts=counts,rates={k:v/n for k,v in counts.items()},
        failure_classes=dict(Counter(r['failure_class'] for r in records)),
        geometry_coverage=len(geometry)/n,geometry_missing=n-len(geometry),
        position_m_valid_only=quantiles([r['position_error_m'] for r in geometry]),
        orientation_deg_valid_only=quantiles([r['orientation_error_deg'] for r in geometry]),
        position_m_success_only=quantiles([r['position_error_m'] for r in records if r['profile_a']]),
        orientation_deg_success_only=quantiles([r['orientation_error_deg'] for r in records if r['profile_a']]),
        all_time_ms=quantiles(np.asarray(elapsed).ravel()/1e6),
        successful_time_ms=quantiles(np.asarray(elapsed)[:,successful].ravel()/1e6),
        label_present='NOT_APPLICABLE',teacher_status='NOT_APPLICABLE',collision='NOT_CHECKED')
    for deadline in (10,50):
        misses=np.asarray(elapsed)>deadline*1000000
        result[f'timeout_{deadline}ms']=dict(measurements=int(misses.sum()),rate=float(misses.mean()),
            profile_a_deadline_rate=float((~misses & successful[None,:]).mean()))
    return result


def neural_raw(item, queries, diagnostics, *, validate=True):
    cp=item['checkpoint']; check_file(item['raw']); path=ROOT/item['raw']['path']
    if item['raw']['rows']!=60000: raise ValueError('model raw denominator')
    first=[]; elapsed=np.empty((5,12000),dtype=np.int64)
    validator=c106.Validator(); reference=PinocchioFK(validator.robot); max_fk_delta=0.
    count=0; cfg=config()
    for i,line in enumerate(path.open(encoding='utf-8')):
        row=json.loads(line); p,j=divmod(i,12000)
        if p>=5: raise ValueError('extra raw rows')
        query=queries[j]
        if (row['query_id'],row['group_id'],row['pass_index'],row['model'],row['checkpoint_sha256'],row['seed']) != (
                query['query_id'],query['query_group_id'],p,item['model'],cp['sha256'],cp['seed']): raise ValueError('neural row identity')
        if (row['subset'],row['mode'],row['query_list_sha256'],row['robot_hashes'],row['collision']) != (
                query['subset'],query['start_class'],cfg['query_sha256'],validator.robot.hashes,'NOT_CHECKED'):
            raise ValueError('neural context identity')
        if type(row['elapsed_ns']) is not int or row['elapsed_ns']<0: raise ValueError('invalid time')
        for deadline in (10,50):
            miss=row['elapsed_ns']>deadline*1000000
            if row[f'timeout_{deadline}ms']!=miss: raise ValueError('timeout corruption')
            for profile in ('a','b'):
                if row[f'profile_{profile}_deadline_{deadline}ms']!=(row['profile_'+profile] and not miss):
                    raise ValueError('deadline success corruption')
        elapsed[p,j]=row['elapsed_ns']
        if p==0:
            if validate:
                q=np.asarray([np.nan if x is None else x for x in row['q_raw_rad']])
                actual=validator.check(q,query)
                if any(actual[k]!=row[k] for k in actual): raise ValueError('independent raw metric corruption')
                if actual['in_limits']:
                    delta=float(np.max(np.abs(validator.fk.forward_kinematics(q)-reference.reference_forward_kinematics(q))))
                    max_fk_delta=max(max_fk_delta,delta)
                    if delta>1e-9: raise ValueError('reference/independent FK disagreement')
            first.append(row)
        else:
            for field in ('q_raw_rad','finite','in_limits','shape_valid','profile_a','profile_b','position_error_m','orientation_error_deg','failure_class'):
                if row[field]!=first[j][field]: raise ValueError('repeated geometry differs')
        count+=1
    if count!=60000: raise ValueError('missing neural row')
    subsets={'overall':np.arange(12000)}
    for s in ('main','boundary','singularity'):
        subsets[s]=np.array([i for i,q in enumerate(queries) if q['subset']==s])
        for m in ('local','wide'):
            subsets[f'{s}/{m}']=np.array([i for i,q in enumerate(queries) if q['subset']==s and q['start_class']==m])
    for m in ('local','wide'): subsets[m]=np.array([i for i,q in enumerate(queries) if q['start_class']==m])
    for key in ('near_limit','singular','position_shift'):
        subsets['exploratory/'+key]=np.array([i for i,d in enumerate(diagnostics) if d[key]],dtype=int)
    summaries={name:breakdown([first[i] for i in idx],elapsed[:,idx]) for name,idx in subsets.items()}
    valid=[r for r in first if r['position_error_m'] is not None]
    def examples(field):
        return [{k:r[k] for k in ('query_id','group_id','subset','mode','failure_class','position_error_m','orientation_error_deg','q_raw_rad')}
                for r in sorted(valid,key=lambda r:(-r[field],r['query_id']))[:20]]
    return dict(model=item['model'],seed=cp['seed'],experiment=cp['experiment'],variant=cp['variant'],
                raw=item['raw'],breakdowns=summaries,worst_position=examples('position_error_m'),
                worst_orientation=examples('orientation_error_deg'),max_reference_fk_delta=max_fk_delta,
                model_load_s=item['model_load_s'],warmup_s=item['warmup_s'],evaluation_wall_s=item['evaluation_wall_s']),first


def fractional_bootstrap(differences,queries,*,repeats=10000,rng_seed=2026100806):
    """Historical repeat means, root-paired with fixed-seed neural results.

    differences: query x fixed training seed x baseline/deadline comparison.
    Identity gate proves unique roots for this frozen inventory. No repeat is N.
    """
    differences=np.asarray(differences,dtype=float)
    if differences.ndim!=3 or len(differences)!=len(queries) or not np.isfinite(differences).all() or np.any(np.abs(differences)>1):
        raise ValueError('fractional differences shape/range')
    if len({q['query_group_id'] for q in queries})!=len(queries) or len({q['query_id'] for q in queries})!=len(queries):
        raise ValueError('fractional bootstrap requires proven unique root/query inventory')
    idxs=[np.array([i for i,q in enumerate(queries) if q['subset']==s]) for s in c106.SUBSETS]
    if any(not len(idx) for idx in idxs): raise ValueError('missing subset')
    rng=np.random.Generator(np.random.PCG64(rng_seed))
    point=np.stack([differences[idx].mean(axis=0) for idx in idxs])
    point=np.concatenate((point,.5*(point[1:2]+point[2:3])),axis=0)
    samples=np.empty((repeats,4,*differences.shape[1:]))
    for b in range(repeats):
        for j,idx in enumerate(idxs): samples[b,j]=differences[rng.choice(idx,len(idx),replace=True)].mean(axis=0)
        samples[b,3]=.5*(samples[b,1]+samples[b,2])
    def pack(p,values):
        return {s:dict(difference=float(p[j]),ci95=np.percentile(values[:,j],[2.5,97.5]).tolist())
                for j,s in enumerate((*c106.SUBSETS,'hard_equal_weight'))}
    return [dict(mean_over_fixed_seeds=pack(point[:,:,c].mean(axis=1),samples[:,:,:,c].mean(axis=2)),
                 per_seed={str(seed):pack(point[:,j,c],samples[:,:,j,c]) for j,seed in enumerate(c106.SEEDS)},
                 unique_queries=len(queries),root_groups=len(queries),bootstrap_replicates=repeats,bootstrap_seed=rng_seed,
                 label='EXPLORATORY; historical baseline repeat mean; conditional on fixed training seeds')
            for c in range(differences.shape[2])]


def summarize(run):
    require_runtime(); out,raw_root=campaign_paths(run)
    evaluation=read(out/'evaluation.json'); identity=read(out/'identity.json')
    if evaluation['status']!='COMPLETE_UNVERIFIED' or evaluation['rows']!=1260000 or len(evaluation['models'])!=21:
        raise ValueError('incomplete campaign')
    queries=[json.loads(line) for line in (ROOT/config()['query_file']).open(encoding='utf-8')]
    diagnostics=read(raw_root/'query-diagnostics.json'); scores={}; all_stats=[]
    for item in evaluation['models']:
        stats,rows=neural_raw(item,queries,diagnostics)
        all_stats.append(stats); cp=item['checkpoint']
        scores[(cp['experiment']+'/'+cp['variant'],cp['seed'])]=[dict(query_id=r['query_id'],group_id=r['group_id'],
            subsets=[r['subset']],seed=r['seed'],success=r['profile_a']) for r in rows]
        print(json.dumps(dict(stage='summary-audit',model=item['model'],rows=60000)),flush=True)
    def observations(arm): return [r for seed in c106.SEEDS for r in scores[(arm,seed)]]
    primary=c106.paired_bootstrap(observations('E-C05/FK_TANH'),observations('E-C03/Q'))
    decision=c106.h2_decision(primary)
    secondary=[]
    for pair in config()['secondary']:
        result=c106.paired_bootstrap(observations(pair['candidate']),observations(pair['control']))
        secondary.append(dict(**pair,result=result,label='EXPLORATORY; nominal CI, no multiple-comparison correction'))
    baseline_stats=[]; historical=[]; comparisons=[]
    for baseline in identity['baselines']:
        check_file(baseline['derived']); source=[json.loads(line) for line in (ROOT/baseline['derived']['path']).open(encoding='utf-8')]
        for deadline in (10,50):
            rr=[r for r in source if r['deadline_ms']==deadline]
            if len(rr)!=60000: raise ValueError('baseline count for deadline')
            success=np.array([r['profile_a'] for r in rr],dtype=float).reshape(5,12000)
            historical.append(success.mean(axis=0)); comparisons.append(dict(baseline=baseline['method'],deadline_ms=deadline))
            summary={}
            for name in ('overall','main','boundary','singularity','local','wide','main/local','main/wide','boundary/local','boundary/wide','singularity/local','singularity/wide'):
                chosen=[r for r in rr if name=='overall' or name==r['subset'] or name==r['mode'] or name==r['subset']+'/'+r['mode']]
                entry=breakdown(chosen,np.array([[r['elapsed_ns'] for r in chosen]]),repetitions=1)
                entry.update(unique_queries=len({r['query_id'] for r in chosen}),measurement_rows=len(chosen),
                    historical_timeout_count=sum(r['timeout'] for r in chosen),
                    historical_timeout_rate=sum(r['timeout'] for r in chosen)/len(chosen),
                    profile_a_deadline_rate=sum(r['profile_a_deadline'] for r in chosen)/len(chosen),
                    profile_b_deadline_rate=sum(r['profile_b_deadline'] for r in chosen)/len(chosen))
                summary[name]=entry
            baseline_stats.append(dict(method=baseline['method'],deadline_ms=deadline,breakdowns=summary,
                                       timing_scope='HISTORICAL_UBUNTU_ROS; not speed-comparable to Windows neural'))
    neural=np.array([[float(r['success']) for r in scores[('E-C05/FK_TANH',s)]] for s in c106.SEEDS]).T
    diffs=neural[:,:,None]-np.asarray(historical).T[:,None,:]
    historical_ci=fractional_bootstrap(diffs,queries)
    write(out/'results.json',dict(status='ANALYZED',primary_h2=primary,H2_decision=decision,secondary=secondary,
        neural=all_stats,baseline=baseline_stats,baseline_paired=[dict(**p,result=result) for p,result in zip(comparisons,historical_ci)],
        collision='NOT_CHECKED',neural_unique_queries=12000,neural_model_query_pairs=252000,neural_measurements=1260000,
        baseline_measurements=600000,remote_archive='NOT_CONFIRMED',speed_superiority='NOT_CLAIMED',time_utc=now()))
    print(json.dumps(dict(status='ANALYZED',H2=decision,primary=primary['mean_over_fixed_seeds'])),flush=True)


def audit(run):
    require_runtime(); out,raw_root=campaign_paths(run)
    identity=read(out/'identity.json'); evaluation=read(out/'evaluation.json'); results=read(out/'results.json')
    if identity['status']!='PASS' or evaluation['rows']!=1260000 or len(results['neural'])!=21:
        raise ValueError('acceptance inventory')
    expected={model_key(cp) for cp in config()['checkpoints']}
    if {r['model'] for r in evaluation['models']}!=expected or {r['model'] for r in results['neural']}!=expected:
        raise ValueError('model inventory drift')
    records=[identity['query'],identity['diagnostics']]
    for b in identity['baselines']: records.extend([b['source'],b['derived']])
    records.extend(m['raw'] for m in evaluation['models'])
    for item in records: check_file(item)
    for n in results['neural']:
        if n['breakdowns']['overall']['n']!=12000: raise ValueError('summary denominator')
        for s,count in config()['counts'].items():
            if n['breakdowns'][s]['n']!=count: raise ValueError('subgroup denominator')
    if results['H2_decision']!=c106.h2_decision(results['primary_h2']): raise ValueError('H2 decision drift')
    commands=[]
    for path in (BASE/'commands').glob('*/command.json'):
        value=read(path)
        for k in ('stdout','stderr'):
            if fingerprint(path.parent/(k+'.log'))['sha256']!=value[k+'_sha256']: raise ValueError('command evidence drift')
        commands.append(dict(path=path.relative_to(ROOT).as_posix(),exit_code=value['exit_code']))
    manifest=[file_record(path,lines=path.suffix=='.jsonl') for path in sorted(raw_root.iterdir()) if path.is_file()]
    write(out/'raw-manifest.json',dict(files=manifest,bytes=sum(x['bytes'] for x in manifest),storage='LOCAL_ONLY',remote_archive='NOT_CONFIRMED'))
    write(out/'acceptance.json',dict(task='C1-06',status='COMPLETE_RESEARCH_EVALUATION',T_C05='PASS',
        H2=results['H2_decision'],primary=results['primary_h2']['mean_over_fixed_seeds'],
        direct_ik_decision='NO_GO' if max(r['breakdowns']['overall']['rates']['profile_a'] for r in results['neural'] if r['variant']=='FK_TANH')<.95 else 'REQUIRES_SEPARATE_REVIEW',
        queries=12000,models=21,training_seeds=list(c106.SEEDS),neural_rows=1260000,baseline_rows=600000,
        stage1_commit=STAGE1_COMMIT,
        stage2_preflight_sha256=fingerprint(STAGE2/'preflight.json')['sha256'],
        results_sha256=fingerprint(out/'results.json')['sha256'],raw_manifest_sha256=fingerprint(out/'raw-manifest.json')['sha256'],
        command_logs=commands,time_utc=now(),G1='NOT_RUN',C1_07='NOT_STARTED',collision='NOT_CHECKED',remote_archive='NOT_CONFIRMED'))
    write(out/'C1-07-handoff.json',dict(task='C1-07',source_task='C1-06',status='INPUTS_READY_RESEARCH_ONLY',model_family='FK_TANH',
        checkpoints=[cp for cp in config()['checkpoints'] if cp['variant']=='FK_TANH'],
        acceptance=file_record(out/'acceptance.json'),results=file_record(out/'results.json'),raw_manifest=file_record(out/'raw-manifest.json'),
        query=identity['query'],normalization=file_record(ROOT/'experiments/C1-02/normalization.json'),
        source_config=file_record(ROOT/'experiments/C1-05/config.json'),evaluation_config=file_record(BASE/'config.json'),
        H2=results['H2_decision'],primary=results['primary_h2'],collision='NOT_CHECKED',operational_use='NOT_APPROVED',
        storage='LOCAL_ONLY weights/raw; remote archive NOT_CONFIRMED',reproduction='experiments/C1-06/stage2/COMMANDS.md',
        limitations=['three fixed training seeds','composite FK loss/output head contrast','different realized training budgets',
                    'historical Linux baseline latency not comparable','empirical bootstrap does not prove zero population success']))
    print(json.dumps(dict(status='PASS',T_C05='PASS',H2=results['H2_decision'],raw_files=len(manifest))),flush=True)
