"""Audit scaling experiment and independently inspect the validation criterion."""
from pathlib import Path
import math
import platform
import xml.etree.ElementTree as ET
import numpy as np
import torch
from neurokinematics.neural import c106r_feature_scale as z


def independent_angle(a,b):
    relative=a@b.T
    vector=np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])/2
    return math.degrees(math.atan2(float(np.linalg.norm(vector)),float((np.trace(relative)-1)/2)))


def breakdown(items,multipliers):
    n=len(items);valid=np.array([x['valid'] for x in items])
    p=np.array([x['position_m'] if x['valid'] else math.inf for x in items])
    a=np.array([x['orientation_deg'] if x['valid'] else math.inf for x in items])
    pa=p<=.002;ra=a<=1
    severity=np.sort(np.maximum(p/.002,a))
    quantile=lambda fraction:float(severity[math.ceil(n*fraction)-1]) if math.isfinite(severity[math.ceil(n*fraction)-1]) else None
    return dict(n=n,valid=int(valid.sum()),invalid=int((~valid).sum()),position_pass=int(pa.sum()),rotation_pass=int(ra.sum()),
        profile_a=int((pa&ra).sum()),valid_both_fail=int((valid&~pa&~ra).sum()),
        valid_position_only_fail=int((valid&~pa&ra).sum()),valid_rotation_only_fail=int((valid&pa&~ra).sum()),
        severity_a=dict(median=quantile(.5),p95=quantile(.95)),
        sensitivity=[dict(multiplier=k,position_m=.002*k,orientation_deg=k,passed=int(((p<=.002*k)&(a<=k)).sum())) for k in multipliers])


def main():
    d=z.d;base=z.BASE;out=base/'audit.json'
    if out.exists():raise FileExistsError(out)
    cfg=d.read_json(z.CONFIG);result=d.read_json(base/'results.json')
    assert result['inputs']==d.read_json(base/'registration.json')['inputs'];d.guard_hashes(result['inputs'])
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    train,val=d.load_data(label_fk=True);roots=d.matched_rows(train,512)[0]['local']
    directions,probe=z.load_source('directions'),z.load_source('probe')
    z.f.check_separation(roots,directions,probe,val)
    cells=result['cells'];assert len(cells)==4 and result['total_updates']==20000
    assert len({x['initial_hash'] for x in cells})==1
    kin=z.f.s.IndependentFK(z.r.load_robot());bounds=np.asarray(z.r.load_robot().limits)
    count=0;maxp=maxangle=0.;reviews=[];scales=[];coverage=[];baselines=[]
    for n in cfg['sizes']:
        datasets=dict(train=directions.take(np.arange(n*8)),original=roots.take(np.arange(n)),
                      same_root_probe=probe.take(np.arange(n*8)),validation=val)
        scaler=d.read_json(base/f'n{n}-scaler.json')
        assert scaler==z.fit_local(datasets['train'],z.r.relative_pose(datasets['train']))
        scales.append(dict(n_roots=n,count=scaler['count'],mean=scaler['mean'],std=scaler['std']))
        train_pose=z.r.relative_pose(datasets['train']);lo=train_pose.min(0);hi=train_pose.max(0)
        for name in ('same_root_probe','validation'):
            rows=datasets[name];pose=z.r.relative_pose(rows)
            masks={'all':np.ones(len(rows.pair_id),dtype=bool)}
            if name=='validation':masks.update(local=rows.mode=='local',wide=rows.mode=='wide')
            for group,mask in masks.items():
                outside=((pose[mask]<lo)|(pose[mask]>hi)).any(1)
                norms=np.linalg.norm((pose[mask]-scaler['mean'])/scaler['std'],axis=1)
                coverage.append(dict(n_roots=n,set=name,group=group,n=int(mask.sum()),
                    outside_train_component_ranges=int(outside.sum()),z_norm_median=float(np.median(norms)),z_norm_p95=float(np.quantile(norms,.95))))
            current=d.geometric_metrics(rows.q_current,rows,details=True)
            baselines.append(dict(n_roots=n,set=name,summary=z.r.summary(current,rows),
                description='unchanged q_current; descriptive reference, no numerical correction'))
        pair=[x for x in cells if x['n_roots']==n];assert [x['arm'] for x in pair]==cfg['arms']
        for cell in pair:
            assert cell['steps']==5000 and cell['batch_size']==8*n and cell['exposures']==8*n*5000
            assert cell['reload']=='EXACT_ALL_SETS'
            if cell['arm']=='RAW':assert cell['control_replay']=='EXACT_TENSORS_AND_ALL_METRICS'
            for key in ('checkpoint','raw','scaler'):assert d.sha(d.ROOT/cell[key]['path'])==cell[key]['sha256']
            payload=torch.load(d.ROOT/cell['checkpoint']['path'],weights_only=True)
            assert type(payload['contract']['torch']) is str and payload['contract']['inputs']==result['inputs']
            assert payload['contract']['train_pair_ids']==datasets['train'].pair_id.tolist()
            raw=d.read_json(d.ROOT/cell['raw']['path']);groups={}
            for name,rows in datasets.items():
                metric=raw[name];items=metric['rows'];count+=len(items)
                assert [x['pair_id'] for x in items]==rows.pair_id.tolist()
                assert len(items)==metric['n']==len(rows.pair_id)
                assert z.r.summary(metric,rows)==cell['metrics'][name]
                for i,x in enumerate(items):
                    q=np.asarray(x['q_rad']);valid=x['q_rad'] is not None and np.isfinite(q).all() and np.all(q>=bounds[:,0]) and np.all(q<=bounds[:,1])
                    assert bool(valid)==x['valid']
                    if valid:
                        t=kin.forward_kinematics(q);p=float(np.linalg.norm(t[:3,3]-rows.position[i]))
                        a=independent_angle(t[:3,:3],z.r.quaternion_rotation(rows.quaternion[i]))
                        maxp=max(maxp,abs(p-x['position_m']));maxangle=max(maxangle,abs(a-x['orientation_deg']))
                        assert abs(p-x['position_m'])<1e-9 and abs(a-x['orientation_deg'])<1e-7
                    else:p=a=math.inf
                    assert x['profile_a']==(p<=.002 and a<=1)
                    assert x['profile_b']==(p<=.001 and a<=.5)
                for profile in ('profile_a','profile_b'):assert sum(x[profile] for x in items)==metric[profile]
                for key in ('position_m','orientation_deg'):
                    ordered=sorted(x[key] if x['valid'] else math.inf for x in items)
                    for label,fraction in (('median',.5),('p95',.95),('p99',.99),('max',1)):
                        value=ordered[math.ceil(fraction*len(items))-1]
                        assert metric[key][label]==(value if math.isfinite(value) else None)
                groups[name]=breakdown(items,cfg['sensitivity_multipliers'])
                if name=='validation':
                    masks=dict(local=rows.mode=='local',wide=rows.mode=='wide',main=rows.family=='main',
                        boundary=rows.family=='boundary',singularity=rows.family=='singularity',missing_teacher=~rows.label_present)
                    for group,mask in masks.items():groups['validation_'+group]=breakdown([x for x,use in zip(items,mask) if use],cfg['sensitivity_multipliers'])
            reviews.append(dict(name=cell['name'],groups=groups))
    # Feasibility witness, never a model input: every wide target shares a
    # validation root with a labeled local target, including solver failures.
    local_map={str(val.source_sample_id[i]):i for i in np.flatnonzero(val.mode=='local')}
    witness=np.array([val.q_target[local_map[str(root)]] for root in val.source_sample_id])
    for i,q in enumerate(witness):
        t=kin.forward_kinematics(q)
        assert np.linalg.norm(t[:3,3]-val.position[i])<=1e-9
        assert np.linalg.norm(t[:3,:3]-z.r.quaternion_rotation(val.quaternion[i]))<=1e-9
    oracle=d.geometric_metrics(witness,val,details=True)
    assert oracle['profile_a']==oracle['profile_b']==3600
    d.write_json(z.RAW/'validation-root-witness.json',oracle)
    suites=ET.parse(base/'tests.xml').getroot().findall('testsuite')
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
    assert sum(int(s.attrib['tests']) for s in suites)==64
    evidence=dict(status='PASS',source_sha256=d.sha(Path(__file__)),tests=64,prediction_rows=count,
        max_independent_position_difference_m=maxp,max_independent_angle_difference_deg=maxangle,
        frozen_files=len(frozen),scalers=scales,cells=reviews,
        posthoc_feature_coverage=coverage,posthoc_unchanged_current_baselines=baselines,
        oracle=dict(n=3600,roots=len(local_map),missing_wide_labels=int((~val.label_present).sum()),profile_a=3600,profile_b=3600,
            raw=z.f.artifact(z.RAW/'validation-root-witness.json'),use='feasibility_only_not_learned_model'),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=str(torch.version.cuda),gpu=torch.cuda.get_device_name(0)),
        old_final_raw='NOT_READ',final_test='NOT_CREATED')
    d.write_json(out,evidence)
    print('PASS:64 tests,33984 independent FK/atan2 checks,3600/3600 A/B root witnesses,122 frozen inputs')


if __name__=='__main__':main()
