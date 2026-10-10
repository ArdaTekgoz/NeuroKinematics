"""Independent FK audit, exact replay, and preregistered continuation decision."""
from pathlib import Path
from copy import deepcopy
import math
import platform
import xml.etree.ElementTree as ET
import numpy as np
import torch
from neurokinematics.neural import c106r_centered as c


def angle(a,b):
    r=a@b.T
    vector=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    return math.degrees(math.atan2(float(np.linalg.norm(vector)),float((np.trace(r)-1)/2)))


def severity(items):
    values=np.sort([max(x['position_m']/.002,x['orientation_deg']) if x['valid'] else math.inf for x in items])
    return np.array([values[math.ceil(len(values)*q)-1] for q in (.5,.95)])


def finite_json(values):return [float(x) if math.isfinite(x) else None for x in values]


def bootstrap(delta,cfg):
    rng=np.random.Generator(np.random.PCG64(cfg['bootstrap_seed']));means=[]
    for start in range(0,cfg['bootstrap_replicates'],250):
        size=min(250,cfg['bootstrap_replicates']-start)
        means.extend(delta[rng.integers(len(delta),size=(size,len(delta)))].mean(1).tolist())
    return np.quantile(means,[.025,.975])


def main():
    d=c.d;base=c.BASE;out=base/'audit.json'
    if out.exists():raise FileExistsError(out)
    d.configure();cfg=d.read_json(c.CONFIG);result=d.read_json(base/'results.json')
    assert result['inputs']==d.read_json(base/'registration.json')['inputs'];d.guard_hashes(result['inputs'])
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    data=c.datasets();norm=d.read_json(d.ROOT/cfg['normalization']);groups,_=c.anchor_data(data)
    values={key:c.r.features(rows,'relative',norm) for key,rows in data.items()}
    robot=c.r.load_robot();kin=c.s.IndependentFK(robot);jac=c.s.IndependentJacobian(robot);bounds=np.asarray(robot.limits)
    cells=result['cells'];assert len(cells)==6 and result['total_updates']==30000 and result['total_exposures']==122880000
    raw={};count=0;derivatives=0;maxp=maxangle=0.;zero_ok=True
    for cell in cells:
        assert cell['steps']==5000 and cell['batch_size']==4096 and cell['exposures']==20480000
        assert cell['parameters']==535558 and cell['reload']=='EXACT_ALL_SIX_SETS'
        for key in ('checkpoint','raw'):assert d.sha(d.ROOT/cell[key]['path'])==cell[key]['sha256']
        payload=torch.load(d.ROOT/cell['checkpoint']['path'],weights_only=True)
        assert payload['contract']['inputs']==result['inputs'] and payload['contract']['train_pair_ids']==data['train'].pair_id.tolist()
        assert payload['contract']['arm']==cell['arm'] and payload['contract']['seed']==cell['seed']
        model=c.e.build_model(cell['seed'],512);model.load_state_dict(payload['model_state_dict'])
        if cell['arm']=='CENTERED':model=c.Centered(model,norm)
        sets=d.read_json(d.ROOT/cell['raw']['path']);raw[cell['name']]=sets
        for key,rows in data.items():
            metric=sets[key];items=metric['rows'];count+=len(items)
            assert c.r.evaluate(model,'residual',rows,values[key])==metric
            assert c.r.summary(metric,rows)==cell['metrics'][key]
            assert [x['pair_id'] for x in items]==rows.pair_id.tolist()
            for i,item in enumerate(items):
                q=np.asarray(item['q_rad']);valid=item['q_rad'] is not None and np.isfinite(q).all() and (q>=bounds[:,0]).all() and (q<=bounds[:,1]).all()
                assert bool(valid)==item['valid']
                if valid:
                    t=kin.forward_kinematics(q);p=float(np.linalg.norm(t[:3,3]-rows.position[i]))
                    a=angle(t[:3,:3],c.r.quaternion_rotation(rows.quaternion[i]))
                    maxp=max(maxp,abs(p-item['position_m']));maxangle=max(maxangle,abs(a-item['orientation_deg']))
                    # atan2 vs acos loses relative precision near zero, far below either profile.
                    assert abs(p-item['position_m'])<1e-9 and abs(a-item['orientation_deg'])<5e-6
                else:p=a=math.inf
                assert item['profile_a']==(p<=.002 and a<=1)
                assert item['profile_b']==(p<=.001 and a<=.5)
            for profile in ('profile_a','profile_b'):assert sum(x[profile] for x in items)==metric[profile]
            for component in ('position_m','orientation_deg'):
                ordered=sorted(x[component] if x['valid'] else math.inf for x in items)
                for name,fraction in (('median',.5),('p95',.95),('p99',.99),('max',1)):
                    v=ordered[math.ceil(fraction*len(items))-1]
                    assert metric[component][name]==(v if math.isfinite(v) else None)
            if cell['arm']=='CENTERED' and key.startswith('zero_'):
                zero_ok &= metric['profile_a']==metric['profile_b']==len(rows.pair_id)
        for key,group in groups.items():
            entry=cell['derivatives'][key];path=d.ROOT/entry['raw']['path'];assert d.sha(path)==entry['raw']['sha256']
            with np.load(path,allow_pickle=False) as m:
                assert np.array_equal(m['anchor_q'],group.q_current)
                q,b=c.a.input_derivative(model,c.a.feature_values(group,'RAW',norm,None))
                assert np.array_equal(q,m['q_zero']) and np.array_equal(b[:,:,:7]@m['tangent'],m['k'])
                valid=np.isfinite(q).all(1)&(q>=bounds[:,0]).all(1)&(q<=bounds[:,1]).all(1)
                assert np.array_equal(valid,m['valid']) and int(valid.sum())==entry['valid']
                shadow=deepcopy(model).double()
                _,b64=c.a.input_derivative(shadow,c.a.feature_values(group,'RAW',norm,None,np.float64))
                shifted=c.a.perturb_queries(group,1e-5)
                fd=c.a.fd_matrix(c.a.predict(shadow,c.a.feature_values(shifted,'RAW',norm,None,np.float64)),32,1e-5)
                assert np.array_equal(fd,m['fd64']) and np.array_equal(b64[:,:,:7]@m['tangent'],m['k64'])
                assert np.allclose(m['fd64'],m['k64'],atol=1e-5,rtol=.001)
                errors=[]
                for i in range(32):
                    assert np.array_equal(jac.jacobian(group.q_current[i]),m['j_anchor'][i])
                    if m['valid'][i]:
                        assert np.array_equal(jac.jacobian(q[i]),m['j_prediction'][i])
                        errors.append(c.a.response_error(m['j_anchor'][i],m['k'][i],m['j_prediction'][i]))
                assert c.a.stats(errors)==entry['actual_task_relative_error_valid_only']
            derivatives+=32
    assert count==87696 and derivatives==384
    paired=[];deltas=[];each_improves=True;severity_ok=True
    local=np.flatnonzero(data['validation'].mode=='local')
    for seed in cfg['seeds']:
        pair=[next(x for x in cells if x['seed']==seed and x['arm']==arm) for arm in cfg['arms']]
        assert pair[0]['initial_hash']==pair[1]['initial_hash']
        local_items=[[raw[x['name']]['validation']['rows'][i] for i in local] for x in pair]
        success=[np.array([row['profile_a'] for row in rows],dtype=float) for rows in local_items]
        delta=success[1]-success[0];deltas.append(delta)
        probe=[x['metrics']['same_root_probe']['profile_a'] for x in pair]
        sev=[severity(x) for x in local_items]
        each_improves &= delta.sum()>0 and probe[1]>probe[0]
        severity_ok &= bool(np.all(sev[1]<=sev[0]))
        paired.append(dict(seed=seed,local_a=[int(x.sum()) for x in success],probe_a=probe,
            local_a_gain_pp=float(delta.mean()*100),local_severity_median_p95=[finite_json(x) for x in sev],
            validation_a=[x['metrics']['validation']['profile_a'] for x in pair],
            new_successes=int((delta>0).sum()),lost_successes=int((delta<0).sum())))
    assert len({x['initial_hash'] for x in cells})==3
    assert cells[0]['control_replay']=='EXACT_TENSORS_AND_FOUR_METRIC_SETS'
    delta=np.mean(deltas,axis=0);ci=bootstrap(delta,cfg)
    assert np.array_equal(bootstrap(np.zeros(1800),cfg),[0,0])
    assert np.array_equal(bootstrap(np.ones(1800),cfg),[1,1])
    decision=bool(each_improves and severity_ok and zero_ok and ci[0]>0)
    junit=ET.parse(base/'tests.xml').getroot();suites=list(junit.iter('testsuite'))
    tests={key:sum(int(x.attrib.get(key,0)) for x in suites) for key in ('tests','failures','errors','skipped')}
    assert tests==dict(tests=72,failures=0,errors=0,skipped=0)
    d.guard_hashes(result['inputs']);d.guard_hashes(frozen)
    d.write_json(out,dict(status='PASS',source_sha256=d.sha(Path(__file__)),tests=tests,frozen_files=len(frozen),
        predictions_replayed_and_independently_checked=count,derivative_anchors_replayed=derivatives,
        max_position_difference_m=maxp,max_angle_difference_deg=maxangle,paired=paired,
        continuation=dict(pass_gate=decision,each_seed_local_and_probe_increase=bool(each_improves),
            each_seed_severity_not_worse=bool(severity_ok),centered_zero_all_a_b=bool(zero_ok),
            local_mean_gain_pp=float(delta.mean()*100),paired_root_bootstrap95_gain_pp=(ci*100).tolist(),
            caveat='Conditional on these three training seeds; validation repeatedly used for research, not independent final'),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=str(torch.version.cuda),
            gpu=torch.cuda.get_device_name(),platform=platform.platform()),old_final_raw='NOT_READ',final_test='NOT_CREATED'))
    print('PASS independent audit:',count,'predictions;',derivatives,'derivative anchors; continuation:',decision,flush=True)


if __name__=='__main__':main()
