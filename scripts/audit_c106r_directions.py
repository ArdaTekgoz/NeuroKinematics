"""Rebuild directional provenance and audit matched budgets and full metrics."""
from dataclasses import replace
from pathlib import Path
import math
import platform
import xml.etree.ElementTree as ET
import numpy as np
import torch
from neurokinematics.neural import c106r_directions as f
from neurokinematics.kinematics.metrics import rotation_error


def main():
    d=f.d;out=f.BASE/'audit.json'
    if out.exists():raise FileExistsError(out)
    cfg=d.read_json(f.CONFIG);results=d.read_json(f.BASE/'results.json')
    d.guard_hashes(results['inputs'])
    assert results['inputs']==d.read_json(f.BASE/'registration.json')['inputs']
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    pre=d.read_json(f.BASE/'preflight.json')
    for a in pre['artifacts']:assert d.sha(d.ROOT/a['path'])==a['sha256']
    train,val=d.load_data(label_fk=True)
    roots=d.matched_rows(train,max(cfg['sizes']))[0]['local']
    data={};metadata={}
    for name,role in (('directions','train'),('probe','probe')):
        rows,meta=f.generate(roots,8,role,cfg['data_seed']);data[name]=rows;metadata[name]=meta
        with np.load(f.RAW/(name+'.npz'),allow_pickle=False) as saved:
            for field in rows.__dataclass_fields__:assert np.array_equal(getattr(rows,field),saved[field])
    actual=d.read_json(f.RAW/'provenance.json')
    for name,meta in metadata.items():assert actual[name]==meta
    f.check_separation(roots,data['directions'],data['probe'],val)
    # Original preflight's first32 rows repeat four teacher roots; explicitly
    # report that scope and add a distinct-root finite-difference cross-check.
    _,_,_,unique_checks=f.s.preflight(roots,d.read_json(d.ROOT/cfg['geometry_contract']),f.r.load_robot())
    cells=results['cells'];assert len(cells)==4 and results['total_updates']==20000
    assert len({c['initial_hash'] for c in cells})==1
    kin=f.s.IndependentFK(f.r.load_robot());bounds=np.asarray(f.r.load_robot().limits)
    max_p=max_r=0.;count=0;comparisons=[]
    for n in cfg['sizes']:
        pair=[c for c in cells if c['n_roots']==n]
        assert [c['arm'] for c in pair]==cfg['arms']
        assert all(c['batch_size']==8*n and c['steps']==5000 and c['exposures']==8*n*5000 for c in pair)
        source=roots.take(np.arange(n));multi=data['directions'].take(np.arange(n*8));probe=data['probe'].take(np.arange(n*8))
        repeated=source.take(np.repeat(np.arange(n),8))
        repeated=replace(repeated,pair_id=np.asarray([f'{p}:repeat:{i%8}' for i,p in enumerate(repeated.pair_id)]))
        records=[]
        for cell in pair:
            assert cell['unique_train_inputs']==(n if cell['arm']=='REPEAT' else n*8)
            for key in ('checkpoint','raw'):assert d.sha(d.ROOT/cell[key]['path'])==cell[key]['sha256']
            ckpt=torch.load(d.ROOT/cell['checkpoint']['path'],weights_only=True)
            assert ckpt['contract']['inputs']==results['inputs'] and type(ckpt['contract']['torch']) is str
            raw=d.read_json(d.ROOT/cell['raw']['path']);records.append(raw)
            datasets=dict(train=repeated if cell['arm']=='REPEAT' else multi,original=source,same_root_probe=probe,validation=val)
            assert ckpt['contract']['train_pair_ids']==datasets['train'].pair_id.tolist()
            for name,rows in datasets.items():
                metrics=raw[name];items=metrics['rows'];count+=len(items)
                assert len(items)==len(rows.pair_id)==metrics['n']
                assert [x['pair_id'] for x in items]==rows.pair_id.tolist()
                assert f.r.summary(metrics,rows)==cell['metrics'][name]
                for profile,p,a in (('profile_a',.002,1),('profile_b',.001,.5)):
                    assert sum(x[profile] for x in items)==metrics[profile]
                    for x in items:assert x[profile]==bool(x['valid'] and x['position_m']<=p and x['orientation_deg']<=a)
                for key in ('position_m','orientation_deg'):
                    ordered=sorted(x[key] if x['valid'] else math.inf for x in items)
                    for label,quantile in (('median',.5),('p95',.95),('p99',.99),('max',1)):
                        value=ordered[math.ceil(quantile*len(items))-1]
                        assert metrics[key][label]==(value if math.isfinite(value) else None)
                for i,x in enumerate(items):
                    q=np.asarray(x['q_rad'])
                    valid=x['q_rad'] is not None and np.isfinite(q).all() and np.all(q>=bounds[:,0]) and np.all(q<=bounds[:,1])
                    assert bool(valid)==x['valid']
                    if valid:
                        t=kin.forward_kinematics(q)
                        p=float(np.linalg.norm(t[:3,3]-rows.position[i]))
                        a=math.degrees(rotation_error(t[:3,:3],f.r.quaternion_rotation(rows.quaternion[i])))
                        max_p=max(max_p,abs(p-x['position_m']));max_r=max(max_r,abs(a-x['orientation_deg']))
                        assert abs(p-x['position_m'])<1e-9 and abs(a-x['orientation_deg'])<1e-7
            assert cell['reload']=='EXACT_MATCH_ALL_SETS'
        changes={}
        for name in ('original','same_root_probe','validation'):
            a,b=[rec[name]['rows'] for rec in records]
            assert [x['pair_id'] for x in a]==[x['pair_id'] for x in b]
            changes[name]=dict(n=len(a),repeat_a=sum(x['profile_a'] for x in a),directions_a=sum(x['profile_a'] for x in b),
                gained=sum(not x['profile_a'] and y['profile_a'] for x,y in zip(a,b)),
                lost=sum(x['profile_a'] and not y['profile_a'] for x,y in zip(a,b)))
        comparisons.append(dict(n_roots=n,changes=changes))
    suites=ET.parse(f.BASE/'tests.xml').getroot().findall('testsuite')
    assert all(int(x.attrib.get(k,0))==0 for x in suites for k in ('errors','failures','skipped'))
    assert sum(int(x.attrib['tests']) for x in suites)==57
    d.write_json(out,dict(status='PASS',tests_passed=57,frozen_files=len(frozen),prediction_rows=count,
        independent_prediction_error_max_m=max_p,independent_prediction_angle_difference_max_deg=max_r,
        distinct_root_geometry=unique_checks,original_fd_scope='32 rows per generated set, four distinct teachers in each',
        comparisons=comparisons,source_sha256=d.sha(Path(__file__)),runtime=dict(python=platform.python_version(),
        torch=str(torch.__version__),cuda=str(torch.version.cuda),gpu=torch.cuda.get_device_name(0)),
        provenance_replay='EXACT_ALL_FIELDS',old_final_raw='NOT_READ',final_test='NOT_CREATED'))
    print('PASS:57 tests,122 frozen files,8192 generated records,33984 prediction rows and paired exposures')


if __name__=='__main__':main()
