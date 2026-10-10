"""Replay zero/small-target predictions and independently audit response matrices."""
from pathlib import Path
import platform
import xml.etree.ElementTree as ET
import numpy as np
import torch
from neurokinematics.neural import c106r_local_response as a


def main():
    d=a.d;out=a.BASE/'audit.json'
    if out.exists():raise FileExistsError(out)
    d.configure();result=d.read_json(a.BASE/'results.json');cfg=d.read_json(a.CONFIG)
    assert result['inputs']==d.read_json(a.BASE/'registration.json')['inputs'];d.guard_hashes(result['inputs'])
    frozen=d.read_json(d.ROOT/'experiments/C1-06R/training-freeze.json')['files'];d.guard_hashes(frozen)
    assert result['optimizer_steps']==0 and len(result['cells'])==4
    train,val=d.load_data(label_fk=True);roots=d.matched_rows(train,512)[0]['local']
    local_val=val.take(np.flatnonzero(val.mode=='local'))
    parent=d.read_json(a.z.BASE/'results.json');norm=d.read_json(d.ROOT/cfg['normalization'])
    robot=a.r.load_robot();jac=a.s.IndependentJacobian(robot);bounds=np.asarray(robot.limits)
    preflight=d.read_json(a.BASE/'preflight.json')
    zero_count=small_count=anchor_count=0;extras=[]
    for cell in result['cells']:
        name=cell['name'];old=next(x for x in parent['cells'] if x['name']==name)
        model=a.e.build_model(2026100901,512)
        model.load_state_dict(torch.load(d.ROOT/old['checkpoint']['path'],weights_only=True)['model_state_dict']);model.eval()
        before=d.state_hash(model);scaler=d.read_json(d.ROOT/old['scaler']['path'])
        assert cell['weights_unchanged'] and cell['source_checkpoint']==old['checkpoint']
        assert d.sha(d.ROOT/cell['zero_raw']['path'])==cell['zero_raw']['sha256']
        raw=d.read_json(d.ROOT/cell['zero_raw']['path']);all_zeros={}
        for pop,base in (('train',roots.take(np.arange(old['n_roots']))),('validation',local_val)):
            for kind in cfg['anchor_types']:
                key=pop+'-'+kind;rows=a.zero_queries(base,kind);all_zeros[key]=rows
                assert raw[key]['anchors']==rows.q_current.tolist()
                q=a.predict(model,a.feature_values(rows,old['arm'],norm,scaler))
                metric=d.geometric_metrics(q,rows,details=True);assert metric==raw[key]['metric']
                summary=cell['zero'][key]
                for k,v in a.r.summary(metric,rows).items():assert summary[k]==v
                assert summary['q_bias_l2_rad']==a.stats(np.linalg.norm(q-rows.q_current,axis=1))
                zero_count+=len(q)
        for key,summary in cell['derivatives'].items():
            assert d.sha(d.ROOT/summary['raw']['path'])==summary['raw']['sha256']
            with np.load(d.ROOT/summary['raw']['path'],allow_pickle=False) as f:
                matrices={k:f[k].copy() for k in f.files}
            m=matrices;n=summary['n'];anchor_count+=n
            # Zero query suffixes contain source indexes; reconstruct selected
            # rows from source pair identity before the diagnostic suffix.
            wanted=[p.split(':zero-')[0] for p in preflight['selection'][key]]
            source=roots if key.startswith('train') else local_val
            lookup={str(p):i for i,p in enumerate(source.pair_id)}
            base=source.take(np.array([lookup[p] for p in wanted]))
            zero=a.zero_queries(base,key.split('-')[1]);assert np.array_equal(m['anchor_q'],zero.q_current)
            q,b=a.input_derivative(model,a.feature_values(zero,old['arm'],norm,scaler))
            k=np.einsum('noi,nij->noj',b[:,:,:7],m['tangent'])
            assert np.array_equal(q,m['q_zero']) and np.array_equal(k,m['k_float32'])
            valid=np.isfinite(q).all(1)&(q>=bounds[:,0]).all(1)&(q<=bounds[:,1]).all(1)
            assert np.array_equal(valid,m['valid_zero']) and int(valid.sum())==summary['valid_prediction_jacobians']
            actual=[];cosines=[];gains=[]
            for i in range(n):
                j=jac.jacobian(zero.q_current[i]);assert np.array_equal(j,m['j_anchor'][i])
                assert a.response_error(j,np.eye(6))==0 and np.isclose(a.response_error(j,np.zeros((6,6))),1)
                if valid[i]:
                    jp=jac.jacobian(q[i]);assert np.array_equal(jp,m['j_prediction'][i])
                    actual.append(a.response_error(j,k[i],jp))
                    target=a.s.normalize_jacobian(j,.9015);output=a.s.normalize_jacobian(jp,.9015)@k[i]
                    den=np.linalg.norm(target,axis=0)*np.linalg.norm(output,axis=0)
                    cosines.extend(np.divide((target*output).sum(0),den,out=np.full(6,np.nan),where=den>0).tolist())
                    gains.extend((np.linalg.norm(output,axis=0)/np.linalg.norm(target,axis=0)).tolist())
            assert a.stats(actual)==summary['actual_task_relative_error_valid_only']
            assert a.stats(np.linalg.norm(k-np.eye(6),axis=(1,2))/np.sqrt(6))==summary['joint_k_minus_identity']
            assert np.allclose(m['fd_shadow64'],m['k_shadow64'],atol=cfg['shadow_gradient_atol'],rtol=cfg['shadow_gradient_rtol'])
            for fd_report in summary['fd']:
                h=fd_report['h_rad'];rows=a.perturb_queries(zero,h)
                predictions=a.predict(model,a.feature_values(rows,old['arm'],norm,scaler))
                assert np.array_equal(predictions,m['pred_'+str(h)]) and np.array_equal(rows.q_target,m['target_'+str(h)])
                assert np.array_equal(a.fd_matrix(predictions,n,h),m['fd_'+str(h)])
                metric=d.geometric_metrics(predictions,rows,details=True)
                assert a.r.summary(metric,rows)==fd_report['small_target']
                assert d.geometric_metrics(rows.q_current,rows)==fd_report['unchanged_current']
                oracle=d.geometric_metrics(rows.q_target,rows)
                assert oracle['profile_a']==oracle['profile_b']==len(rows.pair_id)
                small_count+=len(rows.pair_id)
            extras.append(dict(model=name,group=key,valid_anchor_count=int(valid.sum()),
                task_direction_cosine_valid_only=a.stats(cosines),negative_cosine_directions=sum(x<0 for x in cosines),
                task_column_norm_gain_valid_only=a.stats(gains)))
        assert d.state_hash(model)==before
    assert zero_count==16704 and small_count==18432 and anchor_count==512
    suites=ET.parse(a.BASE/'tests.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites)==68
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
    d.guard_hashes(result['inputs']);d.guard_hashes(frozen)
    d.write_json(out,dict(status='PASS',tests=68,frozen_files=len(frozen),zero_query_replays=zero_count,
        small_query_replays=small_count,derivative_anchor_model_pairs=anchor_count,
        shadow64_fd='PASS_C103_ATOL_RTOL',zero_and_small_oracles='ALL_A_B_PASS',
        task_direction_posthoc=extras,optimizer_steps=0,source_sha256=d.sha(Path(__file__)),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=str(torch.version.cuda),gpu=torch.cuda.get_device_name(0)),
        old_final_raw='NOT_READ',final_test='NOT_CREATED'))
    print('PASS:68 tests,16704 zero queries,18432 small queries,512 derivative anchors,122 frozen files; no training')


if __name__=='__main__':main()
