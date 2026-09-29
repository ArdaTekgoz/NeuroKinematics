"""C1-03 reproducible numerical evidence. This module is NOT a Torch forward.

Pinocchio is the independent numeric oracle. Tensor-to-NumPy conversions here
only inspect completed outputs/derivatives and never feed the Torch graph.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio
import torch

from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics.torch_fk import TorchFK
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian

STAGE1 = '6a56e9e9eeaabb561cb3efdc899a5e5709876783'
STAGE1_SHA = '0ee8682e7848225a4e173519760d7cfbbf04aca41172a2938fbcd0b0ab373af2'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def clean(value):
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return str(value)
    if isinstance(value, np.generic): return value.item()
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def load_contract(root=ROOT):
    root=Path(root); folder=root/'experiments/C1-03'
    manifest=(folder/'SHA256SUMS').read_bytes().replace(b'\r\n',b'\n')
    if sha(manifest)!=STAGE1_SHA: raise ValueError('Stage1 manifest hash mismatch')
    # Tracking documents evolve in Stage2; their approved contents remain in Git.
    historical={'.gitattributes','docs/tasks/C1-03.md','docs/records/STATUS.md','docs/TRACEABILITY.md','docs/roadmaps/C1_Core.md'}
    for line in manifest.decode().splitlines():
        expected,path=line.split('  ',1)
        data=(subprocess.check_output(['git','show',STAGE1+':'+path],cwd=root)
              if path in historical else (root/path).read_bytes())
        if sha(data.replace(b'\r\n',b'\n'))!=expected: raise ValueError('Stage1 hash mismatch: '+path)
    # Original reference/source files must still match; additions are allowed.
    inputs=json.loads((folder/'input-hashes.json').read_text(encoding='utf-8'))
    for entry in inputs['files']:
        if sha((root/entry['path']).read_bytes().replace(b'\r\n',b'\n'))!=entry['canonical_lf_sha256']:
            raise ValueError('immutable input changed: '+entry['path'])
    load_robot(root)  # exact raw robot/TCP bytes, without normalization
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    if len(rows)!=1086 or len({r['id'] for r in rows})!=1086: raise ValueError('sample inventory')
    for row in rows:
        for key,dtype in [('q64','<f8'),('q32','<f4')]:
            if sha(np.array(row[key],dtype=dtype).tobytes())!=row[key+'_sha256']: raise ValueError('q hash')
    return config,rows

def source_hashes(root=ROOT):
    paths=['src/neurokinematics/kinematics/torch_fk.py','src/neurokinematics/core/torch_validation.py']
    return {p:sha((root/p).read_bytes().replace(b'\r\n',b'\n')) for p in paths}

def runtime(root=ROOT):
    expected={'numpy':'2.5.3','pin':'4.1.0','torch':'2.10.0+cpu','pytest':'8.4.2','setuptools':'82.0.1'}
    for file in ['experiments/C1-03/dependency-pins.json','experiments/C1-03/stage2/runtime-pins.json']:
        for entry in json.loads((root/file).read_text()): expected[entry['name']]=entry['version']
    actual={p:metadata.version(p) for p in expected}
    if actual!=expected or platform.python_version()!='3.12.14': raise ValueError('runtime version drift')
    check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    if check.returncode: raise ValueError(check.stdout+check.stderr)
    if Path(sys.prefix) not in Path(np.__file__).parents: raise ValueError('NumPy must use isolated wheel overlay')
    torch.set_num_threads(1)
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'packages':actual,'numpy_file':np.__file__,'torch_file':torch.__file__,
            'pinocchio_file':pinocchio.__file__,'threads':torch.get_num_threads(),
            'source_hashes':source_hashes(root),'pip_check':check.stdout.strip(),
            'config_sha256':sha((root/'experiments/C1-03/config.json').read_bytes()),
            'samples_sha256':sha((root/'experiments/C1-03/samples.jsonl').read_bytes()),
            'cuda':'NOT_RUN','protocol':'r1-math/r2-runtime-harness'}

def outputs_torch(t, config):
    p=t[:3,3]; r=t[:3,:3]; g=config['gradient']
    lp=(p*p.new_tensor([.7,-.4,.2])).sum()
    lr=(r*r.new_tensor(g['W'])).sum()
    loss=(((p-p.new_tensor(g['pd_m']))/.9015)**2).sum()+((r-r.new_tensor(g['Rd']))**2).sum()/8
    return torch.cat((p,r.reshape(-1),torch.stack((lp,lr,loss))))

def outputs_numpy(t, config):
    p=t[:3,3]; r=t[:3,:3]; g=config['gradient']
    lp=np.dot([.7,-.4,.2],p)
    lr=np.sum(np.array(g['W'])*r)
    loss=np.sum(((p-g['pd_m'])/.9015)**2)+np.sum((r-g['Rd'])**2)/8
    return np.concatenate((p,r.ravel(),[lp,lr,loss]))

class Oracle:
    def __init__(self, inputs):
        self.backend=PinocchioFK(inputs)
        self.bounds=np.array(inputs.limits)

    def forward(self,q):
        if type(self.backend) is not PinocchioFK:
            raise ValueError('independent Pinocchio oracle required')
        return self.backend.reference_forward_kinematics(q)

def finite_difference(oracle,q,config,edge=False):
    h=config['gradient']['epsilon_rad']; columns=[]; stencils=[]
    for j in range(len(q)):
        delta=np.zeros_like(q);delta[j]=h
        if q[j]-h<oracle.bounds[j,0]:
            if not edge: raise ValueError('central difference outside limits')
            f0=outputs_numpy(oracle.forward(q),config)
            derivative=(-3*f0+4*outputs_numpy(oracle.forward(q+delta),config)-outputs_numpy(oracle.forward(q+2*delta),config))/(2*h)
            stencil='forward2'
        elif q[j]+h>oracle.bounds[j,1]:
            if not edge: raise ValueError('central difference outside limits')
            f0=outputs_numpy(oracle.forward(q),config)
            derivative=(3*f0-4*outputs_numpy(oracle.forward(q-delta),config)+outputs_numpy(oracle.forward(q-2*delta),config))/(2*h)
            stencil='backward2'
        else:
            derivative=(outputs_numpy(oracle.forward(q+delta),config)-outputs_numpy(oracle.forward(q-delta),config))/(2*h)
            stencil='central'
        columns.append(derivative);stencils.append(stencil)
    return np.array(columns).T,stencils

def derivative_metrics(actual,reference,config):
    diff=np.abs(actual-reference); g=config['gradient']
    threshold=g['atol']+g['rtol']*np.abs(reference)
    return {'max_abs':float(diff.max()),'max_relative':float((diff/np.maximum(abs(reference),1e-12)).max()),
            'max_tolerance_ratio':float((diff/threshold).max()),
            'passed':bool(np.isfinite(actual).all() and np.isfinite(reference).all() and np.all(diff<=threshold))}

def pose_metrics(actual,reference,dtype,config):
    tol=config['fk'][dtype]; structural=config['fk']['homogeneous_atol'][dtype]
    p=float(np.linalg.norm(actual[:3,3]-reference[:3,3])); r=float(np.linalg.norm(actual[:3,:3]-reference[:3,:3]))
    orth=float(np.linalg.norm(actual[:3,:3].T@actual[:3,:3]-np.eye(3)))
    det=float(abs(np.linalg.det(actual[:3,:3])-1)); bottom=float(np.linalg.norm(actual[3]-[0,0,0,1]))
    return {'position_l2_m':p,'rotation_frobenius':r,'orthogonality':orth,'det_error':det,'bottom_error':bottom,
            'passed':bool(np.isfinite(actual).all() and p<=tol['position_l2_m'] and r<=tol['rotation_frobenius'] and max(orth,det,bottom)<=structural)}

class Evidence:
    def __init__(self,path,env):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.records=[];self.env=env
        write_json(self.path/'environment.json',env)

    def add(self,kind,ident,**values):
        record=clean(dict(kind=kind,id=ident,**values))
        self.records.append(record)

    def finish(self,extra):
        records=self.records
        failures=[r for r in records if not r['passed']]
        for name,rows in [('results.jsonl',records),('failures.jsonl',failures)]:
            (self.path/name).write_text(''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
        suite=ET.Element('testsuite',name='C1-03',tests=str(len(records)),failures=str(len(failures)),errors='0',skipped='0')
        for row in records:
            case=ET.SubElement(suite,'testcase',classname=row['kind'],name=row['id'])
            if not row['passed']: ET.SubElement(case,'failure',message='frozen acceptance failed').text=json.dumps(row)
        ET.ElementTree(suite).write(self.path/'junit.xml',encoding='utf-8',xml_declaration=True)
        summary={'status':'PASS' if not failures else 'FAIL','counts':dict(Counter(r['kind'] for r in records)),
                 'failures':len(failures),'source_hashes':self.env['source_hashes'],**extra}
        for dtype in ['float64','float32']:
            group=[r for r in records if r['kind']=='fk-'+dtype and 'position_l2_m' in r]
            if group: summary[dtype]={k:max(r[k] for r in group) for k in ['position_l2_m','rotation_frobenius']}
        group=[r for r in records if r['kind']=='gradient' and 'max_abs' in r]
        if group: summary['gradient']={k:max(r[k] for r in group) for k in ['max_abs','max_relative','max_tolerance_ratio']}
        write_json(self.path/'summary.json',summary)
        checks={p.name:sha(p.read_bytes()) for p in sorted(self.path.iterdir()) if p.is_file()}
        write_json(self.path/'hashes.json',checks)
        return summary

def verify_evidence(path,config,rows,smoke=False):
    path=Path(path);summary=json.loads((path/'summary.json').read_text())
    hashes=json.loads((path/'hashes.json').read_text())
    for name,expected in hashes.items():
        if sha((path/name).read_bytes())!=expected: raise ValueError('raw evidence hash: '+name)
    records=[json.loads(s) for s in (path/'results.jsonl').read_text().splitlines()]
    keys=[(r['kind'],r['id']) for r in records]
    if len(set(keys))!=len(keys): raise ValueError('duplicate evidence id')
    selected=[r for r in rows if not smoke or r['id'] in config['smoke']['sample_ids']]
    for dtype in ['float64','float32']:
        group=[r for r in records if r['kind']=='fk-'+dtype]
        if {r['id'] for r in group}!={r['id'] for r in selected}: raise ValueError('FK inventory')
        for r in group:
            if r['dtype']!=dtype or r['observed_dtype']!=dtype: raise ValueError('dtype label')
            sample=next(x for x in selected if x['id']==r['id'])
            key='q64' if dtype=='float64' else 'q32'
            if r['q']!=sample[key] or r['q_sha256']!=sample[key+'_sha256']: raise ValueError('q identity')
            expected=pose_metrics(np.array(r['torch']),np.array(r['pinocchio']),dtype,config)
            if any(r[k]!=v for k,v in expected.items()): raise ValueError('FK metric mismatch')
    gradients=[r for r in records if r['kind']=='gradient']
    expected_grad=[r['id'] for r in selected if r['group']=='grad']
    if {r['id'] for r in gradients}!=set(expected_grad): raise ValueError('gradient inventory')
    for r in gradients+[r for r in records if r['kind']=='edge-gradient']:
        sample=next(x for x in rows if x['id']==r['id'])
        if r['q']!=sample['q64'] or r['epsilon']!=config['gradient']['epsilon_rad']: raise ValueError('gradient input/epsilon drift')
        if len(r['stencils'])!=6 or (r['kind']=='gradient' and set(r['stencils'])!={'central'}): raise ValueError('gradient stencil drift')
        actual=derivative_metrics(np.array(r['autograd']),np.array(r['finite_difference']),config)
        if any(r[k]!=v for k,v in actual.items()): raise ValueError('gradient metric mismatch')
    if not smoke:
        expected={'batch':10,'gradcheck':32,'jacobian':32,'sensitivity':3,'batch-gradient':2,'edge-gradient':30,'singularity':4}
        counts=Counter(r['kind'] for r in records)
        if any(counts[k]!=v for k,v in expected.items()): raise ValueError('auxiliary inventory')
    failures=[r for r in records if not r['passed']]
    saved=[json.loads(s) for s in (path/'failures.jsonl').read_text().splitlines()]
    if saved!=failures or summary['failures']!=len(failures): raise ValueError('failure inventory')
    if dict(Counter(r['kind'] for r in records))!=summary['counts']: raise ValueError('count mismatch')
    if failures or summary['status']!='PASS': raise ValueError('acceptance failed')
    return summary

def run(output,smoke=False,smoke_witness=None):
    config,rows=load_contract();env=runtime()
    if not smoke:
        if smoke_witness is None: raise ValueError('full run requires smoke witness')
        witness=verify_evidence(smoke_witness,config,rows,smoke=True)
        if witness['source_hashes']!=env['source_hashes']: raise ValueError('source changed after smoke')
    evidence=Evidence(output,env);robot=load_robot();fk=TorchFK.from_frozen();oracle=Oracle(robot)
    call=lambda q:fk(q,robot_id=robot.robot_id,joint_names=robot.joint_names)
    selected=[r for r in rows if not smoke or r['id'] in config['smoke']['sample_ids']]
    def guarded(kind,row,fn):
        try: evidence.add(kind,row['id'],**fn())
        except Exception:
            evidence.add(kind,row['id'],passed=False,exception=traceback.format_exc(),q=row.get('q64'))
    for dtype,key in [('float64','q64'),('float32','q32')]:
        all_values=[]
        for row in selected:
            def evaluate():
                q=torch.tensor(row[key],dtype=getattr(torch,dtype))
                out=call(q);actual=out.double().numpy();reference=oracle.forward(np.array(row[key]))
                all_values.append(actual)
                if out.dtype!=q.dtype or out.device!=q.device: raise ValueError('output dtype/device mismatch')
                return dict(dtype=dtype,observed_dtype=str(out.dtype).split('.')[-1],q=row[key],q_sha256=row[key+'_sha256'],torch=actual,pinocchio=reference,
                            **pose_metrics(actual,reference,dtype,config))
            guarded('fk-'+dtype,row,evaluate)
        if not smoke:
            q=torch.tensor([r[key] for r in rows],dtype=getattr(torch,dtype)); expected=np.array(all_values)
            for size in config['batch_sizes']:
                actual=torch.cat([call(chunk) for chunk in q.split(size)]).double().numpy()
                reversed_out=call(q.flip(0)).flip(0).double().numpy()
                errors=np.linalg.norm(actual[:,:3,:3]-expected[:,:3,:3],axis=(1,2))
                pos=np.linalg.norm(actual[:,:3,3]-expected[:,:3,3],axis=1)
                reverse=np.max(abs(reversed_out-expected))
                tol=config['fk'][dtype]
                evidence.add('batch',dtype+'-'+str(size),dtype=dtype,batch_size=size,sample_ids=[r['id'] for r in rows],
                             position_errors=pos,rotation_errors=errors,reverse_max_abs=reverse,
                             passed=bool(np.isfinite(actual).all() and np.all(pos<=tol['position_l2_m']) and np.all(errors<=tol['rotation_frobenius']) and reverse<=tol['rotation_frobenius']))
    gradient_rows=[r for r in selected if r['group']=='grad']
    if not smoke: gradient_rows += [r for r in rows if r['group'] in ('edge','hand','singularity_candidate')]
    for row in gradient_rows:
        kind='gradient' if row['group']=='grad' else 'edge-gradient'
        def check_gradient():
            q=torch.tensor(row['q64'],dtype=torch.float64,requires_grad=True)
            func=lambda x:outputs_torch(call(x),config)
            actual=torch.autograd.functional.jacobian(func,q).numpy()
            reference,stencils=finite_difference(oracle,q.detach().numpy(),config,edge=(kind=='edge-gradient'))
            t=call(q)
            gp=torch.autograd.grad((t[:3,3]*t.new_tensor([.7,-.4,.2])).sum(),q,retain_graph=True)[0]
            gr=torch.autograd.grad((t[:3,:3]*t.new_tensor(config['gradient']['W'])).sum(),q)[0]
            if not torch.isfinite(gp).all() or not torch.isfinite(gr).all(): raise ValueError('nonfinite backward')
            return dict(q=row['q64'],epsilon=1e-6,stencils=stencils,autograd=actual,finite_difference=reference,
                        position_backward=gp.numpy(),rotation_backward=gr.numpy(),**derivative_metrics(actual,reference,config))
        guarded(kind,row,check_gradient)
        if not smoke and row['group']=='grad':
            q=torch.tensor(row['q64'],dtype=torch.float64,requires_grad=True)
            guarded('gradcheck',row,lambda:dict(passed=torch.autograd.gradcheck(lambda x:outputs_torch(call(x),config)[:12],(q,),eps=1e-6,atol=1e-5,rtol=1e-3,fast_mode=False)))
            def jacobian_check():
                t=call(q);d=torch.autograd.functional.jacobian(call,q).numpy();r=t[:3,:3].detach().numpy()
                angular=[]
                for j in range(6):
                    a=d[:3,:3,j]@r.T;s=(a-a.T)/2;angular.append([s[2,1],s[0,2],s[1,0]])
                jac=np.vstack((d[:3,3,:],np.array(angular).T));refs=[IndependentJacobian(robot).jacobian(row['q64']),PinocchioJacobian(robot).jacobian(row['q64'])]
                metrics=[derivative_metrics(jac,ref,config) for ref in refs]
                return dict(q=row['q64'],torch_jacobian=jac,geometric=refs[0],pinocchio=refs[1],metrics=metrics,passed=all(m['passed'] for m in metrics))
            guarded('jacobian',row,jacobian_check)
    if not smoke:
        for ident in config['gradient']['sensitivity']['sample_ids']:
            row=next(r for r in rows if r['id']==ident)
            def sensitivity():
                q=torch.tensor(row['q64'],dtype=torch.float64,requires_grad=True)
                loss=outputs_torch(call(q),config)[-1];g=torch.autograd.grad(loss,q)[0];norm=torch.linalg.vector_norm(g)
                direction=-g/norm;step=1e-6*direction
                delta=(outputs_torch(call(q+step),config)[-1]-loss).detach().numpy()
                predicted=(g*step).sum().detach().numpy()
                qa=q.detach().numpy();qs=step.detach().numpy()
                independent=outputs_numpy(oracle.forward(qa+qs),config)[-1]-outputs_numpy(oracle.forward(qa),config)[-1]
                threshold=1e-8+1e-3*abs(predicted)
                return dict(q=row['q64'],gradient=g.numpy(),direction=direction.numpy(),predicted=predicted,delta_torch=delta,delta_pinocchio=independent,
                            passed=bool(norm>1e-8 and delta<0 and independent<0 and abs(delta-predicted)<=threshold and abs(independent-predicted)<=threshold))
            guarded('sensitivity',row,sensitivity)
        subset=[r for r in rows if r['group']=='grad'][:7]
        for dtype in [torch.float64,torch.float32]:
            q=torch.tensor([r['q64'] for r in subset],dtype=dtype,requires_grad=True)
            d=torch.autograd.functional.jacobian(lambda x:call(x)[:,:3,:],q)
            maximum=0.;cross=0.
            for i in range(7):
                single=torch.autograd.functional.jacobian(lambda x:call(x)[:3,:],q[i])
                maximum=max(maximum,float((d[i,:,:,i,:]-single).abs().max()))
                for j in range(7):
                    if i!=j: cross=max(cross,float(d[i,:,:,j,:].abs().max()))
            evidence.add('batch-gradient',str(dtype),diagonal_max_abs=maximum,cross_max_abs=cross,passed=bool(torch.isfinite(d).all() and cross==0 and maximum<=1e-5))
        for row in [r for r in rows if r['group']=='singularity_candidate' or r['id']=='hand-zero']:
            j=IndependentJacobian(robot).jacobian(row['q64']);j[:3]/=.9015
            singular=np.linalg.svd(j,compute_uv=False)
            evidence.add('singularity',row['id'],q=row['q64'],singular_values=singular,passed=bool(np.isfinite(singular).all()))
    summary=evidence.finish({'mode':'smoke' if smoke else 'full','gradient_configuration_count':len([r for r in selected if r['group']=='grad'])})
    verify_evidence(output,config,rows,smoke=smoke)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-witness',type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:
        c,r=load_contract();result=verify_evidence(a.output,c,r,smoke=a.smoke)
    else: result=run(a.output,a.smoke,a.smoke_witness)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
