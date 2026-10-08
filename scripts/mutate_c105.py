"""Real in-memory source mutants; unchanged numerical witnesses; no repo mutation."""
import argparse
import difflib
import json
from pathlib import Path
import types
import numpy as np
import torch
from neurokinematics.neural import training_fk, physics, c105
from neurokinematics.neural.c104 import write_json, sha


def kernel_witness(module):
    q=torch.tensor([4.,-3.,2.,-4.,3.,7.],dtype=torch.float64,requires_grad=True)
    original=training_fk.TrainingFK.from_frozen(domain=training_fk.DOMAIN)
    changed=module.TrainingFK.from_frozen(domain=module.DOMAIN)
    kwargs=dict(robot_id=original.robot_id,joint_names=original.joint_names)
    expected=original(q,**kwargs); actual=changed(q,**kwargs)
    assert actual.shape==expected.shape and torch.isfinite(actual).all()
    assert torch.allclose(actual,expected,atol=1e-9,rtol=0),'forward'
    ja=torch.autograd.functional.jacobian(lambda x:changed(x,**kwargs),q)
    je=torch.autograd.functional.jacobian(lambda x:original(x,**kwargs),q)
    assert torch.allclose(ja,je,atol=1e-5,rtol=1e-3),'gradient'


def physics_witness(module):
    z=torch.tensor([[1.2,-.2,.3,.4,.5,.6]],dtype=torch.float64,requires_grad=True)
    y=torch.full_like(z,.4); p=torch.tensor([[.1,.2,.3]],dtype=torch.float64); r=torch.eye(3,dtype=torch.float64)[None]
    expected,_,_=physics.PhysicsLoss().components(z,y,p,r)
    actual,_,_=module.PhysicsLoss().components(z,y,p,r)
    for key in expected:
        assert torch.allclose(expected[key],actual[key],atol=1e-12,rtol=0),key
        assert torch.allclose(torch.autograd.grad(expected[key].sum(),z,retain_graph=True)[0],torch.autograd.grad(actual[key].sum(),z,retain_graph=True)[0],atol=1e-5,rtol=1e-3),key+' gradient'
    quaternion=torch.tensor([.5,.5,.5,.5],dtype=torch.float64)
    assert torch.equal(module.quaternion_matrix(quaternion),module.quaternion_matrix(-quaternion))
    assert torch.equal(module.quaternion_matrix(quaternion),physics.quaternion_matrix(quaternion))
    arm=dict(lambda_p=0.,lambda_R=0.,lambda_lim=0.)
    assert torch.equal(module.combined(actual,arm),expected['q'].mean())
    assert module.normalized_head(z,'FK') is z


def training_witness(module):
    assert module.PAIRS==c105.PAIRS
    assert module.full_quantiles([1.,2.],4)==c105.full_quantiles([1.,2.],4)
    assert module.configuration('FK_TANH')['lambda_lim']==0


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    def add(module,name,before,after,witness): cases.append((module,name,before,after,witness))
    anchor='        n = values.shape[0]'
    for name,change in [('joint-order','values.flip(-1)'),('degrees','values * (torch.pi / 180)'),('clamp','values.clamp(-1,1)'),('wrap','torch.remainder(values, torch.pi)')]:
        add(training_fk,name,anchor,'        values = '+change+'\n'+anchor,kernel_witness)
    for name,change in [('detach','transform.detach()'),('numpy','torch.tensor(transform.detach().numpy())'),('zero-backward','transform.detach() + q.sum()*0'),('missing-tcp','transform @ transform.new_tensor([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])')]:
        before='        return transform[0] if q.ndim == 1 else transform'
        add(training_fk,name,before,'        transform = '+change+'\n'+before,kernel_witness)
    for name,before,after in [
        ('length',' / 0.9015',' / 1.0'),('rotation-scale',' / 8',' / 4'),
        ('joint-reduction','((normalized - target_normalized)**2).sum(-1)','((normalized - target_normalized)**2).mean(-1)'),
        ('rotation-detach',"lr = ((transform[..., :3, :3] - rotation)**2)","lr = ((transform[..., :3, :3].detach() - rotation)**2)"),
        ('position-mm','transform[..., :3, 3] - position','1000 * transform[..., :3, 3] - position'),
        ('limit-zero','(torch.relu(lower - q) / span)**2','(torch.relu(lower - q) / span)**2 * 0'),
        ('zero-lambda',"value = terms['q']","value = terms['q'] + terms['p']"),
        ('hidden-tanh',"else logits","else torch.tanh(logits)"),
        ('quaternion-sign','w, x, y, z = quaternion.unbind(-1)','w, x, y, z = quaternion.unbind(-1)\n    w = w.abs()')]:
        add(physics,name,before,after,physics_witness)
    add(c105,'invalid-denominator','    ordered=np.sort(np.r_[values,np.full(total-len(values),np.inf)])',
        '    total=len(values)\n    ordered=np.sort(np.asarray(values))',training_witness)
    add(c105,'confounded-limit',"('FK','FK_LIMIT')","('FK','FK_TANH')",training_witness)
    records=[]
    for original,name,before,after,witness in cases:
        witness(original)  # witness must pass unmodified production source
        path=Path(original.__file__); source=path.read_text(encoding='utf-8')
        if source.count(before)!=1: raise ValueError('unique mutation anchor: '+name)
        mutant=source.replace(before,after,1)
        module=types.ModuleType('neurokinematics.neural._mutant_'+name.replace('-','_'))
        module.__package__='neurokinematics.neural'; module.__file__=str(path)
        exec(compile(mutant,str(path)+'#'+name,'exec'),module.__dict__)
        status='SURVIVED'; reason=''
        try: witness(module)
        except (AssertionError,RuntimeError) as exc: status='KILLED';reason=repr(exc)
        except Exception as exc: status='ERROR';reason=repr(exc)
        (a.output/(name+'.diff')).write_text(''.join(difflib.unified_diff(source.splitlines(True),mutant.splitlines(True),fromfile=str(path),tofile=name)),encoding='utf-8',newline='\n')
        record=dict(name=name,status=status,killing_test=witness.__name__,reason=reason,source_sha256=sha(path))
        write_json(a.output/(name+'.json'),record);records.append(record)
    result=dict(status='PASS' if all(r['status']=='KILLED' for r in records) else 'FAIL',mutants=records)
    write_json(a.output/'summary.json',result)
    print(json.dumps(dict(status=result['status'],counts={s:sum(r['status']==s for r in records) for s in ['KILLED','SURVIVED','ERROR']})))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
