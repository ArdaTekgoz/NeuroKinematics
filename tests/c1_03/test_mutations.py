"""Real source mutants executed against unchanged numerical assertions."""
import difflib
import json
from pathlib import Path
import types
import numpy as np
import pytest
import torch
from conftest import analytic_inputs
from neurokinematics.kinematics.model import ROOT, load_robot
from neurokinematics.kinematics import torch_fk
from neurokinematics.kinematics.metrics import quaternion_error, quaternion_rotation
from neurokinematics.core import torch_validation as validation

KERNEL_SOURCE=Path(torch_fk.__file__).read_text(encoding='utf-8')
ANCHOR='        n = values.shape[0]'
MUTANTS={
 'M01-order':(ANCHOR,'        values = values.flip(-1)\n'+ANCHOR),
 'M02-degrees':(ANCHOR,'        values = values * (torch.pi / 180)\n'+ANCHOR),
 'M03-axis':('skew = skew64.to(dtype=q.dtype, device=q.device)','skew = -skew64.to(dtype=q.dtype, device=q.device)'),
 'M04-origin-order':('            transform = transform @ origin','            transform = transform if index is not None else transform @ origin'),
 'M05-omit-fixed':('            transform = transform @ origin','            transform = transform @ origin if index is not None else transform'),
 'M06-world-frame':('        return transform[0] if q.ndim == 1 else transform','        transform = q.new_tensor([[0.,-1.,0.,3.],[1.,0.,0.,4.],[0.,0.,1.,5.],[0.,0.,0.,1.]]) @ transform\n        return transform[0] if q.ndim == 1 else transform'),
 'M07-double-base':('        return transform[0] if q.ndim == 1 else transform','        transform = q.new_tensor([[1.,0.,0.,-3.],[0.,1.,0.,-4.],[0.,0.,1.,-5.],[0.,0.,0.,1.]]) @ transform\n        return transform[0] if q.ndim == 1 else transform'),
 'M08-hidden-f32':('        return transform[0] if q.ndim == 1 else transform','        transform = transform.float().to(q.dtype)\n        return transform[0] if q.ndim == 1 else transform'),
 'M09-batch':(ANCHOR,'        values = values.flip(0)\n'+ANCHOR),
 'M10-numpy':(ANCHOR,'        values = torch.from_numpy(values.detach().cpu().numpy()).to(q.device)\n'+ANCHOR),
 'M11-detach':(ANCHOR,'        values = values.detach()\n'+ANCHOR),
 'M11-no-grad':('                rotation = eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)','                with torch.no_grad():\n                    rotation = eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)'),
 'M12-inplace':(ANCHOR,'        values.add_(0.1)\n'+ANCHOR),
 'M13-wrong-backward':(ANCHOR,'        values = values.detach() + (values.flip(-1) - values.flip(-1).detach())\n'+ANCHOR),
 'M14-zero-gradient':(ANCHOR,'        values = values.detach() + values * 0\n'+ANCHOR),
 'M14-constant-gradient':(ANCHOR,'        if values.requires_grad: values.register_hook(lambda grad: torch.ones_like(grad))\n'+ANCHOR),
 'M15-forward-nan':('        return transform[0] if q.ndim == 1 else transform','        transform = transform * float("nan")\n        return transform[0] if q.ndim == 1 else transform'),
 'M15-backward-inf':(ANCHOR,'        if values.requires_grad: values.register_hook(lambda grad: grad * float("inf"))\n'+ANCHOR),
 'M20-strict-limits':('checked < limits[:, 0]','checked <= limits[:, 0]'),
 'M20-rounded-limits':('limits = self._limits.to(device=q.device)','limits = self._limits.to(device=q.device, dtype=q.dtype)'),
}

def compile_module(source,package):
    module=types.ModuleType('c103_mutant');module.__package__=package
    exec(compile(source,'<C1-03-source-mutant>','exec'),module.__dict__)
    return module

def numerical_witness(cls,config,robot):
    kernel=cls(robot);oracle=validation.Oracle(robot)
    q=torch.tensor([.3,-.6,.8,-1.,.5,-.7],dtype=torch.float64,requires_grad=True)
    out=kernel(q)
    assert out.dtype==q.dtype
    assert validation.pose_metrics(out.detach().numpy(),oracle.forward(q.detach().numpy()),'float64',config)['passed']
    derivative=torch.autograd.functional.jacobian(lambda x:validation.outputs_torch(kernel(x),config),q).numpy()
    fd,_=validation.finite_difference(oracle,q.detach().numpy(),config)
    assert validation.derivative_metrics(derivative,fd,config)['passed']
    batch=torch.stack((q.detach(),q.detach()*.5))
    result=kernel(batch)
    for i in range(2):
        assert validation.pose_metrics(result[i].detach().numpy(),oracle.forward(batch[i].numpy()),'float64',config)['passed']
    a4=cls(analytic_inputs('A4'))
    t=a4(torch.tensor([np.pi/2],dtype=torch.float64))
    assert np.linalg.norm(t[:3,3].numpy()-[1,4,3])<=1e-9
    bounds=np.array(robot.limits);mid=bounds.mean(1)
    for j in range(6):
        edge=mid.copy();edge[j]=bounds[j,0]
        kernel(torch.tensor(edge))
        invalid=edge.astype('f4');invalid[j]=np.float32(bounds[j,0])
        if float(invalid[j])>=bounds[j,0]:invalid[j]=np.nextafter(invalid[j],np.float32(-np.inf))
        try:kernel(torch.tensor(invalid))
        except ValueError:pass
        else:raise AssertionError('one ULP outside accepted')

def save(request,ident,source,changed,error):
    folder=request.config.getoption('--mutation-output')
    if folder:
        p=Path(folder);p.mkdir(parents=True,exist_ok=True)
        diff=''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),fromfile='baseline',tofile=ident))
        (p/(ident+'.diff')).write_text(diff,encoding='utf-8',newline='\n')
        validation.write_json(p/(ident+'.json'),dict(id=ident,status='KILLED',baseline='PASS',
            source_sha256=validation.sha(source.encode()),mutant_sha256=validation.sha(changed.encode()),
            killing_test=request.node.nodeid,observed_failure=str(error),expected_failure='numeric/graph/contract rejection',diff=ident+'.diff'))

@pytest.mark.parametrize('ident',list(MUTANTS))
def test_kernel_mutants(ident,request,robot):
    config=json.loads((ROOT/'experiments/C1-03/config.json').read_text())
    numerical_witness(torch_fk.TorchSerialChain,config,robot)
    before,after=MUTANTS[ident]
    assert KERNEL_SOURCE.count(before)==1
    changed=KERNEL_SOURCE.replace(before,after)
    if ident=='M04-origin-order':
        changed=changed.replace('transform = transform @ motion','transform = transform @ motion @ origin')
    mutant=compile_module(changed,'neurokinematics.kinematics')
    try:numerical_witness(mutant.TorchSerialChain,config,robot)
    except (AssertionError,ValueError,RuntimeError) as error:
        save(request,ident,KERNEL_SOURCE,changed,error)
    else:pytest.fail('SURVIVED '+ident)

def test_M16_stencil_mutant(request,robot):
    source=Path(validation.__file__).read_text(encoding='utf-8')
    changed=source.replace("if q[j]-h<oracle.bounds[j,0]:","if False:").replace("elif q[j]+h>oracle.bounds[j,1]:","elif False:")
    module=compile_module(changed,'neurokinematics.core')
    config=json.loads((ROOT/'experiments/C1-03/config.json').read_text());oracle=validation.Oracle(robot)
    q=np.array(robot.limits).mean(1);q[0]=robot.limits[0][0]
    assert validation.finite_difference(oracle,q,config,edge=True)[1][0]=='forward2'
    with pytest.raises(ValueError) as error:module.finite_difference(oracle,q,config,edge=True)
    save(request,'M16-stencil',source,changed,error.value)

def test_M17_shared_oracle_mutant(request,robot):
    source=Path(validation.__file__).read_text(encoding='utf-8')
    changed=source.replace('self.backend=PinocchioFK(inputs)','self.backend=TorchFK.from_frozen()')
    config=json.loads((ROOT/'experiments/C1-03/config.json').read_text())
    validation.Oracle(robot).forward(np.zeros(6))
    module=compile_module(changed,'neurokinematics.core')
    with pytest.raises(ValueError,match='independent Pinocchio') as error:module.Oracle(robot).forward(np.zeros(6))
    save(request,'M17-shared-oracle',source,changed,error.value)

@pytest.mark.parametrize('variant',['order','sign'])
def test_M19_quaternion_mutant(variant,request):
    from neurokinematics.kinematics import metrics
    source=Path(metrics.__file__).read_text(encoding='utf-8')
    q=np.array([.5,.5,-.5,.5])
    assert np.linalg.norm(quaternion_rotation(q)-quaternion_rotation(-q))==0
    assert quaternion_error(q,-q)==0
    if variant=='order':changed=source.replace('w, x, y, z = quaternion_wxyz(value)','x, y, z, w = quaternion_wxyz(value)')
    else:changed=source.replace(' or np.array_equal(a, -b)','').replace('abs(np.dot(a, b))','np.dot(a, b)')
    module=compile_module(changed,'neurokinematics.kinematics')
    try:
        assert np.linalg.norm(module.quaternion_rotation(q)-quaternion_rotation(q))<=1e-9
        assert module.quaternion_error(q,-q)==0
    except AssertionError as error:save(request,'M19-'+variant,source,changed,error)
    else:pytest.fail('quaternion mutant survived')
