import math
from dataclasses import replace
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from conftest import analytic_inputs
from neurokinematics.kinematics.torch_fk import TorchSerialChain
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian

@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('case,q,p', [
    ('A1',[math.pi/2,-math.pi/2],[3,1,0]),
    ('A2',[math.pi/2,0],[0,0,4]),
    ('A3',[math.pi/2,-math.pi/2],[1,3,0]),
    ('A4',[math.pi/2],[1,4,3]),
])
def test_analytic(case,q,p,dtype):
    inputs=analytic_inputs(case); kernel=TorchSerialChain(inputs)
    value=torch.tensor(q,dtype=dtype,requires_grad=True)
    out=kernel(value)
    tol=1e-9 if dtype==torch.float64 else 1e-5
    assert out.shape==(4,4) and out.dtype==dtype and out.device==value.device
    actual=out.detach().double().numpy()
    assert np.linalg.norm(actual[:3,3]-p)<=tol
    expected=PinocchioFK(inputs).reference_forward_kinematics(value.detach().double().numpy())
    assert np.linalg.norm(actual[:3,:3]-expected[:3,:3])<=tol
    if case in ('A1','A3','A4'):
        rotation=([[0,0,1],[1,0,0],[0,1,0]] if case=='A4' else [[1,0,0],[0,0,-1],[0,1,0]])
        assert np.linalg.norm(actual[:3,:3]-rotation)<=tol
    out[:3,3].sum().backward(retain_graph=True)
    assert torch.isfinite(value.grad).all()
    value.grad=None
    (out[:3,:3]*out.new_tensor([[.3,-.2,.5],[.7,.11,-.4],[-.6,.9,.13]])).sum().backward()
    assert torch.isfinite(value.grad).all()

def test_zero_robot(public_fk,robot):
    t=public_fk(torch.zeros(6,dtype=torch.float64),robot_id=robot.robot_id,joint_names=robot.joint_names)
    assert torch.linalg.vector_norm(t[:3,3]-t.new_tensor([.980,0,.435]))<=1e-9
    assert torch.linalg.matrix_norm(t[:3,:3]-t.new_tensor([[0,0,1],[0,1,0],[-1,0,0]]))<=1e-9

@pytest.mark.parametrize('case',['A1','A3','A4'])
def test_base_and_jacobian(case):
    inputs=analytic_inputs(case)
    q=torch.tensor([math.pi/2] if case=='A4' else [math.pi/2,-math.pi/2],dtype=torch.float64,requires_grad=True)
    kernel=TorchSerialChain(inputs)
    t=kernel(q); derivatives=torch.autograd.functional.jacobian(kernel,q)
    r=t[:3,:3].detach().numpy(); d=derivatives.detach().numpy()
    angular=[]
    for j in range(len(q)):
        a=d[:3,:3,j]@r.T; s=(a-a.T)/2
        angular.append([s[2,1],s[0,2],s[1,0]])
    jac=np.vstack((d[:3,3,:],np.array(angular).T))
    for ref in [IndependentJacobian(inputs),PinocchioJacobian(inputs)]:
        assert np.allclose(jac,ref.jacobian(q.detach().numpy()),atol=1e-5,rtol=1e-3)
    if case=='A4': assert np.allclose(jac[:,0],[-2,0,0,0,0,1],atol=1e-9,rtol=0)
    xml=ET.fromstring(inputs.urdf)
    xml.find("joint[@name='mount']/origin").set('xyz','-2 7 1')
    xml.find("joint[@name='mount']/origin").set('rpy','.3 -.4 .5')
    changed=replace(inputs,urdf=ET.tostring(xml))
    assert torch.equal(kernel(q),TorchSerialChain(changed)(q))
    assert np.allclose(t.detach().numpy(),PinocchioFK(changed).reference_forward_kinematics(q.detach().numpy()),atol=1e-9,rtol=0)

@pytest.mark.parametrize('angle',[0,math.pi/2,-math.pi/2,math.pi-1e-7,math.pi,math.pi+1e-7])
def test_pi_matrix_derivative(angle):
    kernel=TorchSerialChain(analytic_inputs('A4'))
    q=torch.tensor([angle],dtype=torch.float64,requires_grad=True)
    t=kernel(q); d=torch.autograd.functional.jacobian(kernel,q)
    c,s=math.cos(angle),math.sin(angle)
    r=np.array([[c,0,s],[s,0,-c],[0,1,0]])
    dr=np.array([[-s,0,c],[c,0,s],[0,0,0]])
    assert np.linalg.norm(t[:3,:3].detach().numpy()-r)<=1e-9
    assert np.linalg.norm(d[:3,:3,0].detach().numpy()-dr)<=1e-9

def test_no_reference_import():
    code='''
import sys, importlib.abc
from neurokinematics.kinematics.custom_fk import IndependentFK
def forbidden(*args, **kwargs):
 raise AssertionError('NumPy FK forbidden in Torch construction/forward')
IndependentFK.forward_kinematics=forbidden
# The unchanged Foundations package initializer eagerly exports IndependentFK.
# Trap its real callable; all subsequent reference imports are forbidden.
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith('pinocchio') or fullname.endswith('custom_fk') or fullname.endswith('pinocchio_fk'):
   raise ImportError('reference forbidden')
sys.meta_path.insert(0,Block())
import torch
from neurokinematics.kinematics.torch_fk import TorchFK
f=TorchFK.from_frozen()
q=torch.zeros(6,dtype=torch.float64,requires_grad=True)
t=f(q,robot_id=f.robot_id,joint_names=f.joint_names)
t[:3,:3].sum().backward()
assert q.grad is not None
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
