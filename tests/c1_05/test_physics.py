"""Analytic and negative gates, independent of the robot's shared parser."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from neurokinematics.kinematics.model import RobotInputs, load_robot
from neurokinematics.kinematics.torch_fk import TorchFK
from neurokinematics.neural.training_fk import TrainingFK, TrainingSerialChain, DOMAIN
from neurokinematics.neural.physics import PhysicsLoss, quaternion_matrix, limit_loss, normalized_head, combined


def analytic(dof):
    joint2 = '' if dof==1 else '''<link name="b"/><joint name="j2" type="revolute"><parent link="a"/><child link="b"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'''
    xml = f'''<robot name="fixture"><link name="world"/><link name="base"/><link name="a"/><link name="tcp"/>
    <joint name="mount" type="fixed"><parent link="world"/><child link="base"/><origin xyz="3 4 5" rpy=".3 -.2 .5"/></joint>
    <joint name="j1" type="revolute"><parent link="base"/><child link="a"/><origin xyz="1 2 3"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>{joint2}
    <joint name="tool" type="fixed"><parent link="{'a' if dof==1 else 'b'}"/><child link="tcp"/><origin xyz="2 0 0" rpy="1.5707963267948966 0 0"/></joint></robot>'''
    return RobotInputs(xml.encode(),tuple(f'j{i+1}' for i in range(dof)),tuple([(-1.,1.)]*dof),'base','a','tcp','fixture',{})


@pytest.mark.parametrize('dof',[1,2])
@pytest.mark.parametrize('angle',[-7.,-math.pi,0.,math.pi-1e-6,math.pi,math.pi+1e-6,2*math.pi])
def test_analytic(dof,angle):
    f=TrainingSerialChain(analytic(dof)); q=torch.tensor([angle]*dof,dtype=torch.float64,requires_grad=True)
    t=f(q); jac=torch.autograd.functional.jacobian(f,q).numpy(); a=dof*angle; c,s=math.cos(a),math.sin(a)
    expected_r=np.array([[c,0,s],[s,0,-c],[0,1,0]])
    expected_p=np.array([1+2*c,2+2*s,3.])
    if dof==2: expected_p+=np.array([2*math.cos(angle),2*math.sin(angle),0])
    assert np.linalg.norm(t[:3,:3].detach().numpy()-expected_r)<=1e-9
    assert np.linalg.norm(t[:3,3].detach().numpy()-expected_p)<=1e-9
    dr=np.array([[-s,0,c],[c,0,s],[0,0,0]])
    for j in range(dof):
        dp=np.array([-2*s,2*c,0.])
        if dof==2 and j==0: dp+=np.array([-2*math.sin(angle),2*math.cos(angle),0])
        assert np.linalg.norm(jac[:3,:3,j]-dr)<=1e-9
        assert np.linalg.norm(jac[:3,3,j]-dp)<=1e-9


def test_public_limits_and_opt_in():
    f=TrainingFK.from_frozen(domain=DOMAIN); public=TorchFK.from_frozen()
    q=torch.full((6,),10.,dtype=torch.float64,requires_grad=True)
    kwargs=dict(robot_id=f.robot_id,joint_names=f.joint_names)
    with pytest.raises(ValueError): public(q,**kwargs)
    with pytest.raises(ValueError): TrainingFK.from_frozen(domain='default')
    with pytest.raises(TypeError): TrainingFK.from_frozen()
    with pytest.raises(ValueError): f(q,**dict(kwargs,units='deg'))
    with pytest.raises(ValueError): f(q,**dict(kwargs,joint_names=tuple(reversed(f.joint_names))))
    with pytest.raises(ValueError): f(q,**dict(kwargs,robot_id='wrong'))
    assert torch.isfinite(torch.autograd.grad(f(q,**kwargs)[:3,3].sum(),q)[0]).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('n',[1,2,7,1024])
def test_batch_extreme(dtype,n):
    f=TrainingFK.from_frozen(domain=DOMAIN); kw=dict(robot_id=f.robot_id,joint_names=f.joint_names)
    q=torch.arange(6,dtype=dtype)+10
    assert torch.allclose(f(q.expand(n,6),**kw),f(q,**kw).expand(n,4,4),atol=1e-6,rtol=0)
    for j in range(6):
        for sign in [-1,1]:
            q=torch.zeros(6,dtype=dtype); q[j]=sign*torch.finfo(dtype).max
            assert torch.isfinite(f(q,**kw)).all()
    for bad in [torch.full((6,),float('nan')),torch.full((6,),float('inf')),torch.zeros(0,6),torch.zeros(5)]:
        with pytest.raises(ValueError): f(bad,**kw)
    with pytest.raises(TypeError): f(torch.zeros(6,dtype=torch.int64),**kw)


@pytest.mark.parametrize('angle',[0.,.4,math.pi-1e-6,math.pi,math.pi+1e-6])
def test_quaternion_chordal(angle):
    q=torch.tensor([math.cos(angle/2),0.,0.,math.sin(angle/2)],dtype=torch.float64)
    r=quaternion_matrix(q)
    assert torch.equal(r,quaternion_matrix(-q))
    assert abs(float(((r-torch.eye(3))**2).sum()/8)-math.sin(angle/2)**2)<1e-12
    assert np.allclose(r.numpy(),[[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]],atol=1e-12)
    with pytest.raises(ValueError): quaternion_matrix(torch.zeros(4))


def test_loss_scaling_and_gradient():
    loss=PhysicsLoss(); z=torch.tensor([[1.2,-.2,.3,.4,.5,.6]],dtype=torch.float64,requires_grad=True)
    target=z.detach()+.1; q=loss.raw(z)
    t=loss.fk(q,robot_id=loss.fk.robot_id,joint_names=loss.fk.joint_names).detach()
    p=t[:,:3,3]+torch.tensor([.9015,0.,0.]); r=t[:,:3,:3]@quaternion_matrix(torch.tensor([0.,0.,0.,1.],dtype=torch.float64))
    terms,actual_q,_=loss.components(z,target,p,r)
    assert torch.equal(q,actual_q)
    assert float(terms['q'].detach())==pytest.approx(.06)
    assert float(terms['p'].detach())==pytest.approx(1.)
    assert float(terms['R'].detach())==pytest.approx(1.)
    assert float(terms['lim'].detach())==pytest.approx(.08)
    for key in ['p','q','lim']:
        assert torch.linalg.vector_norm(torch.autograd.grad(terms[key].sum(),z,retain_graph=True)[0])>0
    arm=dict(lambda_p=0.,lambda_R=0.,lambda_lim=0.)
    direct=((z-target)**2).sum(-1).mean()
    assert torch.equal(combined(terms,arm),direct)
    assert torch.equal(torch.autograd.grad(combined(terms,arm),z,retain_graph=True)[0],torch.autograd.grad(direct,z)[0])


def test_limit_and_tanh():
    lo=torch.tensor([-2.],dtype=torch.float64); hi=torch.tensor([4.],dtype=torch.float64)
    q=torch.tensor([[-8.],[1.],[10.],[-2.],[4.]],dtype=torch.float64,requires_grad=True)
    l=limit_loss(q,lo,hi)
    assert torch.equal(l,torch.tensor([1.,0.,1.,0.,0.],dtype=torch.float64))
    assert torch.allclose(torch.autograd.grad(l.sum(),q)[0].flatten(),torch.tensor([-1/3,0.,1/3,0.,0.],dtype=torch.float64))
    z=torch.tensor([-10.,0.,10.],dtype=torch.float64,requires_grad=True)
    y=normalized_head(z,'FK_TANH')
    assert torch.all((y>=0)&(y<=1))
    assert torch.allclose(torch.autograd.grad(y.sum(),z)[0],(1-torch.tanh(z)**2)/2)
    for arm in ['Q','FK','FK_LIMIT']: assert normalized_head(z,arm) is z


def test_loader_seals_test():
    from neurokinematics.neural.c104 import _read_split, contracts
    c,_,n=contracts()
    with pytest.raises(ValueError,match='sealed'): _read_split('test',c,n,label_fk=False)
