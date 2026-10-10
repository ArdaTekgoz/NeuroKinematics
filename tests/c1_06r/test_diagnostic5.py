import numpy as np
import torch
from neurokinematics.neural import c106r_diagnostic5 as s


def test_pose_delta_uses_base_axes_and_correct_sign():
    teacher=np.eye(4);teacher[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    angle=.03
    rx=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    predicted=teacher.copy();predicted[:3,:3]=rx@teacher[:3,:3];predicted[:3,3]=[.001,-.002,.003]
    np.testing.assert_allclose(s.pose_delta(predicted,teacher),[.001,-.002,.003,angle,0,0],atol=1e-14)


def test_jacobian_predicts_small_pose_perturbation():
    robot=s.r.load_robot();fk=s.r.PinocchioFK(robot);j=s.IndependentJacobian(robot)
    q=np.array([.3,-.6,.8,-1,.5,-.7]);dq=np.array([1,-2,3,-1,2,-3])*1e-7
    np.testing.assert_allclose(s.pose_delta(fk.reference_forward_kinematics(q+dq),fk.reference_forward_kinematics(q)),j.jacobian(q)@dq,atol=1e-12,rtol=0)


def test_gradient_probe_does_not_modify_parameters_and_matches_Q_derivative():
    robot=s.r.load_robot();fk=s.r.PinocchioFK(robot)
    bounds=np.asarray(robot.limits);q=np.array([.3,-.6,.8,-1,.5,-.7]);target=q+.01
    x=torch.zeros((2,13));x[:,-6:]=torch.tensor((q-bounds[:,0])/(bounds[:,1]-bounds[:,0]),dtype=torch.float32)
    y=torch.tensor(np.tile((target-bounds[:,0])/(bounds[:,1]-bounds[:,0]),(2,1)),dtype=torch.float32)
    t=fk.reference_forward_kinematics(target)
    p=torch.tensor(np.tile(t[:3,3],(2,1)));rot=torch.tensor(np.tile(t[:3,:3],(2,1,1)))
    model=s.e.build_model(2026100901,256,device='cpu');before=s.d.state_hash(model)
    result,outputs=s.gradient_probe(model,x,y,p,rot)
    assert s.d.state_hash(model)==before and all(p.grad is None for p in model.parameters())
    assert abs(result['losses']['Q']-float(((x[:,-6:]-y)**2).sum(-1).mean()))<1e-12
    assert all(np.isfinite(v).all() for v in outputs.values())
    assert result['parameter_norms']['Q']>0 and result['parameter_norms']['P']>0 and result['parameter_norms']['R']>0
