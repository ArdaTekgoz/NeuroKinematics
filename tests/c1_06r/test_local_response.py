import numpy as np
from neurokinematics.neural import c106r_local_response as a


def test_ideal_and_hold_response_controls_and_sign_mutant():
    robot=a.r.load_robot();q=np.mean(np.asarray(robot.limits),axis=1)
    j=a.s.IndependentJacobian(robot).jacobian(q)
    assert a.response_error(j,np.eye(6))==0
    assert np.isclose(a.response_error(j,np.zeros((6,6))),1)
    assert np.isclose(a.response_error(j,-np.eye(6)),2)


def test_feature_tangent_frame_and_scaler_against_independent_fk_fd():
    rows=a.z.load_source('directions').take(np.arange(0,64,8));zero=a.zero_queries(rows,'current')
    norm=a.d.read_json(a.d.ROOT/'experiments/C1-06R/diagnostic3/normalization.json')
    local=a.d.read_json(a.z.BASE/'n64-scaler.json');robot=a.r.load_robot()
    jac=a.s.IndependentJacobian(robot);fk=a.s.IndependentFK(robot);h=1e-6
    shifted=a.perturb_queries(zero,h)
    for mode in ('RAW','LOCAL_Z'):
        feature=a.feature_values(shifted,mode,norm,local,np.float64)[:,:7].reshape(8,6,2,7)
        fd=((feature[:,:,0]-feature[:,:,1])/(2*h)).transpose(0,2,1)
        expected=np.asarray([a.feature_tangent(jac.jacobian(q),fk.forward_kinematics(q)[:3,:3],mode,norm,local) for q in zero.q_current])
        assert np.allclose(fd,expected,atol=1e-5,rtol=1e-3)
        wrong=np.asarray([a.feature_tangent(jac.jacobian(q),np.eye(3),mode,norm,local) for q in zero.q_current])
        assert np.linalg.norm(fd-wrong)>1


def test_zero_and_small_target_oracles_preserve_limits_and_pose():
    roots=a.z.load_source('directions').take(np.arange(0,32,8));zero=a.zero_queries(roots,'current')
    exact=a.d.geometric_metrics(zero.q_current,zero)
    assert exact['profile_a']==exact['profile_b']==4
    shifted=a.perturb_queries(zero,1e-5)
    oracle=a.d.geometric_metrics(shifted.q_target,shifted)
    held=a.d.geometric_metrics(shifted.q_current,shifted)
    assert oracle['profile_a']==oracle['profile_b']==48
    assert held['profile_a']==48  # Tiny-step A success alone cannot establish response.


def test_fd_axis_order_recovers_identity_and_nontrivial_linear_map():
    n=3;h=.001;k=np.arange(36).reshape(6,6)/10
    deltas=np.repeat(np.eye(6),2,axis=0)*np.tile([1,-1],6)[:,None]*h
    outputs=np.tile(deltas@k.T,(n,1))
    assert np.allclose(a.fd_matrix(outputs,n,h),k)
