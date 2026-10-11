import numpy as np
import torch
from neurokinematics.neural import c106r_centered as c


def fixture():
    torch.manual_seed(19)
    net=torch.nn.Sequential(torch.nn.Linear(13,17),torch.nn.SiLU(),torch.nn.Linear(17,6)).double()
    return net,c.Centered(net,dict(mean=[.1,-.2,.3],std=[.4,.5,.6]))


def test_zero_input_preserves_current_without_mutation_and_zero_output():
    net,model=fixture();x=torch.randn(9,13,dtype=torch.float64);before=x.clone()
    x0=model.zero_input(x)
    assert torch.equal(x,before) and torch.equal(x0[:,-6:],x[:,-6:])
    assert torch.equal(model(x0),torch.zeros((9,6),dtype=torch.float64))
    assert torch.equal(c.d.predict(model,x0,'residual'),x[:,-6:])
    assert model.zero_pose.shape==(7,) and torch.equal(model.zero_pose[3:],torch.tensor([1.,0.,0.,0.],dtype=torch.float64))


def test_centering_preserves_target_derivative_but_cancels_current_on_zero_manifold():
    net,model=fixture();x=torch.randn(1,13,dtype=torch.float64,requires_grad=True)
    raw=torch.autograd.functional.jacobian(net,x)[0,:,0,:]
    centered=torch.autograd.functional.jacobian(model,x)[0,:,0,:]
    assert torch.equal(raw[:,:7],centered[:,:7])
    q=torch.randn(1,6,dtype=torch.float64,requires_grad=True)
    def on_zero(q):return model(torch.cat((model.zero_pose[None],q),dim=1))
    assert torch.equal(torch.autograd.functional.jacobian(on_zero,q),torch.zeros(1,6,1,6,dtype=torch.float64))


def test_nonzero_target_finite_difference_and_parameter_gradients():
    _,model=fixture();x=torch.randn(1,13,dtype=torch.float64,requires_grad=True);h=1e-5
    jac=torch.autograd.functional.jacobian(model,x)[0,:,0,:]
    fd=[]
    for j in range(13):
        delta=torch.zeros_like(x);delta[0,j]=h
        fd.append(((model(x+delta)-model(x-delta))/(2*h)).detach()[0])
    assert torch.allclose(jac,torch.stack(fd,dim=1),atol=1e-9,rtol=1e-6)
    model(x).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) for p in model.parameters())


def test_real_zero_pose_roundtrip_strict_profiles_and_same_parameter_count():
    rows=c.a.zero_queries(c.z.load_source('directions').take(np.arange(0,256,8)),'current')
    norm=c.d.read_json(c.d.ROOT/'experiments/C1-06R/diagnostic3/normalization.json')
    torch.manual_seed(42)
    net=torch.nn.Sequential(torch.nn.Linear(13,32),torch.nn.SiLU(),torch.nn.Linear(32,6))
    model=c.Centered(net,norm)
    x=torch.tensor(c.r.features(rows,'relative',norm))
    q=c.r.decode_joints(c.d.predict(model,x,'residual')).detach().numpy()
    result=c.d.geometric_metrics(q,rows)
    assert result['profile_a']==result['profile_b']==32
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in net.parameters())
