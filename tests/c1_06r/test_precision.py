import pytest
import torch
from neurokinematics.kinematics.model import load_robot
from neurokinematics.neural.c106r_precision import decode_joints


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_endpoints_and_unclipped_invalid(dtype, device):
    b = torch.tensor(load_robot().limits, dtype=torch.float64, device=device)
    z = torch.tensor([[0.]*6, [1.]*6, [-.001]*6, [1.001]*6], dtype=dtype, device=device)
    q = decode_joints(z)
    assert torch.equal(q[0], b[:, 0])
    assert torch.equal(q[1], b[:, 1])
    assert (q[2] < b[:, 0]).all() and (q[3] > b[:, 1]).all()


def test_derivative_is_joint_span_including_midpoint():
    z = torch.tensor([[0.]*6, [.2]*6, [.5]*6, [.8]*6, [1.]*6], dtype=torch.float64, requires_grad=True)
    decode_joints(z).sum().backward()
    b = torch.tensor(load_robot().limits, dtype=torch.float64)
    assert torch.equal(z.grad, (b[:, 1]-b[:, 0]).expand_as(z))
    assert torch.autograd.gradcheck(decode_joints, (z,), eps=1e-6, atol=1e-5, rtol=.001)


def test_nonfinite_is_preserved_for_invalid_prediction_accounting():
    z = torch.zeros(6)
    z[0] = torch.nan
    assert torch.isnan(decode_joints(z)[0])
    with pytest.raises(ValueError):
        decode_joints(torch.zeros(7))
