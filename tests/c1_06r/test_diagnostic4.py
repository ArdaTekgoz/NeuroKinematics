import torch
from neurokinematics.neural import c106r_diagnostic4 as e


def test_reference_initialization_matches_historical_model():
    old = e.d.build_model(2026100901, device='cpu')
    new = e.build_model(2026100901, 256, device='cpu')
    assert e.d.state_hash(old) == e.d.state_hash(new)
    x = torch.randn(17, 13)
    assert torch.equal(old(x), new(x))


def test_capacity_changes_parameters_but_preserves_initial_function():
    small = e.build_model(2026100901, 256, device='cpu')
    wide = e.build_model(2026100901, 512, device='cpu')
    assert sum(p.numel() for p in small.parameters()) == 136710
    assert sum(p.numel() for p in wide.parameters()) == 535558
    x = torch.randn(17, 13)
    for model in (small, wide):
        assert torch.equal(e.d.predict(model, x, 'residual'), x[:, -6:])


def test_loss_scaling_preserves_zero_and_multiplies_gradient():
    model = e.build_model(2026100901, 256, device='cpu').double()
    x, y = torch.randn(17, 13, dtype=torch.float64), torch.randn(17, 6, dtype=torch.float64)
    plain = e.q_loss(model, x, y)
    grads = torch.autograd.grad(plain, tuple(model.parameters()))
    scaled = e.q_loss(model, x, y, 1e6)
    scaled_grads = torch.autograd.grad(scaled, tuple(model.parameters()))
    assert torch.equal(scaled, plain * 1e6)
    for a,b in zip(grads, scaled_grads):
        torch.testing.assert_close(b, a*1e6, rtol=1e-12, atol=1e-10)
    assert float(e.q_loss(model, x, x[:, -6:], 1e6).detach()) == 0
