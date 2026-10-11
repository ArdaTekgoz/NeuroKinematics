import numpy as np
import pytest
import torch
from neurokinematics.neural import c106r_diagnostic2 as d


def test_residual_initially_preserves_current_and_absolute_is_midpoint():
    model = d.build_model(2026100901, "cpu")
    x = torch.randn(20, 13)
    x[:, -6:] = torch.rand(20, 6)
    assert torch.equal(d.predict(model, x, "residual"), x[:, -6:])
    assert torch.equal(d.predict(model, x, "absolute"), torch.full((20, 6), .5))
    assert d.state_hash(model) == d.state_hash(d.build_model(2026100901, "cpu"))
    with pytest.raises(ValueError):
        d.predict(model, x, "unknown")


def test_residual_offset_and_gradient_do_not_clip():
    model = d.build_model(1, "cpu")
    with torch.no_grad():
        model.layers[-1].bias.fill_(2.)
    x = torch.zeros(4, 13)
    x[:, -6:] = .2
    z = d.predict(model, x, "residual")
    assert (z > 1).all()
    z.sum().backward()
    assert torch.equal(model.layers[-1].bias.grad, torch.full((6,), 4.))


def test_matched_design_keeps_roots_and_shared_local():
    train, _ = d.load_data(label_fk=False)
    subsets, shared = d.matched_rows(train, 64)
    a, b = subsets["local"], subsets["mixed"]
    assert np.array_equal(a.source_sample_id, b.source_sample_id)
    assert (a.mode == "local").sum() == 64
    assert (b.mode == "local").sum() == (b.mode == "wide").sum() == 32
    assert set(shared.pair_id) <= set(a.pair_id) & set(b.pair_id)
    assert len(set(a.source_sample_id)) == 64
    with pytest.raises(ValueError):
        d.matched_rows(train, 63)
    contract = d.checkpoint_contract(d.read_json(d.CONFIG), "residual", a)
    assert type(contract["torch"]) is str
