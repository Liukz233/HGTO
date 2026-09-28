"""Exercise the real density/network/Adam path with an injected rejected state."""

import copy
import numpy as np
import pytest
import torch
from hgto.nonlinear import optimization as m
from test_nonlinear_oc import small_case


@pytest.mark.parametrize("exhaust", [False, True])
def test_retry_restores_adam_and_clears_stale_gradients(monkeypatch, tmp_path, exhaust):
    torch.set_num_threads(2)
    setup = small_case()
    spec = dict(
        case=setup.name,
        nx=8,
        ny=4,
        L=4.0,
        H=2.0,
        volume_fraction=0.45,
        unit_direction=[0.0, -1.0],
        port_nodes=np.flatnonzero(setup.f[:, 1]).tolist(),
    )
    actual_network, actual_solve = m.graph_network, m.solve
    actual_backward, Adam = torch.Tensor.backward, torch.optim.Adam
    captured, steps, errors = {}, [], []

    def network(*a, **kw):
        net, inputs = actual_network(*a, **kw)
        captured["net"] = net
        return net, inputs

    class AuditedAdam(Adam):
        def step(self, *a, **kw):
            steps.append(
                dict(
                    params=[p.detach().clone() for p in captured["net"].parameters()],
                    state=copy.deepcopy(self.state_dict()),
                    grad=[p.grad.clone() for p in captured["net"].parameters()],
                )
            )
            return super().step(*a, **kw)

    def backward(rho, gradient=None, *a, **kw):
        params = list(captured["net"].parameters())
        expected = torch.autograd.grad(rho, params, grad_outputs=gradient, retain_graph=True)
        result = actual_backward(rho, gradient, *a, **kw)
        errors.append(max(float((p.grad - g).abs().max()) for p, g in zip(params, expected)))
        return result

    calls = 0

    def solve(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 3 or (exhaust and calls >= 3):
            raise RuntimeError("injected rejected mechanics trial")
        return actual_solve(*a, **kw)

    monkeypatch.setattr(m, "graph_network", network)
    monkeypatch.setattr(m, "solve", solve)
    monkeypatch.setattr(torch.optim, "Adam", AuditedAdam)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    out = tmp_path / "run"
    m.run(
        setup.name,
        "nh",
        0.001,
        out,
        setup=setup,
        spec=spec,
        steps=1,
        max_updates=3,
        uniform_initial=True,
        architecture=dict(n_freq=4, hidden_dim=8, sigma=1.0, paper_K=1),
        device="cpu",
        load_steps=3,
        beta_final=2.0,
        backtracking=True,
        acceptance="fixed_parameter_descent",
        early_stopping=False,
        max_candidate_backtracks=2,
        learning_rate=0.001,
        final_learning_rate=0.001,
    )
    # Calls 3 and 4 are the rejected step and its half-size retry. They must
    # originate from identical parameters, Adam moments and previous gradient.
    for key in ("params", "grad"):
        for a, b in zip(steps[1][key], steps[2][key]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in steps[1]["state"]["state"].items():
        for name, a in value.items():
            torch.testing.assert_close(a, steps[2]["state"]["state"][key][name], rtol=0, atol=0)
    assert (
        steps[2]["state"]["param_groups"][0]["lr"] == steps[1]["state"]["param_groups"][0]["lr"] / 2
    )
    assert max(errors) < 1e-13
    import json

    record = json.loads((out / "record.json").read_text())
    if exhaust:
        assert record["termination"] == "stalled"
        assert record["design_updates"] == 1
        np.testing.assert_array_equal(
            np.load(out / "rho.npy"), np.load(out / "last_accepted_density.npy")
        )
    else:
        assert record["design_updates"] == 3
        assert calls == 5
