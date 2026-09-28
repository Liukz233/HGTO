"""Regression for Newton cycling in the weighted, filtered volume equation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from hgto.linear2d.design import volume_density


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_perforated_optimization_root_and_implicit_gradient(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    fixture = np.load(Path(__file__).parent / "fixtures/perforated_volume_root.npz")
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    z = tensor(fixture["logits"]).requires_grad_()
    weights = tensor(fixture["weights"])
    matrix = torch.sparse_coo_tensor(
        torch.as_tensor(fixture["indices"], dtype=torch.long, device=device),
        tensor(fixture["values"]),
        size=(len(z), len(z)),
        check_invariants=True,
    ).coalesce()

    def density(value):
        return volume_density(
            value,
            float(fixture["volume"]),
            matrix,
            float(fixture["beta"]),
            float(fixture["rho_min"]),
            weights,
        )

    rho = density(z)
    assert float((rho @ weights / weights.sum()).detach()) == pytest.approx(0.4, abs=2e-14)
    mass_grad = torch.autograd.grad(rho @ weights, z, retain_graph=True)[0]
    assert float(mass_grad.abs().max()) < 1e-13
    rng = np.random.default_rng(29)
    objective_weights = tensor(rng.normal(size=len(z)))
    direction = tensor(rng.normal(size=len(z)))
    direction /= direction.norm()
    derivative = torch.autograd.grad(rho @ objective_weights, z)[0] @ direction
    h = 1e-5
    finite = (
        (density(z.detach() + h * direction) - density(z.detach() - h * direction))
        @ objective_weights
    ) / (2 * h)
    assert float(derivative) == pytest.approx(float(finite), rel=3e-6, abs=1e-8)
