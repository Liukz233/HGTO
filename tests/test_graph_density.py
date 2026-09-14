"""Density-network cache and weighted volume derivatives."""

import copy
import pytest
import torch

from hgto.fem.mesh import structured_q4, build_element_dual_graph
from hgto.topopt.parameterize import ChebNetDensity
from hgto.topopt.parameterize.features.static import unit_bbox_centroids
from hgto.topopt.pipeline.projection import volume_project


def test_static_cache_keeps_outputs_and_parameter_gradients():
    mesh = structured_q4(6, 3)
    coords = unit_bbox_centroids(mesh)
    edges = torch.as_tensor(build_element_dual_graph(mesh)["edge_index"])
    torch.manual_seed(4)
    network = ChebNetDensity(12, 1.0, 4, hidden_dim=16, paper_K=2)
    cached = copy.deepcopy(network)
    cached.prepare_static(coords, edges)
    output = network(coords, edges)
    output_cached = cached(coords, edges)
    torch.testing.assert_close(output, output_cached, rtol=1e-12, atol=1e-12)
    weights = torch.linspace(-1.0, 1.0, len(coords))
    (output * weights).sum().backward()
    (output_cached * weights).sum().backward()
    for left, right in zip(network.parameters(), cached.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-11, atol=1e-12)


def test_exact_volume_on_nonuniform_elements_and_implicit_gradient():
    logits = torch.tensor([-0.8, 0.3, 1.1, -0.2, 0.5], requires_grad=True)
    volumes = torch.tensor([0.2, 0.7, 1.1, 2.0, 0.4])
    target = 0.43 * volumes.sum()
    density, _ = volume_project(logits, volumes, target, beta=2.0, rho_min=0.001)
    assert float((density @ volumes).detach()) == pytest.approx(float(target), abs=1e-12)
    derivative = torch.autograd.grad(density @ volumes, logits, retain_graph=True)[0]
    torch.testing.assert_close(derivative, torch.zeros_like(derivative), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(
        lambda z: volume_project(z, volumes, target, beta=2.0, rho_min=0.001)[0],
        (logits,),
        eps=1e-5,
        atol=1e-7,
        rtol=1e-5,
    )


def test_filtered_volume_map_on_nonuniform_cells():
    from hgto.linear2d.design import volume_density

    # Unequal row-normalized neighborhoods make the transpose essential.
    A = torch.tensor(
        [[0.8, 0.2, 0.0, 0.0], [0.1, 0.6, 0.3, 0.0], [0.0, 0.2, 0.7, 0.1], [0.0, 0.0, 0.4, 0.6]]
    ).to_sparse()
    measures = torch.tensor([0.5, 0.8, 1.4, 2.0])
    logits = torch.tensor([-0.7, 0.3, 1.1, -0.2], requires_grad=True)
    density = volume_density(logits, 0.43, A, beta=4.0, volumes=measures)
    assert float((density @ measures / measures.sum()).detach()) == pytest.approx(0.43, abs=1e-12)
    derivative = torch.autograd.grad(density @ measures, logits)[0]
    torch.testing.assert_close(derivative, torch.zeros_like(derivative), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(
        lambda z: volume_density(z, 0.43, A, beta=4.0, volumes=measures),
        (logits,),
        eps=1e-5,
        atol=1e-7,
        rtol=1e-5,
    )
