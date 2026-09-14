"""Graph density representation and its exact-volume implicit map."""

import torch


class _VolumeDensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, A, beta, rho_min, volume, volumes):
        weights = torch.ones_like(z) if volumes is None else volumes.to(z)
        if weights.shape != z.shape or not bool(torch.all(weights > 0)):
            raise ValueError("volumes must be positive and match logits")
        weight_sum = weights.sum()
        lower = float(z.min()) - 48.0
        upper = float(z.max()) + 48.0
        shift = 0.5 * (lower + upper)
        den = 2 * torch.tanh(z.new_tensor(beta / 2)) if beta else None
        transpose_matrix = None if A is None else A.transpose(0, 1).coalesce()

        def apply(v, transpose=False):
            if A is None:
                return v
            matrix = transpose_matrix if transpose else A
            return torch.sparse.mm(matrix, v[:, None])[:, 0]

        for _ in range(100):
            x = torch.sigmoid(z - shift)
            dx = x * (1 - x)
            bar = apply(x)
            if beta:
                t = torch.tanh(beta * (bar - 0.5))
                h = (den / 2 + t) / den
                dh = beta * (1 - t * t) / den
            else:
                h = bar
                dh = torch.ones_like(bar)
            rho = rho_min + (1 - rho_min) * h
            dh = dh * (1 - rho_min)
            q = dx * apply(dh * weights, True)
            error = float((rho * weights).sum() / weight_sum) - volume
            slope = float(q.sum() / weight_sum)
            if abs(error) < 2e-14:
                break
            if error > 0:
                lower = shift
            else:
                upper = shift
            proposal = shift + error / slope if slope > 1e-15 else float("nan")
            shift = proposal if lower < proposal < upper else 0.5 * (lower + upper)
        else:
            raise RuntimeError("volume root failed")
        assert float(q.sum() / weight_sum) > 1e-14, "saturated volume map"
        ctx.save_for_backward(dx, dh, q)
        ctx.transpose_matrix = transpose_matrix
        return rho

    @staticmethod
    def backward(ctx, g):
        dx, dh, q = ctx.saved_tensors
        pull = (
            dh * g
            if ctx.transpose_matrix is None
            else torch.sparse.mm(ctx.transpose_matrix, (dh * g)[:, None])[:, 0]
        )
        base = dx * pull
        return base - q * base.sum() / q.sum(), None, None, None, None, None


def volume_density(logits, volume, A=None, beta=0.0, rho_min=0.001, volumes=None):
    """Physical density with an implicitly differentiated volume constraint.

    ``A`` is an optional row-normalized sparse spatial filter. ``volumes``
    allows unequal cell measures; omitted measures are all one. The shift
    root includes the complete sigmoid/filter/projection chain.
    """
    return _VolumeDensity.apply(logits, A, float(beta), float(rho_min), float(volume), volumes)


def build_graph_design(problem, mesh, device, seed=0, n_freq=32):
    from hgto.fem.mesh.dual_graph import build_element_dual_graph
    from hgto.topopt.parameterize.chebnet import ChebNetDensity
    from hgto.topopt.parameterize.features.static import unit_bbox_centroids

    torch.manual_seed(seed)
    coords = unit_bbox_centroids(mesh).to(device)
    edges = torch.as_tensor(
        build_element_dual_graph(mesh)["edge_index"], dtype=torch.long, device=device
    )
    network = ChebNetDensity(n_freq, 1.0, seed, hidden_dim=64, paper_K=1, dim=2).to(device)
    network.prepare_static(coords, edges)
    return network, coords, edges
