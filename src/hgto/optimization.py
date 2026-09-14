"""Graph-network Adam with physical stopping after material/projection continuation."""

from dataclasses import dataclass, asdict
import time
import numpy as np
import torch
from hgto.linear2d.design import volume_density
from hgto.stopping import PhysicalStopping, StopConfig
from hgto.linear2d.filters import DensityMap
from hgto.topopt.parameterize.chebnet import ChebNetDensity
from hgto.topopt.parameterize.features.static import unit_bbox_centroids
from hgto.fem.mesh.dual_graph import build_element_dual_graph


@dataclass
class GraphOptimizerConfig:
    n_freq: int = 64
    sigma: float = 2.0
    hidden_dim: int = 64
    seed: int = 0
    stage_steps: tuple = (50, 50, 50, 200)
    betas: tuple = (1.0, 2.0, 4.0, 8.0)
    penalties: tuple = (1.0, 2.0, 3.0, 3.0)
    learning_rate: float = 0.01
    final_learning_rate: float = 0.001
    rho_min: float = 0.001
    snapshot_interval: int = 25
    uniform_initialization: bool = True
    symmetry_axes: tuple = ()
    device: str = "cuda:0"
    max_updates: int = 1000
    early_stopping: bool = True
    stop_window: int = 10
    stop_objective_tolerance: float = 1e-3
    stop_density_tolerance: float = 5e-3
    stop_patience: int = 5


def optimize_graph(
    mesh,
    physics,
    volume,
    radius,
    config=GraphOptimizerConfig(),
    centroids=None,
    volumes=None,
    started_at=None,
    resume_from=None,
):
    """Optimize graph weights, accepting any mesh and a (C, dC/drho) oracle."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    def now():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    t0 = now() if started_at is None else started_at
    coords = unit_bbox_centroids(mesh).to(device=device, dtype=torch.float64)
    edges = torch.as_tensor(
        build_element_dual_graph(mesh)["edge_index"], dtype=torch.long, device=device
    )
    net = ChebNetDensity(
        config.n_freq,
        config.sigma,
        config.seed,
        hidden_dim=config.hidden_dim,
        paper_K=1,
        dim=coords.shape[1],
    ).to(device=device, dtype=torch.float64)
    if config.uniform_initialization:
        with torch.no_grad():
            for lin in net.conv2.lins:
                lin.weight.zero_()
            net.conv2.bias.zero_()
    net.prepare_static(coords, edges)
    if centroids is None:
        centroids = mesh.element_centroids()
    symmetries = []
    if config.symmetry_axes:
        from scipy.spatial import cKDTree

        points = np.asarray(centroids)
        tree = cKDTree(points)
        for axis in config.symmetry_axes:
            reflected = points.copy()
            reflected[:, axis] = points[:, axis].min() + points[:, axis].max() - points[:, axis]
            distance, indices = tree.query(reflected)
            if distance.max() > 1e-8 or len(np.unique(indices)) != len(points):
                raise ValueError("Requested symmetry does not map mesh cells onto themselves")
            symmetries.append(torch.as_tensor(indices, device=device, dtype=torch.long))
    mapping = DensityMap(np.asarray(centroids), radius).A
    if volumes is not None:
        # DensityMap is row normalized for equal cells.  Reweight its columns
        # by true volumes, then renormalize for nonuniform cells.
        from scipy import sparse

        mapping = mapping @ sparse.diags(volumes)
        mapping = sparse.diags(1 / np.asarray(mapping.sum(1)).ravel()) @ mapping
    coo = mapping.tocoo()
    A = torch.sparse_coo_tensor(
        torch.as_tensor(np.stack([coo.row, coo.col]), device=device),
        torch.as_tensor(coo.data, device=device, dtype=torch.float64),
        coo.shape,
    ).coalesce()
    weights = (
        None if volumes is None else torch.as_tensor(volumes, device=device, dtype=torch.float64)
    )
    history = []
    snapshots = []
    C0 = None
    previous = None
    accepted = 0
    if len(config.betas) != len(config.stage_steps) or len(config.betas) != len(config.penalties):
        raise ValueError("Beta, penalty and stage-step schedules must have equal length")
    if config.penalties[-1] != 3.0:
        raise ValueError("The reported final physical model uses p=3")
    penalty = 3.0
    setup_s = now() - t0

    def evaluate(beta, phase, save=False):
        nonlocal C0, previous, accepted
        logits = net(coords, edges)
        for indices in symmetries:
            logits = 0.5 * (logits + logits[indices])
        rho = volume_density(logits, volume, A, beta, config.rho_min, weights)
        C, g = physics.evaluate(rho)
        C = float(C)
        if C0 is None:
            C0 = C
        g = torch.as_tensor(g, device=device, dtype=rho.dtype)
        elapsed = now() - t0
        history.append(
            dict(
                iteration=len(history),
                C=C,
                beta=float(beta),
                penalty=float(penalty),
                elapsed_s=elapsed,
                phase=phase,
                volume=float(rho.detach().mean())
                if weights is None
                else float((rho.detach() * weights).sum() / weights.sum()),
                accepted_step=accepted,
            )
        )
        if save:
            change = float((rho.detach() - previous).abs().max()) if previous is not None else 1.0
            history[-1]["max_density_change"] = change
            if accepted % config.snapshot_interval == 0:
                snapshots.append((len(history) - 1, elapsed, rho.detach().cpu().numpy().copy()))
            previous = rho.detach().clone()
            accepted += 1
        # Surrogate value C has exactly the analytically supplied derivative.
        loss = (rho * g).sum() / C0
        loss = loss - loss.detach() + rho.new_tensor(C / C0)
        return rho, loss, C

    opt = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    start_update = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint["network_state"])
        opt.load_state_dict(checkpoint["optimizer_state"])
        C0 = checkpoint["C0"]
        start_update = checkpoint["update"]
        if config.early_stopping:
            raise ValueError("Continuation checks must disable stopping")
    if config.max_updates <= start_update or config.max_updates < sum(config.stage_steps[:-1]):
        raise ValueError("Maximum updates must exceed the completed continuation/resume point")
    stop = PhysicalStopping(
        StopConfig(
            window=config.stop_window,
            objective_tolerance=config.stop_objective_tolerance,
            density_tolerance=config.stop_density_tolerance,
            patience=config.stop_patience,
        )
    )
    offsets = np.r_[0, np.cumsum(config.stage_steps[:-1])]
    stop_reason = "max_updates"
    last_check = None
    for update in range(start_update, config.max_updates + 1):
        stage = min(int(np.searchsorted(offsets, update, side="right") - 1), len(config.betas) - 1)
        beta = config.betas[stage]
        penalty = config.penalties[stage]
        if hasattr(physics, "set_penalty"):
            physics.set_penalty(penalty)
        elif hasattr(physics, "operator"):
            if torch.is_tensor(physics.operator.p):
                physics.operator.p.fill_(float(penalty))
            else:
                physics.operator.p = float(penalty)
        elif penalty != 3.0:
            raise ValueError("Physics backend cannot change the SIMP exponent")
        opt.zero_grad(set_to_none=True)
        rho, loss, C = evaluate(beta, "adam", True)
        history[-1]["update"] = update
        last_check = stop.observe(
            C,
            rho.detach().cpu().numpy(),
            parameters=(penalty, beta),
            continuation_complete=stage == len(config.betas) - 1,
            volume_error=history[-1]["volume"] - volume,
            residual=physics.max_residual,
        )
        history[-1].update(last_check)
        if (config.early_stopping and last_check["converged"]) or update == config.max_updates:
            stop_reason = (
                "converged" if config.early_stopping and last_check["converged"] else "max_updates"
            )
            history[-1]["phase"] = "final"
            break
        fraction = min(1.0, (update - offsets[stage]) / max(config.stage_steps[stage] - 1, 1))
        opt.param_groups[0]["lr"] = float(
            config.final_learning_rate
            + 0.5
            * (config.learning_rate - config.final_learning_rate)
            * (1 + np.cos(np.pi * fraction))
        )
        loss.backward()
        opt.step()
        if (update + 1) % config.snapshot_interval == 0:
            print(
                f"Adam update={update + 1} p={penalty:g} beta={beta:g} C={C:.8g} stable={stop.stable}/{config.stop_patience} elapsed={now() - t0:.1f}s",
                flush=True,
            )
    wall = now() - t0
    snapshots.append((len(history) - 1, wall, rho.detach().cpu().numpy().copy()))
    return dict(
        rho=rho.detach().cpu().numpy(),
        history=history,
        snapshots=snapshots,
        resume_checkpoint=dict(
            network_state=net.state_dict(),
            optimizer_state=opt.state_dict(),
            C0=C0,
            update=update,
            config=asdict(config),
        ),
        network_state={k: v.detach().cpu() for k, v in net.state_dict().items()},
        summary=dict(
            method="HGTO",
            C_raw=C,
            volume=history[-1]["volume"],
            wall_s=wall,
            setup_s=setup_s,
            state_evaluations=len(history),
            config=asdict(config),
            trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
            optimizer="Adam",
            design_updates=update,
            max_updates=config.max_updates,
            converged=stop_reason == "converged",
            stopping=asdict(stop.config),
            stopping_final=last_check,
            network_device=str(next(net.parameters()).device),
            density_device=str(rho.device),
            termination=stop_reason,
            timing_scope="caller start through final graph-state evaluation; independent validation is added by experiment driver",
        ),
    )
