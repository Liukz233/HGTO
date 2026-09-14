"""Coupled graph-density optimization with finite-deformation and plastic states.

NH uses a two-dimensional compressible energy with small-strain
plane-stress-matched constants. J2 uses small-strain plane-strain plasticity.
NH supports terminal external work or twice the complementary work.
Plastic design uses peak-load external work; unloading is a separate evaluation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import copy, csv, json, time, traceback
from hgto._runtime import preserve_default_dtype
import numpy as np
import torch
from hgto.fem.mesh.q4 import structured_q4, Q4Mesh
from hgto.fem import MechanicsOperator
from hgto.fem.physics.neohookean import solve_nh_state
from hgto.fem.physics.neohookean.adjoint import (
    wang2014_compliance_sensitivity,
    wang2014_potential_sensitivity,
)
from hgto.fem.physics.neohookean.constitutive import deformation_gradient
from hgto.fem.physics.j2.solver import solve_j2_history
from hgto.fem.physics.j2.adjoint import transient_compliance_sensitivity
from hgto.fem.solvers.tangent import assemble_element_matrices, solve_sparse_tangent
from hgto.fem.kernels import gauss_strains
from hgto.fem.mesh.dual_graph import build_element_dual_graph
from hgto.topopt.parameterize.chebnet import ChebNetDensity
from hgto.topopt.parameterize.features.static import unit_bbox_centroids
from hgto.topopt.pipeline.filter import DensityFilter
from hgto.linear2d.design import volume_density
from hgto.stopping import PhysicalStopping, StopConfig


@dataclass
class CaseSetup:
    mesh: Q4Mesh
    fixed_dofs: np.ndarray
    f: np.ndarray
    passive_solid: np.ndarray
    passive_void: np.ndarray
    volume_fraction: float
    name: str
    raw: dict


def graph_network(setup, seed, architecture, device="cpu"):
    coords = unit_bbox_centroids(setup.mesh).to(device)
    edges = torch.as_tensor(
        build_element_dual_graph(setup.mesh)["edge_index"], dtype=torch.long, device=device
    )
    network = ChebNetDensity(seed=seed, dim=2, **architecture).to(device)
    return network, {"coords": coords, "edge_index": edges}


def density_field(logits, filt, volumes, target, beta, rho_min):
    return volume_density(logits, float(target / volumes.sum()), filt.F, beta, rho_min, volumes)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def case(name, refine=1, nh_transition_beta=None):
    if name not in (
        "bridge",
        "large_cantilever",
        "plastic_lug",
        "cantilever_nh",
        "connection_j2",
        "bridge_nh",
    ):
        raise ValueError(f"Unknown nonlinear example: {name}")
    if not isinstance(refine, int) or refine < 1:
        raise ValueError("refine must be a positive integer")
    if name == "bridge_nh":
        nx, ny, L, H, vf = 96 * refine, 32 * refine, 24.0, 8.0, 0.4
    elif name == "bridge":
        nx, ny, L, H, vf = 48 * refine, 16 * refine, 24.0, 8.0, 0.4
    elif name == "cantilever_nh":
        nx, ny, L, H, vf = 96 * refine, 24 * refine, 24.0, 6.0, 0.45
    else:
        nx, ny, L, H, vf = (
            48 * refine,
            24 * refine,
            24.0,
            12.0,
            0.45 if name == "large_cantilever" else 0.4,
        )
    m = structured_q4(nx, ny)
    m.coords[:, 0] *= L / nx
    m.coords[:, 1] *= H / ny
    hole = None
    if name in ("plastic_lug", "connection_j2"):
        cent = m.element_centroids()
        hole = {"center": [10.5, 6.5], "radius": 2.0}
        keep = np.linalg.norm(cent - np.array(hole["center"]), axis=1) >= hole["radius"]
        ec = m.econn[keep]
        used = np.unique(ec)
        ids = np.full(m.n_nodes, -1)
        ids[used] = np.arange(len(used))
        m = Q4Mesh(nx, ny, m.coords[used], ids[ec].astype(np.int64))
    left = np.flatnonzero(np.isclose(m.coords[:, 0], 0))
    if name in ("bridge", "bridge_nh"):
        right = np.flatnonzero(np.isclose(m.coords[:, 0], L))
        mount = np.r_[left, right]
        ports = np.flatnonzero(
            np.isclose(m.coords[:, 1], H) & (abs(m.coords[:, 0] - L / 2) <= 1.5 + 1e-10)
        )
        direction = np.array([0.0, -1.0])
    else:
        mount = left
        port_center = H / 4 if name == "connection_j2" else H / 2
        port_halfwidth = 0.75 if name == "cantilever_nh" else 1.5
        ports = np.flatnonzero(
            np.isclose(m.coords[:, 0], L)
            & (abs(m.coords[:, 1] - port_center) <= port_halfwidth + 1e-10)
        )
        direction = (
            np.array([0.0, -1.0])
            if name in ("large_cantilever", "cantilever_nh", "connection_j2", "bridge_nh")
            else np.array([1.0, -0.2]) / np.sqrt(1.04)
        )
    fixed = np.sort(np.r_[2 * mount, 2 * mount + 1])
    force = np.zeros((m.n_nodes, 2))
    force[ports] = direction / len(ports)
    spec = {
        "case": name,
        "nx": nx,
        "ny": ny,
        "L": L,
        "H": H,
        "volume_fraction": vf,
        "hole": hole,
        "port_nodes": ports.tolist(),
        "port_coordinates": m.coords[ports].tolist(),
        "unit_direction": direction.tolist(),
        "fixed_dofs": fixed.tolist(),
        "mesh_elements": m.n_elements,
        "mesh_nodes": m.n_nodes,
    }
    if name in ("cantilever_nh", "bridge_nh"):
        transition = 500.0 if nh_transition_beta is None else float(nh_transition_beta)
        if not np.isfinite(transition) or transition <= 0:
            raise ValueError("NH transition sharpness must be finite and positive")
        spec["nh_interpolation"] = {
            "gamma_mode": "simp_heaviside",
            "beta0": transition,
            "eta0": 0.01,
        }
    elif nh_transition_beta is not None:
        raise ValueError("NH transition override requires a supported Neo-Hookean case")
    setup = CaseSetup(
        m, fixed, force, np.zeros(m.n_elements, bool), np.zeros(m.n_elements, bool), vf, name, spec
    )
    return setup, spec


def operator(setup, p=3.0, tangent_backend="scipy", device="cpu"):
    if tangent_backend not in ("scipy", "pypardiso", "cudss"):
        raise ValueError("Unknown tangent backend")
    op = MechanicsOperator(
        setup.mesh,
        setup.fixed_dofs,
        {"E0": 1.0, "Emin": 1e-6, "nu": 0.3, "p": p},
        device=device,
        use_fused=False,
    )
    if (op.device.type == "cuda") != (tangent_backend == "cudss"):
        raise ValueError("CUDA states require cuDSS; CPU states require a CPU backend")
    op.sparse_tangent_backend = tangent_backend
    op.nh_interpolation = dict(setup.raw.get("nh_interpolation", {}))
    return op


def solve(
    op,
    rho,
    force,
    physics,
    gradient=False,
    load_steps=8,
    yield_stress=0.0015,
    hardening=0.1,
    objective="terminal_work",
):
    from hgto.fem.solvers.cudss_tangent import solve_cuda_tangent

    linear_solver = solve_cuda_tangent if op.device.type == "cuda" else solve_sparse_tangent
    if rho.device != op.device or force.device != op.device:
        raise ValueError("Density, force and mechanics must share a device")
    if physics not in ("linear", "nh", "j2", "elastic_j2"):
        raise ValueError(f"Unknown physical model: {physics}")
    if load_steps < 1:
        raise ValueError("load_steps must be positive")
    if objective not in ("terminal_work", "complementary_work"):
        raise ValueError(f"Unknown objective: {objective}")
    if objective == "complementary_work" and physics not in ("linear", "nh"):
        raise ValueError("Complementary work requires conservative elasticity")
    if physics == "linear":
        if op.device.type != "cpu":
            raise ValueError(
                "Legacy linear comparison uses the independent CPU evaluator; use elastic_j2 for the GPU plastic control"
            )
        E = op.Emin + (op.E0 - op.Emin) * rho.pow(op.p)
        K = assemble_element_matrices(op, op.Ke0 * E[:, None, None])
        from scipy.sparse.linalg import spsolve

        free = op.free_dof_mask
        u = force.new_zeros(op.n_dof)
        u[free] = torch.as_tensor(
            spsolve(K, force.reshape(-1)[free].numpy()), dtype=op.dtype, device=op.device
        )
        u = u.reshape(1, op.n_nodes, 2)
        C = float((u[0] * force).sum())
        ue = u[:, op.econn, :].reshape(1, op.n_elements, 8)
        g = (
            -op.p
            * (op.E0 - op.Emin)
            * rho.pow(op.p - 1)
            * torch.einsum("lei,eij,lej->e", ue, op.Ke0, ue)
            if gradient
            else None
        )
        rhs = force.reshape(-1)[free].numpy()
        solution = u.reshape(-1)[free].numpy()
        residual = float(
            np.linalg.norm(K @ solution - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )
        return C, g, {"u": u, "residual": residual, "history_u": u.clone()}
    if physics == "nh":
        interpolation = getattr(op, "nh_interpolation", {})
        st = solve_nh_state(
            op,
            rho,
            force[None],
            n_ramp=load_steps,
            rtol=1e-8,
            pcg_rtol=1e-10,
            max_newton=100,
            max_backtracks=35,
            record_ramp_history=True,
            linear_solver=linear_solver,
            **interpolation,
        )
        if objective == "complementary_work":
            value = -2 * float(st.potential)
            g = (
                -2 * wang2014_potential_sensitivity(op, rho, st, **interpolation)
                if gradient
                else None
            )
        else:
            value = float(st.compliance)
            g = (
                wang2014_compliance_sensitivity(
                    op,
                    rho,
                    st,
                    force[None],
                    rtol=1e-9,
                    linear_solver=linear_solver,
                    **interpolation,
                )
                if gradient
                else None
            )
        F = deformation_gradient(st.u, op.econn, op.dN_dx)
        det = torch.linalg.det(F)
        solid = rho > 0.5
        solid = solid if solid.any() else rho >= rho.median()
        d = {
            "u": st.u,
            "history_u": st.ramp_u[0],
            "residual": float(st.ramp_residual_rel.max()),
            "newton": int(st.ramp_newton_iterations.sum()),
            "line_search_backtracks": int(st.backtracks.sum()),
            "terminal_work": float(st.compliance),
            "potential": float(st.potential),
            "min_det_F_material": float(det[0, solid].min()),
            "F": F,
        }
        return value, g, d
    sy = 1e6 if physics == "elastic_j2" else yield_stress
    factors = (
        torch.linspace(1 / load_steps, 1.0, load_steps, dtype=op.dtype, device=op.device)
        if physics == "j2"
        else force.new_ones(1)
    )
    forces = factors[:, None, None] * force[None]
    if gradient:
        C, g, st = transient_compliance_sensitivity(
            op, rho, forces, sy, hardening, rtol=1e-9, pcg_rtol=1e-10, linear_solver=linear_solver
        )
    else:
        st = solve_j2_history(
            op,
            rho,
            forces,
            sy,
            hardening,
            rtol=1e-9,
            pcg_rtol=1e-10,
            max_newton=70,
            linear_solver=linear_solver,
        )
        C = float((st["u"][0] * force).sum())
        g = None
    alpha = st["history"][-1]["alpha"]
    eps = gauss_strains(st["u"], op.econn, op.dN_dx)
    norm = torch.sqrt(eps[..., 0] ** 2 + eps[..., 1] ** 2 + 0.5 * eps[..., 2] ** 2)
    solid = rho > 0.5
    solid = solid if solid.any() else rho >= rho.median()
    d = {
        "u": st["u"],
        "history_u": torch.stack([h["u"][0] for h in st["history"]]),
        "history_alpha": torch.stack([h["alpha"][0] for h in st["history"]]),
        "history_plastic_strain": torch.stack([h["plastic_strain"][0] for h in st["history"]]),
        "residual": max(h["residual_rel"] for h in st["history"]),
        "newton": sum(h["newton_iters"] for h in st["history"]),
        "plastic_fraction_material": float((alpha[0, solid] > 1e-8).double().mean()),
        "alpha_max_material": float(alpha[0, solid].max()),
        "strain_max_material": float(norm[0, solid].max()),
        "strain_p99_material": float(torch.quantile(norm[0, solid].reshape(-1), 0.99)),
    }
    return float(C), g, d


def diagnostics(d, spec):
    u = d["u"][0]
    ports = spec["port_nodes"]
    direction = u.new_tensor(spec["unit_direction"])
    delta = float((u[ports] * direction).sum(1).mean())
    return {k: v for k, v in d.items() if not isinstance(v, torch.Tensor)} | {
        "port_displacement": delta,
        "port_displacement_over_L": delta / spec["L"],
        "umax_over_L": float(torch.linalg.vector_norm(u, dim=1).max()) / spec["L"],
    }


@preserve_default_dtype
def run(
    name,
    physics,
    load,
    output,
    steps=180,
    seed=0,
    yield_stress=0.0015,
    hardening=0.1,
    beta_final=8.0,
    backtracking=False,
    objective="terminal_work",
    uniform_initial=False,
    architecture=None,
    device="cuda:0",
    max_updates=1000,
    early_stopping=True,
    stop_config=None,
    resume_from=None,
    learning_rate=0.003,
    final_learning_rate=0.0003,
    tail_half_life=40.0,
    minimum_learning_rate=1e-5,
    nh_transition_beta=None,
    tangent_backend=None,
    state_device=None,
):
    if steps < 1 or load <= 0 or beta_final < 1:
        raise ValueError("Use positive steps/load and beta_final >= 1")
    torch.set_default_dtype(torch.float64)
    state_device = state_device or device
    tangent_backend = tangent_backend or (
        "cudss" if torch.device(state_device).type == "cuda" else "scipy"
    )
    setup, spec = case(name, nh_transition_beta=nh_transition_beta)
    op = operator(setup, tangent_backend=tangent_backend, device=state_device)
    out = Path(output)
    if (out / "record.json").exists():
        print("Already done", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    volumes = torch.tensor(setup.mesh.element_volumes(), device=device)
    filt = DensityFilter(setup.mesh, 1.0, device=device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    architecture = dict(
        architecture or {"n_freq": 32, "hidden_dim": 64, "sigma": 1.0, "paper_K": 1}
    )
    net, inputs = graph_network(setup, seed, architecture, device=device)
    if uniform_initial:
        with torch.no_grad():
            for layer in net.conv2.lins:
                layer.weight.zero_()
            if net.conv2.bias is not None:
                net.conv2.bias.zero_()
    mirror = (
        torch.arange(setup.mesh.n_elements, device=device)
        .reshape(spec["ny"], spec["nx"])
        .flip(1)
        .reshape(-1)
        if name == "bridge_nh"
        else None
    )

    def material_logits():
        values = net(inputs["coords"], inputs["edge_index"])
        return values if mirror is None else 0.5 * (values + values[mirror])

    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    force = torch.tensor(setup.f, device=op.device) * load
    load_steps = 12 if physics in ["j2", "nh"] else 1
    protocol = spec | {
        "physics": physics,
        "objective": objective,
        "force_resultant": load,
        "seed": seed,
        "max_updates": max_updates,
        "continuation_reference_updates": steps,
        "architecture": architecture,
        "filter_radius_physical": 1.0,
        "rho_min": 0.001,
        "p_schedule": [1.0, 3.0],
        "beta_schedule": [1.0, beta_final],
        "optimizer": "Adam with cosine then bounded exponential learning-rate decay, clip .5",
        "learning_rate": learning_rate,
        "final_learning_rate": final_learning_rate,
        "tail_half_life": tail_half_life,
        "minimum_learning_rate": minimum_learning_rate,
        "load_steps": load_steps,
        "yield_stress": yield_stress,
        "hardening": hardening,
        "candidate_backtracking": bool(backtracking),
        "initial_density": "uniform" if uniform_initial else "random_network",
        "network_device": str(next(net.parameters()).device),
        "state_device": str(op.device),
        "state_backend": "CUDA cuDSS general sparse LU"
        if op.device.type == "cuda"
        else "CPU sparse tangent solve",
        "tangent_backend": tangent_backend,
        "gpu_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
    }
    stop = PhysicalStopping(stop_config or StopConfig())
    protocol.update(
        state_precision="float64",
        stopping=asdict(stop.config),
        early_stopping=early_stopping,
        density_symmetry_axes=[0] if mirror is not None else [],
    )
    dump(out / "protocol.json", protocol)
    np.save(out / "coords.npy", setup.mesh.coords)
    np.save(out / "econn.npy", setup.mesh.econn)
    np.save(out / "unit_force.npy", setup.f)
    rows = []
    snaprho = []
    snapiter = []
    start = time.perf_counter()
    C0 = None
    previous = None
    rejections = []
    start_update = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint["network_state"])
        opt.load_state_dict(checkpoint["optimizer_state"])
        C0 = checkpoint["C0"]
        start_update = checkpoint["update"]
        if early_stopping:
            raise ValueError("Continuation checks must disable stopping")
    if max_updates <= start_update or max_updates < int(0.8 * steps):
        raise ValueError("Maximum updates must exceed continuation/resume point")
    termination = "max_updates"
    try:
        for it in range(start_update, max_updates + 1):
            fraction = min(1.0, it / max(1, int(0.7 * steps)))
            p = 1 + 2 * fraction
            beta = beta_final ** min(1.0, it / max(1, int(0.8 * steps)))
            op.p.fill_(p)
            opt.zero_grad()
            z = material_logits()
            rho = density_field(
                z, filt, volumes, spec["volume_fraction"] * volumes.sum(), beta, 0.001
            )
            t = time.perf_counter()
            reductions = 0
            while True:
                try:
                    C, g, d = solve(
                        op,
                        rho.detach().to(op.device),
                        force,
                        physics,
                        it < max_updates,
                        load_steps,
                        yield_stress,
                        hardening,
                        objective=objective,
                    )
                    break
                except Exception as exc:
                    np.save(out / "last_rejected_density.npy", rho.detach().cpu().numpy())
                    rejections.append(
                        {
                            "iteration": it,
                            "reduction": reductions,
                            "error": str(exc),
                            "elapsed": time.perf_counter() - start,
                        }
                    )
                    dump(out / "candidate_rejections.json", rejections)
                    if not backtracking or previous is None or reductions >= 10:
                        raise
                    reductions += 1
                    net.load_state_dict(previous["net"])
                    opt.load_state_dict(previous["optimizer"])
                    for param, grad in zip(net.parameters(), previous["gradients"]):
                        param.grad = grad.clone()
                    for group in opt.param_groups:
                        group["lr"] = previous["lr"] * (0.5**reductions)
                    opt.step()
                    z = material_logits()
                    rho = density_field(
                        z, filt, volumes, spec["volume_fraction"] * volumes.sum(), beta, 0.001
                    )
            if C0 is None:
                C0 = C
            row = {
                "iteration": it,
                "candidate_step_reductions": reductions,
                "C": C,
                "p": p,
                "beta": beta,
                "volume": float((rho.detach() * volumes).sum() / volumes.sum()),
                "Mnd": float(400 * (rho.detach() * (1 - rho.detach())).mean()),
                "state_seconds": time.perf_counter() - t,
                "elapsed": time.perf_counter() - start,
                **diagnostics(d, spec),
            }
            row.update(
                stop.observe(
                    C,
                    rho.detach().cpu().numpy(),
                    parameters=(p, beta),
                    continuation_complete=p == 3.0 and beta == beta_final,
                    volume_error=row["volume"] - spec["volume_fraction"],
                    residual=d["residual"],
                    accepted=reductions == 0,
                )
            )
            rows.append(row)
            np.save(out / "last_accepted_density.npy", rho.detach().cpu().numpy())
            finish = (early_stopping and row["converged"]) or it == max_updates
            with (out / "history.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            if it % 20 == 0 or finish:
                snaprho.append(rho.detach().cpu().numpy().copy())
                snapiter.append(it)
                np.savez_compressed(out / "snapshots.npz", rho=np.stack(snaprho), steps=snapiter)
                print(
                    json.dumps({"case": name, "physics": physics, "seed": seed, **row}), flush=True
                )
            if finish:
                termination = "converged" if early_stopping and row["converged"] else "max_updates"
                break
            rho.backward(g.to(device) / C0)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            lr = float(
                final_learning_rate
                + (learning_rate - final_learning_rate)
                * 0.5
                * (1 + np.cos(np.pi * min(1.0, it / max(1, steps - 1))))
            )
            if it >= steps:
                lr = max(
                    minimum_learning_rate,
                    final_learning_rate * 2.0 ** (-(it - steps + 1) / tail_half_life),
                )
            for group in opt.param_groups:
                group["lr"] = lr
            if backtracking:
                previous = {
                    "net": copy.deepcopy(net.state_dict()),
                    "optimizer": copy.deepcopy(opt.state_dict()),
                    "gradients": [param.grad.clone() for param in net.parameters()],
                    "lr": lr,
                }
            opt.step()
        torch.save(
            dict(
                network_state=net.state_dict(), optimizer_state=opt.state_dict(), C0=C0, update=it
            ),
            out / "resume.pt",
        )
        if op.device.type == "cuda":
            protocol.update(
                gpu_tangent_solves=op._cudss_tangent.solves,
                cudss_version="0.8.0",
                cudss_hybrid_execute=False,
                cudss_hybrid_memory=False,
                state_precision="float64",
            )
        protocol.update(
            design_updates=it, termination=termination, converged=termination == "converged"
        )
        dump(out / "protocol.json", protocol)
        np.save(out / "rho.npy", rho.detach().cpu().numpy())
        np.savez_compressed(
            out / "states.npz",
            **{k: v.detach().cpu().numpy() for k, v in d.items() if isinstance(v, torch.Tensor)},
        )
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "architecture": architecture,
                "seed": seed,
            },
            out / "network.pt",
        )
        dump(
            out / "record.json",
            {
                "protocol": protocol,
                "final": row,
                "wall_s": time.perf_counter() - start,
                "termination": termination,
                "converged": termination == "converged",
                "design_updates": it,
            },
        )
    except Exception as e:
        np.save(out / "failed_density.npy", rho.detach().cpu().numpy())
        dump(
            out / "failure.json",
            {
                "message": str(e),
                "traceback": traceback.format_exc(),
                "completed_states": len(rows),
                "protocol": protocol,
            },
        )
        raise
