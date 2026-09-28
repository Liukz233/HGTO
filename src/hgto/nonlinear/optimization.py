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


NH_CASES = ("cantilever_nh", "bridge_nh", "lbracket_nh")


def validate_nh_solver(nh_solver):
    """Validate optional Newton and adaptive load-bisection budgets."""
    solver = dict(nh_solver)
    unknown = set(solver) - {"max_newton", "adaptive_budgets"}
    if unknown:
        raise ValueError(f"Unknown nh_solver options: {sorted(unknown)}")
    if "max_newton" in solver:
        if type(solver["max_newton"]) is not int or solver["max_newton"] < 1:
            raise ValueError("max_newton must be a positive integer")
    if "adaptive_budgets" in solver:
        budgets = dict(solver["adaptive_budgets"])
        allowed = {"max_subdivisions", "max_failed_attempts", "max_extra_increments"}
        if set(budgets) - allowed:
            raise ValueError(f"Unknown adaptive budgets: {sorted(set(budgets) - allowed)}")
        if any(type(v) is not int or v < 0 for v in budgets.values()):
            raise ValueError("Adaptive budgets must be nonnegative integers")
        solver["adaptive_budgets"] = budgets
    return solver


def case(name, refine=1, nh_transition_beta=None, nh_solver=None):
    """Return the mesh, supports and unit load of a named nonlinear example.

    ``nh_solver`` optionally sets the Newton iteration cap per load increment
    (``max_newton``) and the adaptive load-bisection budgets
    (``adaptive_budgets``); both are recorded in the returned specification.
    """
    if name not in (
        "bridge",
        "large_cantilever",
        "plastic_lug",
        "connection_j2",
        *NH_CASES,
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
    elif name == "lbracket_nh":
        # 16 x 16 square without its upper-right 8 x 8 quadrant (3,072 cells).
        nx, ny, L, H, vf = 64 * refine, 64 * refine, 16.0, 16.0, 0.4
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
    keep = None
    cent = m.element_centroids()
    if name in ("plastic_lug", "connection_j2"):
        hole = {"center": [10.5, 6.5], "radius": 2.0}
        keep = np.linalg.norm(cent - np.array(hole["center"]), axis=1) >= hole["radius"]
    elif name == "lbracket_nh":
        keep = ~((cent[:, 0] > L / 2) & (cent[:, 1] > H / 2))
    if keep is not None:
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
    elif name == "lbracket_nh":
        # Upper edge of the vertical arm fixed; downward load on the upper 1.5 units
        # of the free end of the horizontal arm.
        mount = np.flatnonzero(np.isclose(m.coords[:, 1], H))
        ports = np.flatnonzero(
            np.isclose(m.coords[:, 0], L) & (m.coords[:, 1] >= H / 2 - 1.5 - 1e-10)
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
        "refine": refine,
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
        "port_weights": (force[ports] @ direction).tolist(),
    }
    if name == "lbracket_nh":
        spec.update(
            removed_region=[L / 2, H / 2, L, H],
            volume_reference="area of the L-shaped finite-element domain",
            mesh_area=float(m.element_volumes().sum()),
        )
    if name in NH_CASES:
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
    if nh_solver is not None:
        if name not in NH_CASES:
            raise ValueError("nh_solver applies to Neo-Hookean cases only")
        spec["nh_solver"] = validate_nh_solver(nh_solver)
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
    # Optional state-solver budgets (recorded in the case spec/protocol).
    solver = setup.raw.get("nh_solver", {})
    if "max_newton" in solver:
        op.nh_max_newton = int(solver["max_newton"])
    if "adaptive_budgets" in solver:
        op.nh_adaptive_budgets = dict(solver["adaptive_budgets"])
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
        nh_solver = solve_nh_state
        if getattr(op, "adaptive_nh_load", False):
            from hgto.fem.physics.neohookean.adaptive import solve_nh_adaptive

            nh_solver = solve_nh_adaptive
        extra = (
            dict(getattr(op, "nh_adaptive_budgets", {})) if nh_solver is not solve_nh_state else {}
        )
        # Physics-field continuation (optional): an elastic state is path independent, so the
        # equilibrium may be continued at the full load from the previous accepted state. Any
        # failure falls back to incremental loading from the undeformed state below.
        u_start = getattr(op, "nh_state_start", None)
        st = None
        if u_start is not None:
            try:
                st = solve_nh_state(
                    op,
                    rho,
                    force[None],
                    u0=u_start,
                    n_ramp=1,
                    rtol=1e-8,
                    pcg_rtol=1e-10,
                    max_newton=int(getattr(op, "nh_max_newton", 100)),
                    max_backtracks=35,
                    record_ramp_history=True,
                    linear_solver=linear_solver,
                    **interpolation,
                )
            except RuntimeError:
                st = None
        state_start = "previous_state" if st is not None else "undeformed"
        if st is None:
            st = nh_solver(
                op,
                rho,
                force[None],
                n_ramp=load_steps,
                rtol=1e-8,
                pcg_rtol=1e-10,
                max_newton=int(getattr(op, "nh_max_newton", 100)),
                max_backtracks=35,
                record_ramp_history=True,
                linear_solver=linear_solver,
                **extra,
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
            "load_increments": getattr(st, "adaptive_successful_increments", load_steps),
            "load_cutback_failures": getattr(st, "adaptive_failed_attempts", 0),
            "state_start": state_start,
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
    weights = u.new_tensor(spec.get("port_weights", [1 / len(ports)] * len(ports)))
    delta = float(((u[ports] * direction).sum(1) * weights).sum())
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
    setup=None,
    spec=None,
    acceptance="fixed_parameter_descent",
    acceptance_relative_tolerance=1e-8,
    max_candidate_backtracks=20,
    stationarity_tolerance=None,
    min_fixed_updates=50,
    load_steps=None,
    adaptive_load=False,
    max_wall_seconds=None,
    beta_ramp_base=None,
    p_initial=1.0,
    state_continuation=False,
):
    """Optimize a nonlinear graph-density design and archive its history.

    ``beta_ramp_base`` keeps the geometric projection rate of the reference
    schedule, base**(k / (0.8 * steps)), capped at ``beta_final``; with
    ``beta_final`` above the base this continues the same ramp. With
    ``state_continuation`` each Neo-Hookean analysis starts at the full load
    from the previous accepted equilibrium and falls back to incremental
    loading from the undeformed state if Newton fails.
    """
    if steps < 1 or load <= 0 or beta_final < 1:
        raise ValueError("Use positive steps/load and beta_final >= 1")
    if acceptance not in ("state_solvable", "fixed_parameter_descent"):
        raise ValueError("Unknown graph candidate acceptance rule")
    if stationarity_tolerance is not None and stationarity_tolerance <= 0:
        raise ValueError("stationarity_tolerance must be positive or None")
    if max_candidate_backtracks < 0 or acceptance_relative_tolerance < 0 or min_fixed_updates < 0:
        raise ValueError("Invalid convergence or candidate acceptance settings")
    if max_wall_seconds is not None and max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive or None")
    backtracking = backtracking or acceptance == "fixed_parameter_descent"
    torch.set_default_dtype(torch.float64)
    state_device = state_device or device
    tangent_backend = tangent_backend or (
        "cudss" if torch.device(state_device).type == "cuda" else "scipy"
    )
    if setup is None:
        setup, spec = case(name, nh_transition_beta=nh_transition_beta)
    elif spec is None:
        spec = setup.raw
    op = operator(setup, tangent_backend=tangent_backend, device=state_device)
    op.adaptive_nh_load = bool(adaptive_load)
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
    load_steps = (12 if physics in ["j2", "nh"] else 1) if load_steps is None else int(load_steps)
    if load_steps < 1:
        raise ValueError("load_steps must be positive")
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
        "p_schedule": [float(p_initial), 3.0],
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
        "candidate_acceptance": acceptance,
        "acceptance_relative_tolerance": acceptance_relative_tolerance,
        "max_candidate_backtracks": max_candidate_backtracks,
        "stationarity_tolerance": stationarity_tolerance,
        "stationarity_measure": "L2 parameter gradient of J/current_J, independent of learning rate",
        "min_fixed_updates": min_fixed_updates,
        "adaptive_load": bool(adaptive_load),
        "newton_counter_scope": "Successful load increments only; failed attempts counted separately; wall time includes all attempts",
        "max_wall_seconds": max_wall_seconds,
        "beta_ramp_base": beta_ramp_base,
        "state_continuation": bool(state_continuation),
        "adaptive_load_budgets": {
            "max_subdivisions": 6,
            "max_failed_attempts": 8,
            "max_extra_increments": 32,
        }
        | dict(getattr(op, "nh_adaptive_budgets", {}))
        | {"max_newton_per_increment": int(getattr(op, "nh_max_newton", 100))}
        if adaptive_load
        else None,
        "resume_from": str(resume_from) if resume_from is not None else None,
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
    if max_updates <= start_update or max_updates < int(0.8 * steps):
        raise ValueError("Maximum updates must exceed continuation/resume point")
    termination = "max_updates"
    fixed_updates = 0
    try:
        for it in range(start_update, max_updates + 1):
            fraction = min(1.0, it / max(1, int(0.7 * steps)))
            p = p_initial + (3.0 - p_initial) * fraction
            if beta_ramp_base is None:
                beta = beta_final ** min(1.0, it / max(1, int(0.8 * steps)))
            else:
                beta = min(
                    float(beta_final), float(beta_ramp_base) ** (it / max(1, int(0.8 * steps)))
                )
            op.p.fill_(p)
            opt.zero_grad()
            z = material_logits()
            rho = density_field(
                z, filt, volumes, spec["volume_fraction"] * volumes.sum(), beta, 0.001
            )
            t = time.perf_counter()
            reductions = 0
            stalled = False
            check_descent = (
                acceptance == "fixed_parameter_descent"
                and previous is not None
                and previous["parameters"] == (p, beta)
            )
            while True:
                try:
                    if (
                        max_wall_seconds is not None
                        and time.perf_counter() - start >= max_wall_seconds
                    ):
                        raise TimeoutError("Nonlinear optimization reached its wall-time budget")
                    C, g, d = solve(
                        op,
                        rho.detach().to(op.device),
                        force,
                        physics,
                        True,
                        load_steps,
                        yield_stress,
                        hardening,
                        objective=objective,
                    )
                    if check_descent and C > previous["value"] * (
                        1 + acceptance_relative_tolerance
                    ):
                        raise RuntimeError(
                            f"Objective increase at fixed parameters: {C} > {previous['value']}"
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
                    timed_out = isinstance(exc, TimeoutError)
                    if previous is not None and (
                        timed_out or (check_descent and reductions >= max_candidate_backtracks)
                    ):
                        # No rejected trial is a completed update or a
                        # convergence event. Return the last accepted design.
                        net.load_state_dict(previous["net"])
                        opt.load_state_dict(previous["optimizer"])
                        rho = previous["rho"]
                        C, d = previous["value"], previous["details"]
                        op.p.fill_(previous["parameters"][0])
                        row = rows[-1]
                        it = row["iteration"]
                        stalled = True
                        termination = "wall_time_budget" if timed_out else "stalled"
                        break
                    if (
                        not backtracking
                        or previous is None
                        or reductions >= max_candidate_backtracks
                    ):
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
            if stalled:
                break
            if C0 is None:
                C0 = C
            # A retried Adam step has restored the preceding gradients.
            # Clear them before differentiating this accepted density; without
            # this reset, every backtrack adds stale gradients to the new one.
            opt.zero_grad()
            rho.backward(g.to(device) / C0)
            normalized_gradient = float(
                torch.sqrt(
                    sum(
                        (
                            param.grad.square().sum()
                            for param in net.parameters()
                            if param.grad is not None
                        ),
                        rho.new_zeros(()),
                    )
                )
            ) * abs(C0 / max(abs(C), 1e-30))
            fixed_updates = fixed_updates + 1 if p == 3.0 and beta == beta_final else 0
            row = {
                "iteration": it,
                "candidate_step_reductions": reductions,
                "C": C,
                "p": p,
                "beta": beta,
                "volume": float((rho.detach() * volumes).sum() / volumes.sum()),
                "Mnd": float(
                    400 * (rho.detach() * (1 - rho.detach()) * volumes).sum() / volumes.sum()
                ),
                "state_seconds": time.perf_counter() - t,
                "elapsed": time.perf_counter() - start,
                "parameter_gradient_norm": normalized_gradient,
                "fixed_parameter_states": fixed_updates,
                "descent_checked": check_descent,
                "load_cutback_failures": d.get("load_cutback_failures", 0),
                "load_increments": d.get("load_increments", load_steps),
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
                    # A backtracked candidate reaching this point passed both
                    # equilibrium and the configured objective acceptance gate.
                    accepted=True,
                )
            )
            row["plateau_detected"] = row["converged"]
            row["stationarity_satisfied"] = (
                stationarity_tolerance is None or normalized_gradient <= stationarity_tolerance
            )
            row["converged"] = bool(
                row["converged"]
                and row["stationarity_satisfied"]
                and fixed_updates >= min_fixed_updates
            )
            rows.append(row)
            if state_continuation and physics == "nh":
                op.nh_state_start = d["u"].detach().clone()
            np.save(out / "last_accepted_density.npy", rho.detach().cpu().numpy())
            torch.save(
                dict(
                    network_state=net.state_dict(),
                    optimizer_state=opt.state_dict(),
                    C0=C0,
                    update=it,
                ),
                out / "last_accepted_resume.pt",
            )
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
                    "value": C,
                    "parameters": (p, beta),
                    "rho": rho.detach().clone(),
                    "details": d,
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
