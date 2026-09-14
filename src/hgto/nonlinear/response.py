"""Cross-evaluation of separately optimized linear and nonlinear designs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from hgto.nonlinear.optimization import case, operator, solve, diagnostics
from hgto.fem.physics.j2.solver import solve_j2_history
from hgto.fem.solvers.tangent import solve_sparse_tangent
from hgto.fem.kernels import gauss_strains


def _read_design(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory / "record.json").read_text())
    protocol_path = directory / "protocol.json"
    protocol = (
        json.loads(protocol_path.read_text()) if protocol_path.exists() else record["protocol"]
    )
    arrays = {
        name: np.load(directory / f"{name}.npy")
        for name in ("rho", "coords", "econn", "unit_force")
    }
    return directory, record, protocol, arrays


def compare_designs(linear_directory, nonlinear_directory, output):
    """Evaluate two saved densities using a common nonlinear model and history.

    Inputs must be compatible outputs of the nonlinear optimization example.
    Neither design is modified. Files contain full physical states and the
    work-conjugate port displacement; unloading is never the objective.
    """
    inputs = [_read_design(linear_directory), _read_design(nonlinear_directory)]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = inputs[1][2]
    name = protocol["case"]
    nonlinear_model = protocol["physics"]
    if nonlinear_model not in ("nh", "j2"):
        raise ValueError("The nonlinear input must be an NH or J2 optimized design")
    models = ["elastic_j2", "j2"] if nonlinear_model == "j2" else ["linear", "nh"]
    if inputs[0][2]["physics"] != models[0]:
        raise ValueError(f"The linear input must use {models[0]} for this comparison")
    _, base_spec = case(name)
    refinement = protocol["nx"] // base_spec["nx"]
    setup, spec = case(name, refine=refinement)
    op = operator(setup, p=3.0)
    for _, _, candidate, arrays in inputs:
        for key in ("case", "nx", "ny", "volume_fraction", "force_resultant"):
            if candidate[key] != protocol[key]:
                raise ValueError(f"Design protocols differ in {key}")
        if candidate["fixed_dofs"] != protocol["fixed_dofs"]:
            raise ValueError("Design support constraints differ")
        if not np.array_equal(arrays["econn"], setup.mesh.econn):
            raise ValueError("Saved connectivity differs from the declared problem")
        if not np.allclose(arrays["coords"], setup.mesh.coords, rtol=0, atol=1e-12):
            raise ValueError("Saved coordinates differ from the declared problem")
        if not np.allclose(arrays["unit_force"], setup.f, rtol=0, atol=1e-14):
            raise ValueError("Saved applied load differs from the declared problem")
        if arrays["rho"].shape != (op.n_elements,) or not np.isfinite(arrays["rho"]).all():
            raise ValueError("Saved density is invalid")
    load = protocol["force_resultant"]
    steps = int(protocol.get("load_steps", 12))
    sy = protocol.get("yield_stress", 0.0015)
    hardening = protocol.get("hardening", 0.1)
    force = torch.as_tensor(setup.f, dtype=op.dtype, device=op.device) * load
    rows = []
    for (directory, record, candidate, arrays), design in zip(inputs, models):
        rho = torch.as_tensor(arrays["rho"], dtype=op.dtype, device=op.device)
        volumes = torch.as_tensor(setup.mesh.element_volumes(), dtype=op.dtype, device=op.device)
        item = {
            "design": design,
            "case": name,
            "density_source": str(directory),
            "volume": float((rho * volumes).sum() / volumes.sum()),
            "load": load,
        }
        for physics in models:
            C, _, fields = solve(op, rho, force, physics, False, steps, sy, hardening)
            arrays_out = {
                k: value.detach().cpu().numpy()
                for k, value in fields.items()
                if isinstance(value, torch.Tensor)
            }
            np.savez_compressed(output / f"{design}_design__{physics}_response.npz", **arrays_out)
            item[physics] = {"C": C, **diagnostics(fields, spec)}
            if physics == "nh":
                factors = np.r_[0.0, np.linspace(1 / steps, 1, steps)]
                u = np.concatenate([np.zeros((1, op.n_nodes, 2)), arrays_out["history_u"]])
                delta = (
                    (u[:, spec["port_nodes"]] * np.asarray(spec["unit_direction"])).sum(-1).mean(-1)
                )
                np.savez_compressed(
                    output / f"{design}_loadpath.npz",
                    factors=factors,
                    load=factors * load,
                    delta=delta,
                    u=u,
                )
        if nonlinear_model == "j2":
            factors = torch.cat(
                [
                    torch.linspace(1 / steps, 1, steps, dtype=op.dtype),
                    torch.linspace((steps - 1) / steps, 0, steps, dtype=op.dtype),
                ]
            )
            state = solve_j2_history(
                op,
                rho,
                factors[:, None, None] * force[None],
                sy,
                hardening,
                rtol=1e-9,
                pcg_rtol=1e-10,
                max_newton=80,
                linear_solver=solve_sparse_tangent,
            )
            us = torch.stack([h["u"][0] for h in state["history"]])
            alpha = torch.stack([h["alpha"][0] for h in state["history"]])
            plastic = torch.stack([h["plastic_strain"][0] for h in state["history"]])
            strain = gauss_strains(us, op.econn, op.dN_dx)
            strain_norm = torch.sqrt(
                strain[..., 0] ** 2 + strain[..., 1] ** 2 + 0.5 * strain[..., 2] ** 2
            )
            delta = (
                (
                    us[:, spec["port_nodes"]]
                    * torch.as_tensor(spec["unit_direction"], dtype=op.dtype)
                )
                .sum(-1)
                .mean(-1)
            )
            mask = rho > 0.5
            if not bool(mask.any()):
                raise ValueError("No material cells above density 0.5 for plastic diagnostics")
            peak = float(delta[steps - 1])
            residual = float(delta[-1])
            item["loading_unloading"] = {
                "peak_displacement": peak,
                "residual_displacement": residual,
                "residual_over_peak": residual / peak,
                "residual_over_L": residual / spec["L"],
                "maximum_material_strain_full_path": float(strain_norm[:, mask].max()),
                "maximum_material_alpha_full_path": float(alpha[:, mask].max()),
                "peak_matches_optimization_relative": abs(peak - item["j2"]["port_displacement"])
                / abs(peak),
                "max_residual": max(h["residual_rel"] for h in state["history"]),
            }
            np.savez_compressed(
                output / f"{design}_loadpath.npz",
                factors=np.r_[0.0, factors.numpy()],
                load=np.r_[0.0, factors.numpy() * load],
                delta=np.r_[0.0, delta.numpy()],
                u=np.concatenate([np.zeros((1, op.n_nodes, 2)), us.numpy()]),
                alpha=alpha.numpy(),
                plastic_strain=plastic.numpy(),
                strain_norm=strain_norm.numpy(),
                material_mask=mask.numpy(),
            )
        rows.append(item)
    result = {
        "spec": spec,
        "comparisons": rows,
        "nonlinear_objective_reduction_vs_linear_design": 1
        - rows[1][nonlinear_model]["C"] / rows[0][nonlinear_model]["C"],
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    flat = [
        {
            "case": name,
            "design": row["design"],
            "reanalysis": model,
            "load": load,
            "C": row[model]["C"],
            "volume": row["volume"],
            "port_displacement": row[model]["port_displacement"],
            "port_displacement_over_L": row[model]["port_displacement_over_L"],
        }
        for row in rows
        for model in models
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return result
