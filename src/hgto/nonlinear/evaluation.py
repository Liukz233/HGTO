"""Common evaluation of unchanged densities in the finite-deformation study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from hgto._runtime import preserve_default_dtype
from hgto.nonlinear.optimization import case, operator, solve, diagnostics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@preserve_default_dtype
def evaluate_nh(directory, *, output=None):
    """Recompute J, terminal work and the actual load path, without density repair.

    Canonical inputs contain protocol, rho, coordinates, connectivity and the
    unit force. An explicit output directory keeps verification costs separate
    from the original optimization record.
    """
    torch.set_default_dtype(torch.float64)
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory / "verification"
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((directory / "protocol.json").read_text())
    setup, spec = case(
        protocol["case"], nh_transition_beta=protocol.get("nh_interpolation", {}).get("beta0")
    )
    if protocol.get("nh_interpolation", {}) != spec.get("nh_interpolation", {}):
        raise ValueError("Saved nonlinear interpolation differs from the declared canonical case")
    for name, expected in [
        ("coords", setup.mesh.coords),
        ("econn", setup.mesh.econn),
        ("unit_force", setup.f),
    ]:
        actual = np.load(directory / f"{name}.npy")
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-13):
            raise ValueError(f"{name} differs from the declared canonical case")
    if protocol["fixed_dofs"] != spec["fixed_dofs"]:
        raise ValueError("Support constraints differ from the canonical case")
    density = np.load(directory / "rho.npy")
    if density.shape != (setup.mesh.n_elements,) or not np.isfinite(density).all():
        raise ValueError("Density has invalid shape or nonfinite values")
    if density.min() < 0 or density.max() > 1:
        raise ValueError("Physical density lies outside [0,1]")
    load = float(protocol["force_resultant"])
    if load <= 0:
        raise ValueError("Use a positive physical load")
    start = time.perf_counter()
    op = operator(setup, p=3.0)
    rho = torch.as_tensor(density, dtype=op.dtype, device=op.device)
    force = torch.from_numpy(setup.f) * load
    steps = int(protocol.get("load_steps", 12))
    objective, _, fields = solve(op, rho, force, "nh", False, steps, objective="complementary_work")
    seconds = time.perf_counter() - start
    volumes = setup.mesh.element_volumes()
    record = dict(
        case=spec["case"],
        objective="complementary_work",
        J=objective,
        C_terminal=fields["terminal_work"],
        volume=float(np.dot(density, volumes) / volumes.sum()),
        target_volume=spec["volume_fraction"],
        force_resultant=load,
        evaluation_s=seconds,
        density_sha256=sha256(directory / "rho.npy"),
        density_processing="none",
        **diagnostics(fields, spec),
    )
    history = fields["history_u"].detach().numpy()
    us = np.concatenate([np.zeros((1, op.n_nodes, 2)), history])
    factors = np.r_[0.0, np.linspace(1 / steps, 1.0, steps)]
    delta = (us[:, spec["port_nodes"]] * np.asarray(spec["unit_direction"])).sum(-1).mean(-1)
    if not np.isclose(delta[-1] * load, record["C_terminal"], rtol=1e-12, atol=1e-14):
        raise RuntimeError("Port displacement is not work-conjugate to the applied load")
    np.savez_compressed(output / "response.npz", load=load * factors, delta=delta, u=us)
    (output / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


import shutil


def finalize(directory, method="HGTO"):
    f = Path(directory)
    r = json.loads((f / "record.json").read_text())
    p = json.loads((f / "protocol.json").read_text())
    if p["physics"] != "nh":
        raise ValueError("This common evaluator is for NH; plasticity uses compare_designs")
    v = evaluate_nh(f)
    error = abs(v["J"] / r["final"]["C"] - 1)
    if error > 1e-7 or v["residual"] > 1e-8:
        raise ValueError("Independent evaluation differs from the saved physical state")
    v.update(
        method=method,
        training_to_validation_relative_error=error,
        wall_s=r["wall_s_including_setup"] + v["evaluation_s"],
        optimization_s=r["wall_s_including_setup"],
        validation_s=v["evaluation_s"],
        time_scope="Construction and optimization plus one independent final evaluation; imports and plotting excluded",
        termination=r["termination"],
        converged=r["converged"],
        design_updates=r["design_updates"],
        network_device=p.get("network_device"),
        state_device=p.get("state_device", "cpu"),
    )
    (f / "result.json").write_text(json.dumps(v, indent=2) + "\n")
    shutil.copy2(f / "verification/response.npz", f / "response.npz")
    return v
