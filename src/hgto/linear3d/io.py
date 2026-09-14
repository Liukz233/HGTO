"""Density mapping and portable three-dimensional result files."""

import json
import numpy as np


def mapping(x, A, beta):
    bar = A @ x
    if not beta:
        return bar, np.ones_like(bar)
    t = np.tanh(beta * (bar - 0.5))
    den = 2 * np.tanh(beta / 2)
    return (np.tanh(beta / 2) + t) / den, beta * (1 - t * t) / den


def save(folder, problem, mesh, rho, u, history, snapshots, result):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "rho.npy", rho)
    np.save(folder / "displacement.npy", u)
    np.savez_compressed(
        folder / "geometry.npz",
        coords=mesh.coords,
        econn=mesh.econn,
        fixed_dofs=problem.fixed_dofs,
        force=problem.force,
        shape=[problem.nelz, problem.nely, problem.nelx],
    )
    np.savez_compressed(
        folder / "snapshots.npz", steps=[x[0] for x in snapshots], rho=[x[1] for x in snapshots]
    )
    result.update(
        case=problem.name,
        steps=len(history),
        C_raw=float(problem.force @ u),
        volume=float(np.mean(rho)),
        grayness_percent=float(400 * np.mean(rho * (1 - rho))),
        target_volume=problem.volfrac,
        filter_radius=problem.rmin,
        description=problem.bc_note,
        penalty=3.0,
        E0=1.0,
        Emin=1e-6,
        nu=0.3,
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (folder / "case.json").write_text(json.dumps(problem.extra, indent=2) + "\n")
    print(json.dumps(result), flush=True)
