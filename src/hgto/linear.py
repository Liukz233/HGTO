"""Linear topology optimization with a common independent physical evaluation."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from hgto._runtime import json_values
from hgto.optimization import GraphOptimizerConfig, optimize_graph
from hgto.reference import ScikitFEMElasticity
from hgto.reference.oc import OCConfig, optimize_oc


def problem_from_settings(settings, config_dir=Path(".")):
    category = settings.get("category", "linear2d")
    case = dict(settings["case"])
    if category == "linear2d":
        from hgto.linear2d.problems import make_problem
        from hgto.linear2d.physics import make_mesh

        problem = make_problem(**case)
        mesh = make_mesh(problem)
        geometry = dict(
            coords=problem.coords,
            cells=problem.cells,
            fixed=problem.fixed,
            forces=problem.forces,
            shape=problem.shape,
            active_cells=problem.active_cells,
        )
        original = problem
    elif category == "linear3d":
        from hgto.case_studies import make_3d_case

        original, mesh = make_3d_case(**case)
        problem = SimpleNamespace(
            name=original.name,
            description=original.bc_note,
            coords=mesh.coords,
            cells=mesh.econn,
            fixed=original.fixed_dofs,
            forces=original.force[:, None],
            centroids=mesh.element_centroids(),
            volume_fraction=original.volfrac,
            filter_radius=original.rmin,
        )
        geometry = dict(
            coords=mesh.coords,
            cells=mesh.econn,
            fixed=original.fixed_dofs,
            forces=original.force[:, None],
            shape=(original.nelz, original.nely, original.nelx),
        )
    elif category == "domains":
        from hgto.domains.io import load_case

        metadata, arrays, mesh = load_case(config_dir / case["mesh"])
        problem = SimpleNamespace(
            name=metadata["name"],
            description=metadata.get("description", metadata["name"]),
            coords=arrays["coords"],
            cells=arrays["econn"],
            fixed=arrays["fixed_dofs"],
            forces=arrays["force"].reshape(-1, 1),
            centroids=mesh.element_centroids(),
            volume_fraction=metadata["target_volume"],
            filter_radius=metadata["filter_radius"],
        )
        geometry = dict(
            coords=problem.coords,
            cells=problem.cells,
            fixed=problem.fixed,
            forces=problem.forces,
            element_volumes=arrays["element_volumes"],
        )
        original = metadata
    else:
        raise ValueError(f"Unsupported study category {category}")
    return category, problem, mesh, geometry, original


class LibraryPhysics:
    def __init__(self, fem):
        self.fem = fem

    def evaluate(self, rho):
        c, g, _ = self.fem.evaluate(rho.detach().cpu().numpy())
        return c, g

    def set_penalty(self, p):
        self.fem.penalty = float(p)

    @property
    def max_residual(self):
        return self.fem.max_residual


def save(output, geometry, result, settings, source):
    output = Path(output)
    np.save(output / "rho.npy", result["rho"])
    np.save(output / "state.npy", result["state"])
    np.savez_compressed(output / "geometry.npz", **geometry)
    snap = result["snapshots"]
    np.savez_compressed(
        output / "snapshots.npz",
        steps=[s[0] for s in snap],
        elapsed_s=[s[1] for s in snap],
        rho=np.stack([s[2] for s in snap]),
    )
    for name, obj in [
        ("result.json", result["summary"]),
        ("history.json", result["history"]),
        ("source_manifest.json", source),
    ]:
        (output / name).write_text(json.dumps(json_values(obj), indent=2, allow_nan=False) + "\n")
    fields = list(dict.fromkeys(k for row in result["history"] for k in row))
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["history"])
    if "resume_checkpoint" in result:
        torch.save(result["resume_checkpoint"], output / "resume.pt")
    if "network_state" in result:
        torch.save(result["network_state"], output / "network.pt")
    if "design_variables" in result:
        np.save(output / "design.npy", result["design_variables"])
    (output / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))


def run(settings, output, config_dir=Path("."), threads=2, resume_from=None):
    """Return one fully evaluated design; never overwrite a result directory."""
    method = settings.get("method", "hgto")
    if method not in ("hgto", "oc"):
        raise ValueError("Use the NTopo adapter for the separate TensorFlow runtime")
    if method == "oc" and settings.get("oc", {}).get("device", "cpu") != "cpu":
        raise ValueError(
            "The mature-library SIMP-OC implementation uses CPU mechanics; select --device cpu."
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    package = Path(__file__).parent
    source = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    opts = dict(settings.get(method, {}))
    device = opts.get("device", "cpu" if method == "oc" else "cuda:0")
    if method == "oc":
        opts.pop("device", None)
        device = "cpu"
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    category, problem, mesh, geometry, original = problem_from_settings(settings, config_dir)
    if method == "oc":
        config = OCConfig(**opts)

        def progress(row):
            if (row["iteration"] + 1) % 25 == 0:
                print(
                    f"OC update={row['iteration'] + 1} C={row['C']:.8g} elapsed={row['elapsed_s']:.1f}s",
                    flush=True,
                )

        result = optimize_oc(problem, config, started_at=t0, callback=progress)
    else:
        config = GraphOptimizerConfig(**opts)
        library = None
        if device == "cpu":
            library = ScikitFEMElasticity.from_problem(problem)
            physics = LibraryPhysics(library)
        elif category in ("linear2d", "domains"):
            from hgto.linear2d.physics import GraphElasticity

            physics = GraphElasticity(problem, mesh, device)
        else:
            from hgto.linear3d.physics import Physics

            physics = Physics(original, mesh, device)
        volumes = geometry.get("element_volumes")
        result = optimize_graph(
            mesh,
            physics,
            problem.volume_fraction,
            problem.filter_radius,
            config,
            centroids=problem.centroids,
            volumes=volumes,
            started_at=t0,
            resume_from=resume_from,
        )
        result["summary"]["max_state_residual"] = physics.max_residual
        result["summary"]["state_backend"] = (
            "scikit-fem CPU"
            if library is not None
            else (
                "FP64 graph Jacobi-PCG GPU"
                if getattr(physics.operator, "preconditioner", "mgcg") == "jacobi"
                else (
                    "FP64 graph SA-AMG-PCG GPU"
                    if physics.operator.preconditioner == "amgcg"
                    else "FP64 graph MG-PCG GPU"
                )
            )
        )
        if library is None:
            result["summary"]["state_preconditioner"] = physics.operator.preconditioner
        result["summary"]["state_device"] = (
            "cpu" if library is not None else str(physics.operator.device)
        )
        if hasattr(physics, "iterations"):
            result["summary"]["state_iterations"] = physics.iterations
        if library is not None:
            library.close()
    optimizer_end = time.perf_counter()
    reference = ScikitFEMElasticity.from_problem(problem)
    c, g, u = reference.evaluate(result["rho"])
    metadata = reference.metadata()
    relative = abs(c - result["summary"]["C_raw"]) / c
    if relative > 1e-6:
        raise RuntimeError(f"Common-library final compliance mismatch {relative}")
    v = reference.element_volumes
    achieved = float(v @ result["rho"] / v.sum())
    if abs(achieved - problem.volume_fraction) > 1e-7:
        raise RuntimeError("Final physical volume is infeasible")
    reference.close()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    result["state"] = u
    result["summary"].update(
        C_raw=c,
        volume=achieved,
        target_volume=problem.volume_fraction,
        case=problem.name,
        description=problem.description,
        filter_radius=problem.filter_radius,
        wall_s=wall,
        optimization_end_s=optimizer_end - t0,
        validation_s=wall - (optimizer_end - t0),
        independent_relative_C_error=relative,
        independent_fem=metadata,
        cpu_threads=threads,
        grayness_percent=float(400 * (v @ (result["rho"] * (1 - result["rho"]))) / v.sum()),
        gray_fraction=float(v @ ((result["rho"] > 0.05) & (result["rho"] < 0.95)) / v.sum()),
        hardware=(torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "CPU"),
        timing_scope="Case construction, physics/network setup, all optimization evaluations, and final independent scikit-fem physical-field evaluation; excludes imports, serialization, plots and offline trajectory rescoring",
    )
    if category == "linear3d":
        result["summary"]["case_definition"] = original.extra
    result["snapshots"].append((len(result["history"]) - 1, wall, result["rho"].copy()))
    settings = dict(settings)
    settings[method] = asdict(config)
    save(output, geometry, result, settings, source)
    return result
