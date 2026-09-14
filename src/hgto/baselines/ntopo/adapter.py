"""Map public Problem geometry/loads to NTopo, without changing its algorithm."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def run(
    problem,
    out,
    *,
    python=None,
    tf_keras_path=None,
    device="cpu",
    seed=42,
    outer=200,
    inner=1000,
    batches=50,
    snapshot_interval=10,
    high_resolution=None,
    upstream=None,
    stopping=None,
):
    """Run upstream train_mmse; low budgets are smoke tests, never paper results.

    ``high_resolution=(nx, ny)`` evaluates the same mesh-free field on an
    additional grid, recording zero additional training and its readout cost.
    """
    upstream_path = Path(
        upstream
        or os.environ.get("HGTO_NTOPO_UPSTREAM_PATH")
        or Path(__file__).resolve().parents[4] / "third_party" / "ntopo" / "upstream"
    ).resolve()
    if not (upstream_path / "ntopo" / "train.py").is_file():
        raise FileNotFoundError(
            "NTopo author source is missing. Use the source checkout with "
            "an editable HGTO installation, or set HGTO_NTOPO_UPSTREAM_PATH "
            "to its third_party/ntopo/upstream directory."
        )
    out = Path(out).resolve()
    if (out / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing NTopo run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    is3d = hasattr(problem, "nelz")
    if is3d:
        family = problem.extra.get("family")
        if family not in ("cantilever_3d", "torsion_member", "four_foot_support"):
            raise ValueError(
                "3D NTopo mapping accepts validated cantilever_3d/torsion_member/four_foot_support boxes."
            )
        coords, cells = problem.coords(), problem.econn()
        shape = (problem.nelz, problem.nely, problem.nelx)
        centers, fixed, forces = (
            problem.element_centroids(),
            problem.fixed_dofs,
            problem.force[:, None],
        )
        active = np.arange(problem.n_elem)
        volume, radius, description = problem.volfrac, problem.rmin, problem.bc_note
        extra = problem.extra
        if high_resolution:
            raise ValueError("3D high-resolution readout is not implemented.")
    elif hasattr(problem, "domain_metadata"):
        if problem.name not in ("l_bracket_domain", "perforated_bracket"):
            raise ValueError("Unknown irregular-domain NTopo mapping")
        if problem.forces.shape[1] != 1 or problem.coords.shape[1] != 2:
            raise ValueError("Irregular NTopo requires one simultaneous 2D load")
        family = problem.name
        coords, cells, centers = problem.coords, problem.cells, problem.centroids
        fixed, forces = problem.fixed, problem.forces
        shape, active = (len(cells),), np.arange(len(cells))
        volume, radius, description = (
            problem.volume_fraction,
            problem.filter_radius,
            problem.description,
        )
        extra = problem.domain_metadata
        if high_resolution:
            raise ValueError("Irregular results are read at the canonical cell centroids")
    else:
        if problem.forces.shape[1] != 1:
            raise ValueError("NTopo adapter currently accepts one simultaneous load case.")
        shape = tuple(problem.shape)
        if len(shape) != 2 or len(problem.coords[0]) != 2:
            raise ValueError("Unsupported problem geometry.")
        family = problem.name.rsplit("_", 1)[0]
        if family not in ("cantilever", "inclined", "mbb", "bridge", "l_bracket"):
            raise ValueError(f"Unsupported boundary-condition family: {family}")
        coords, cells, centers = problem.coords, problem.cells, problem.centroids
        fixed, forces, active = problem.fixed, problem.forces, problem.active_cells
        volume, radius, description = (
            problem.volume_fraction,
            problem.filter_radius,
            problem.description,
        )
        extra = {}
        if high_resolution and len(active) != shape[0] * shape[1]:
            raise ValueError(
                "Higher-resolution readout currently requires a complete rectangular2D domain."
            )
    np.savez(
        out / "input.npz",
        coords=coords,
        cells=cells,
        centroids=centers,
        fixed=fixed,
        forces=forces,
        active_cells=active,
    )
    config = dict(
        case=problem.name,
        family=family,
        shape=list(shape),
        dimension=3 if is3d else 2,
        extra=extra,
        description=description,
        volume_fraction=volume,
        canonical_filter_radius=radius,
        outer=outer,
        inner=inner,
        batches=batches,
        snapshot_interval=snapshot_interval,
        seed=seed,
        high_resolution=list(high_resolution) if high_resolution else None,
    )
    if stopping is not None:
        if is3d:
            raise ValueError("Online stopping is currently validated for the 2D pilots only")
        config["stopping"] = dict(stopping)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        TF_USE_LEGACY_KERAS="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="" if device == "cpu" else str(device),
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="2",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    overlay = tf_keras_path or env.get("HGTO_NTOPO_TF_KERAS_PATH")
    if overlay:
        env["PYTHONPATH"] = str(overlay) + os.pathsep + env.get("PYTHONPATH", "")
    interpreter = python or env.get("HGTO_NTOPO_PYTHON", sys.executable)
    worker_source = Path(__file__).with_name("worker.py").read_bytes()
    worker_file = out / "worker_executed.py"
    worker_file.write_bytes(worker_source)
    env["HGTO_NTOPO_UPSTREAM_PATH"] = str(upstream_path)
    start = time.perf_counter()
    with open(out / "worker.log", "w") as log:
        if stopping is None:
            process = subprocess.run(
                [str(interpreter), str(worker_file), "--out", str(out)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        else:
            from .stopping_monitor import run_monitored

            process = run_monitored(
                [str(interpreter), str(worker_file), "--out", str(out)],
                env=env,
                log=log,
                out=out,
                config=config,
            )
    wall = time.perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"NTopo exited {process.returncode}. See {out / 'worker.log'}")
    result = json.loads((out / "result.json").read_text())
    result["wall_s_process"] = wall
    result["executed_worker_sha256"] = hashlib.sha256(worker_source).hexdigest()
    result["runtime_python"] = str(interpreter)
    result["runtime_tf_keras_overlay"] = str(overlay) if overlay else None
    result["device_requested"] = device
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return np.load(out / "rho.npy"), result
