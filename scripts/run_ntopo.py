"""Run author NTopo from a configuration YAML and independently score its raw density.

Launch this script with the HGTO/scikit-fem interpreter. TensorFlow training
runs in the separate interpreter selected by --tf-python or
HGTO_NTOPO_PYTHON. Existing result directories are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def load_problem(config):
    """Read only the common geometry section; HGTO/OC optimizer options are unused."""
    import yaml
    from hgto.linear import problem_from_settings

    config = Path(config).resolve()
    settings = yaml.safe_load(config.read_text())
    category = settings.get("category", "linear2d")
    if category not in ("linear2d", "linear3d", "domains"):
        raise ValueError(
            "The paper NTopo entry point supports the revised linear and irregular cases"
        )
    if category != "domains":
        family = settings["case"]["family"]
        allowed = (
            ("cantilever", "mbb", "inclined")
            if category == "linear2d"
            else ("cantilever_3d", "four_foot_support", "torsion_member")
        )
        if family not in allowed:
            raise ValueError(f"Unsupported paper NTopo case: {family}")
    category, problem, _, _, original = problem_from_settings(settings, config.parent)
    if category == "domains":
        if problem.name not in ("l_bracket_domain", "perforated_bracket"):
            raise ValueError(f"Unsupported paper NTopo domain: {problem.name}")
        problem.domain_metadata = original
    # The adapter uses Problem3DSpec's exact patch metadata and callable mesh API.
    return category, original if category == "linear3d" else problem, settings


def _threads(count):
    if count < 1:
        raise ValueError("FEM thread count must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(count)
    import torch

    torch.set_num_threads(count)


def score_final(folder, *, threads=2, solver="auto"):
    """Return final-scoring metadata without writing or changing any run file.

    Primary field names match the study's retained NTopo scoring records.
    Histories are intentionally not rescored; their diagnostic cost is zero.
    """
    _threads(threads)
    import numpy as np
    from hgto.reference import ScikitFEMElasticity
    from hgto.linear2d.problems import make_problem

    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    record = json.loads((folder / "result.json").read_text())
    with np.load(folder / "input.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in ("coords", "cells", "fixed", "forces")}
    target = float(config["volume_fraction"])

    def evaluate(fields, density_path):
        rho = np.load(density_path, allow_pickle=False)
        if (
            rho.shape != (len(fields["cells"]),)
            or not np.isfinite(rho).all()
            or np.any(rho < 0.0)
            or np.any(rho > 1.0)
        ):
            raise ValueError(
                "Expected one finite raw density in [0,1] per cell; no repair is permitted"
            )
        density_hash = hashlib.sha256(density_path.read_bytes()).hexdigest()
        started = time.perf_counter()
        fem = ScikitFEMElasticity(
            **fields, E0=1.0, Emin=1e-6, nu=0.3, penalty=3.0, thickness=1.0, solver=solver
        )
        setup_wall = time.perf_counter() - started
        try:
            final_started = time.perf_counter()
            c, _, _ = fem.evaluate(rho)
            volume = fem.element_volumes
            achieved = float(volume @ rho / volume.sum())
            final = dict(
                compliance=float(c),
                volume_fraction=achieved,
                relative_volume_violation=achieved / target - 1.0,
                grayness=float(volume @ (4 * rho * (1 - rho)) / volume.sum()),
                free_dof_residual=fem.last_residual,
            )
            final_wall = time.perf_counter() - final_started
            metadata = fem.metadata()
        finally:
            fem.close()
        elapsed = time.perf_counter() - started
        if hashlib.sha256(density_path.read_bytes()).hexdigest() != density_hash:
            raise RuntimeError("Raw density file changed during final evaluation")
        return dict(
            final=final,
            evaluation_backend=metadata,
            setup_wall_s=setup_wall,
            final_evaluation_wall_s=final_wall,
            evaluation_wall_s=elapsed,
            density_sha256=density_hash,
        )

    result = evaluate(data, folder / "rho.npy")
    process_wall = float(record["wall_s_process"])
    result.update(
        method="NTopo",
        case=config["case"],
        density="raw emitted NTopo, unchanged",
        history=[],
        diagnostic_history_evaluation_wall_s=0.0,
        cpu_threads=threads,
        wall_s_process=process_wall,
        comparison_wall_s=process_wall + result["setup_wall_s"] + result["final_evaluation_wall_s"],
        timing_scope="Final FEM setup and one state evaluation are additional to the unchanged training-process wall time; no history rescoring. evaluation_wall_s also includes solver cleanup. File loading, hashing and JSON output are outside the FEM intervals.",
        comparison_timing_scope="Process wall plus setup_wall_s plus final_evaluation_wall_s for the primary grid; the optional fine-grid row retains full shared process wall plus its evaluation_wall_s.",
    )
    high = config.get("high_resolution")
    if high:
        nx, ny = map(int, high)
        problem = make_problem(config["family"], nx, ny, volume=target)
        fields = dict(
            coords=problem.coords, cells=problem.cells, fixed=problem.fixed, forces=problem.forces
        )
        fine = evaluate(fields, folder / "rho_high_resolution.npy")
        fine.update(
            case=problem.name,
            grid_nx_ny=[nx, ny],
            comparison_wall_s=process_wall + fine["evaluation_wall_s"],
            trained_field_relationship="Same continuous NTopo density field; no additional training.",
        )
        result["high_resolution"] = fine
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Paper configuration YAML; its case definition is shared",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New native NTopo result directory"
    )
    parser.add_argument(
        "--tf-python", help="TensorFlow interpreter; alternatively set HGTO_NTOPO_PYTHON"
    )
    parser.add_argument(
        "--tf-keras-path", help="Optional overlay; alternatively HGTO_NTOPO_TF_KERAS_PATH"
    )
    parser.add_argument(
        "--upstream",
        help="Optional author-source directory; alternatively HGTO_NTOPO_UPSTREAM_PATH",
    )
    parser.add_argument("--device", default="0", help="TensorFlow GPU index (default 0), or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer", type=int, help="Default: 200 in 2D, 100 in 3D")
    parser.add_argument("--inner", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument(
        "--high-resolution",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        help="Extra cantilever readout of the same field, e.g. 240 80",
    )
    parser.add_argument("--fem-threads", type=int, default=2)
    parser.add_argument(
        "--fem-solver", choices=("auto", "scipy", "pypardiso", "pypardiso-spd"), default="auto"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and display configuration without training or writing output",
    )
    parser.add_argument(
        "--stopping-config",
        type=Path,
        help="JSON StopConfig for read-only online physical stopping; outer becomes a safety cap",
    )
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.fem_threads < 1:
        parser.error("fem-threads must be positive")
    category, problem, settings = load_problem(args.config)
    outer = args.outer if args.outer is not None else (100 if category == "linear3d" else 200)
    if any(value < 1 for value in (outer, args.inner, args.batches, args.snapshot_interval)):
        parser.error("outer, inner, batches and snapshot-interval must be positive")
    high = tuple(args.high_resolution) if args.high_resolution else None
    stopping = json.loads(args.stopping_config.read_text()) if args.stopping_config else None
    if high:
        if category != "linear2d" or settings["case"]["family"] != "cantilever":
            parser.error("Additional fine-grid readout is supported only for the cantilever")
        ny, nx = problem.shape
        if high[0] < nx or high[1] < ny or high == (nx, ny) or high[0] * ny != high[1] * nx:
            parser.error(
                "Fine-grid dimensions must increase resolution while preserving the cantilever aspect ratio"
            )
    interpreter = args.tf_python or os.environ.get("HGTO_NTOPO_PYTHON")
    launch = dict(
        case=problem.name,
        category=category,
        seed=args.seed,
        device=args.device,
        outer=outer,
        inner=args.inner,
        batches=args.batches,
        snapshot_interval=args.snapshot_interval,
        high_resolution=high,
        tf_python=interpreter,
        tf_keras_path=args.tf_keras_path or os.environ.get("HGTO_NTOPO_TF_KERAS_PATH"),
        upstream=args.upstream or os.environ.get("HGTO_NTOPO_UPSTREAM_PATH"),
        fem_threads=args.fem_threads,
        fem_solver=args.fem_solver,
    )
    print(json.dumps(launch, indent=2), flush=True)
    if args.dry_run:
        return
    if not interpreter:
        parser.error("Set HGTO_NTOPO_PYTHON or --tf-python to the separate TensorFlow interpreter")
    if args.output.exists():
        raise FileExistsError(f"Use a new result directory: {args.output}")
    from importlib.util import find_spec

    if find_spec("skfem") is None:
        parser.error(
            "Install HGTO's base installation in this launcher environment before training"
        )
    if args.fem_solver.startswith("pypardiso") and find_spec("pypardiso") is None:
        parser.error("The selected final solver requires HGTO's fast-cpu extra")
    from hgto.baselines.ntopo import run

    run(
        problem,
        args.output,
        python=interpreter,
        tf_keras_path=args.tf_keras_path,
        upstream=args.upstream,
        device=args.device,
        seed=args.seed,
        outer=outer,
        inner=args.inner,
        batches=args.batches,
        snapshot_interval=args.snapshot_interval,
        high_resolution=high,
        stopping=stopping,
    )
    evaluation = score_final(args.output, threads=args.fem_threads, solver=args.fem_solver)
    with (args.output / "common_fem_evaluation.json").open("x") as stream:
        json.dump(evaluation, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # These records are separate from the adapter's own executed run config.
    (args.output / "benchmark_config.yaml").write_bytes(args.config.read_bytes())
    (args.output / "launch_options.json").write_text(json.dumps(launch, indent=2) + "\n")
    report = dict(
        case=evaluation["case"],
        **evaluation["final"],
        comparison_wall_s=evaluation["comparison_wall_s"],
    )
    if "high_resolution" in evaluation:
        report["high_resolution"] = evaluation["high_resolution"]["final"]
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
