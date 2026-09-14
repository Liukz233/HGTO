"""Run the disclosed NTopo NH adaptation and score its unchanged final field.

Launch with the HGTO interpreter. TensorFlow uses its separate environment.
The default 200 outer iterations, 1000 displacement steps and 50 density
batches retain the author's selected 2D training budget. Smaller budgets
are diagnostics and are explicitly recorded as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback


PUBLIC = Path(__file__).resolve().parents[1]


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def score_final(folder, *, threads=2, load_steps=12):
    """Use the shared finite-deformation evaluator on the unmodified density."""
    import numpy as np
    import torch
    from hgto.nonlinear.optimization import case
    from hgto.nonlinear.evaluation import evaluate_nh

    torch.set_num_threads(threads)
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    before = hashlib.sha256((folder / "rho.npy").read_bytes()).hexdigest()
    rho = np.load(folder / "rho.npy", allow_pickle=False)
    if (
        rho.ndim != 1
        or not np.isfinite(rho).all()
        or np.any(rho < config["rho_min"])
        or np.any(rho > 1.0)
    ):
        raise ValueError("Expected the unchanged finite physical density in [rho_min,1]")
    setup, spec = case(config["case"], nh_transition_beta=config.get("beta0"))
    if config["gamma_mode"] != spec["nh_interpolation"]["gamma_mode"]:
        raise ValueError("Training interpolation differs from the common NH material")
    data = np.load(folder / "input.npz", allow_pickle=False)
    expected = dict(
        coords=setup.mesh.coords,
        cells=setup.mesh.econn,
        fixed=setup.fixed_dofs,
        forces=setup.f * config["load"],
    )
    if any(not np.array_equal(data[name], value) for name, value in expected.items()):
        raise ValueError("Training geometry/load differs from the common NH case")
    for name, value in [
        ("coords", setup.mesh.coords),
        ("econn", setup.mesh.econn),
        ("unit_force", setup.f),
    ]:
        np.save(folder / f"{name}.npy", value)
    protocol = dict(
        **spec,
        physics="nh",
        force_resultant=config["load"],
        objective="complementary_work",
        load_steps=load_steps,
        method="NTopo (NH adaptation)",
        gamma_mode=config["gamma_mode"],
        mixed_sign_oc=config.get("mixed_sign_oc", "native"),
        density_processing="None; raw NTopo field, no projection or volume repair.",
    )
    write(folder / "protocol.json", protocol)
    final = evaluate_nh(folder, output=folder / "verification")
    if hashlib.sha256((folder / "rho.npy").read_bytes()).hexdigest() != before:
        raise RuntimeError("The raw final density changed during independent evaluation")
    record = json.loads((folder / "result.json").read_text())
    report = dict(
        final=final,
        objective="J=-2 min_u Pi=2(f.u-U)",
        load_steps=load_steps,
        cpu_threads=threads,
        evaluation_wall_s=final["evaluation_s"],
        density_sha256=before,
        wall_s_process=record["wall_s_process"],
        comparison_wall_s=record["wall_s_process"] + final["evaluation_s"],
        density_processing="None; raw NTopo density, no thresholding or volume repair.",
        timing_scope="Full TensorFlow process plus one independent common NH setup and load-path solution. File loading, hashing and output serialization are excluded from the final evaluation interval.",
        evaluator_source_sha256=hashlib.sha256(
            (PUBLIC / "src/hgto/nonlinear/evaluation.py").read_bytes()
        ).hexdigest(),
    )
    write(folder / "common_nh_evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load", type=float, required=True)
    parser.add_argument("--case", choices=["cantilever_nh", "bridge_nh"], default="cantilever_nh")
    parser.add_argument("--device", default="0", help="GPU index, or cpu")
    parser.add_argument("--tf-python", default=os.environ.get("HGTO_NTOPO_PYTHON"))
    parser.add_argument("--tf-keras-path", default=os.environ.get("HGTO_NTOPO_TF_KERAS_PATH"))
    parser.add_argument("--upstream", type=Path, default=os.environ.get("HGTO_NTOPO_UPSTREAM_PATH"))
    parser.add_argument("--outer", type=int, default=200)
    parser.add_argument("--inner", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--samples", type=int, default=7500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--fem-threads", type=int, default=2)
    parser.add_argument("--load-steps", type=int, default=12)
    parser.add_argument("--nh-transition-beta", type=float)
    args = parser.parse_args()
    if not args.tf_python:
        parser.error("Set HGTO_NTOPO_PYTHON or --tf-python")
    if args.load <= 0 or any(
        v < 1
        for v in (
            args.outer,
            args.inner,
            args.batches,
            args.samples,
            args.snapshot_interval,
            args.fem_threads,
            args.load_steps,
        )
    ):
        parser.error("Load and all iteration/sample/thread counts must be positive")
    if args.output.exists():
        raise FileExistsError("Use a new result directory")
    upstream = (args.upstream or PUBLIC / "third_party/ntopo/upstream").resolve()
    if not (upstream / "ntopo/train.py").exists():
        raise FileNotFoundError("Author source missing")
    import numpy as np
    from hgto.nonlinear.optimization import case

    setup, spec = case(args.case, nh_transition_beta=args.nh_transition_beta)
    out = args.output.resolve()
    out.mkdir(parents=True)
    config = dict(
        case=args.case,
        family="bridge" if args.case == "bridge_nh" else "cantilever",
        load=args.load,
        volume_fraction=spec["volume_fraction"],
        rho_min=0.001,
        penalty=3.0,
        emin_fraction=1e-6,
        gamma_mode="simp_heaviside",
        beta0=spec["nh_interpolation"]["beta0"],
        eta0=0.01,
        mixed_sign_oc="reciprocal_linear",
        seed=args.seed,
        outer=args.outer,
        inner=args.inner,
        batches=args.batches,
        sample_budget=args.samples,
        snapshot_interval=args.snapshot_interval,
    )
    write(out / "config.json", config)
    np.savez_compressed(
        out / "input.npz",
        coords=setup.mesh.coords,
        cells=setup.mesh.econn,
        centroids=setup.mesh.element_centroids(),
        fixed=setup.fixed_dofs,
        forces=setup.f * args.load,
    )
    write(out / "case.json", spec)
    write(
        out / "protocol.json",
        spec
        | dict(
            physics="nh",
            force_resultant=args.load,
            objective="complementary_work",
            load_steps=args.load_steps,
            method="NTopo (NH adaptation)",
        ),
    )
    source = PUBLIC / "third_party/ntopo/nonlinear_worker.py"
    worker = out / "worker_executed.py"
    shutil.copy2(source, worker)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="" if args.device == "cpu" else args.device,
        TF_USE_LEGACY_KERAS="1",
        TF_CPP_MIN_LOG_LEVEL="2",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="2",
        HGTO_NTOPO_UPSTREAM_PATH=str(upstream),
        HGTO_PUBLIC_PATH=str(PUBLIC),
        PYTHONDONTWRITEBYTECODE="1",
    )
    if args.tf_keras_path:
        env["PYTHONPATH"] = args.tf_keras_path + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    with (out / "worker.log").open("w") as log:
        process = subprocess.run(
            [args.tf_python, str(worker), "--out", str(out)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - started
    process_record = dict(
        exit_code=process.returncode,
        wall_s_process=elapsed,
        runtime_python=args.tf_python,
        device_requested=args.device,
        executed_worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
    )
    write(out / "process_record.json", process_record)
    if process.returncode:
        raise RuntimeError(f"NTopo NH failed; complete diagnostic log: {out / 'worker.log'}")
    record = json.loads((out / "result.json").read_text())
    record.update(process_record)
    write(out / "result.json", record)
    try:
        report = score_final(out, threads=args.fem_threads, load_steps=args.load_steps)
    except Exception as error:
        write(
            out / "common_nh_failure.json",
            dict(
                error=str(error),
                traceback=traceback.format_exc(),
                density_processing="Raw NTopo field retained unchanged.",
            ),
        )
        raise
    print(
        json.dumps(dict(**report["final"], comparison_wall_s=report["comparison_wall_s"]), indent=2)
    )


if __name__ == "__main__":
    main()
