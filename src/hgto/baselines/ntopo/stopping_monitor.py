"""Read-only physical convergence monitor; never returns a design or gradient."""

import json
import subprocess
import time
from dataclasses import asdict
import numpy as np
from hgto.reference import ScikitFEMElasticity
from hgto.stopping import PhysicalStopping, StopConfig


def run_monitored(command, *, env, log, out, config):
    options = dict(config["stopping"])
    solver = options.pop("solver", "pypardiso-spd")
    stop_config = StopConfig(**options)
    monitor = PhysicalStopping(stop_config)
    with np.load(out / "input.npz") as data:
        fields = {k: data[k] for k in ("coords", "cells", "fixed", "forces")}
    started = time.perf_counter()
    fem = ScikitFEMElasticity(**fields, E0=1.0, Emin=1e-6, nu=0.3, penalty=3.0, solver=solver)
    setup_s = time.perf_counter() - started
    weights = fem.element_volumes / fem.element_volumes.sum()
    fine = None
    if config.get("high_resolution"):
        from hgto.linear2d.problems import make_problem

        nx, ny = config["high_resolution"]
        problem = make_problem(config["family"], nx, ny, volume=config["volume_fraction"])
        fine = ScikitFEMElasticity.from_problem(problem, solver=solver)
        fine_weights = fine.element_volumes / fine.element_volumes.sum()
        fine_monitor = PhysicalStopping(stop_config)
    setup_s = time.perf_counter() - started
    records = []
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            log.write(line)
            log.flush()
            if not line.startswith("STOP_REQUEST "):
                continue
            request = json.loads(line[len("STOP_REQUEST ") :])
            iteration = int(request["iteration"])
            tick = time.perf_counter()
            rho = np.load(out / "snapshots" / f"rho_{iteration:04d}.npy")
            if (
                rho.shape != weights.shape
                or not np.isfinite(rho).all()
                or np.any((rho < 0) | (rho > 1))
            ):
                raise ValueError("Invalid raw density in convergence check")
            objective, _, _ = fem.evaluate(rho)
            volume = float(weights @ rho)
            check = monitor.observe(
                objective,
                rho,
                parameters=(3.0,),
                continuation_complete=True,
                volume_error=volume - config["volume_fraction"],
                residual=fem.last_residual,
            )
            check.update(
                iteration=iteration,
                C=float(objective),
                volume=volume,
                evaluation_s=time.perf_counter() - tick,
            )
            if fine is not None:
                high = np.load(out / "snapshots" / f"rho_high_{iteration:04d}.npy")
                high_c, _, _ = fine.evaluate(high)
                high_volume = float(fine_weights @ high)
                high_check = fine_monitor.observe(
                    high_c,
                    high,
                    parameters=(3.0,),
                    continuation_complete=True,
                    volume_error=high_volume - config["volume_fraction"],
                    residual=fine.last_residual,
                )
                high_check.update(C=float(high_c), volume=high_volume)
                check["fine_grid"] = high_check
                check["converged"] = bool(check["converged"] and high_check["converged"])
                check["evaluation_s"] = time.perf_counter() - tick
            records.append(check)
            (out / "stopping_history.json").write_text(json.dumps(records, indent=2) + "\n")
            # Only the stop decision is sent to the training process. No FEM
            # displacement, objective or derivative enters the NTopo update.
            process.stdin.write(json.dumps({"converged": check["converged"]}) + "\n")
            process.stdin.flush()
        process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        fem.close()
        if fine is not None:
            fine.close()
        (out / "stopping_monitor.json").write_text(
            json.dumps(
                dict(
                    config=asdict(stop_config),
                    setup_s=setup_s,
                    evaluations=len(records),
                    evaluation_s=sum(x["evaluation_s"] for x in records),
                    scope="Read-only common-FEM convergence checks at every outer iteration; all monitoring time included in process wall; no gradients or fields returned to training.",
                ),
                indent=2,
            )
            + "\n"
        )
    return process
