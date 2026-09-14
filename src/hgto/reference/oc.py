"""Headless SIMP-OC using scikit-fem mechanics in two or three dimensions."""

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy import sparse

from hgto.linear2d.filters import DensityMap
from .scikit_fem import ScikitFEMElasticity
from hgto.stopping import PhysicalStopping, StopConfig


@dataclass(frozen=True)
class OCConfig:
    betas: tuple = (1.0, 2.0, 4.0, 8.0)
    max_stage_steps: int = 1200
    min_stage_steps: int = 40
    move: float = 0.2
    change_tolerance: float = 0.01
    compliance_tolerance: float = 0.001
    stable_steps: int = 5
    snapshot_interval: int = 25
    solver: str = "auto"
    rho_min: float = 0.001
    penalties: tuple | None = None
    stopping_rule: str = "joint"
    continuation_stage_cap: int | None = None


class PhysicalDensityMap(DensityMap):
    """The same post-projection density floor used by HGTO's density map."""

    def __init__(self, centroids, radius, rho_min):
        super().__init__(centroids, radius)
        self.rho_min = rho_min

    def physical(self, x, beta=0.0):
        rho, derivative = super().physical(x, beta)
        return (self.rho_min + (1 - self.rho_min) * rho, (1 - self.rho_min) * derivative)


def weighted_oc_update(x, dc, dv, mapping, volume, beta, weights, move):
    """Multiplicative OC; bisection enforces physical, area/volume-weighted mass."""
    lower, upper = np.maximum(0.0, x - move), np.minimum(1.0, x + move)
    scale = np.sqrt(np.maximum(-dc, 0.0) / np.maximum(dv, 1e-30))
    left, right = 0.0, max(1.0, float(np.max(scale * scale)))
    for _ in range(100):
        multiplier = 0.5 * (left + right)
        candidate = np.clip(x * scale / np.sqrt(multiplier), lower, upper)
        rho, _ = mapping.physical(candidate, beta)
        if float(weights @ rho) > volume:
            left = multiplier
        else:
            right = multiplier
        if (right - left) / (right + left + 1e-30) < 1e-8:
            break
    return candidate, rho


def optimize_oc(
    problem, config=OCConfig(), *, physics=None, started_at=None, initial_design=None, callback=None
):
    """Return ``rho/history/snapshots/summary/design_variables``.

    The problem exposes coords, cells, fixed, forces, centroids, filter_radius,
    volume_fraction, name and description; ``types.SimpleNamespace`` suffices.
    Every recorded compliance belongs to the recorded physical field at the
    recorded SIMP penalty; the final penalty is always p=3.
    With a continuation-stage cap, intermediate parameter stages advance at
    that cap (or their density criterion); only the final stage establishes
    optimization convergence. Legacy uncapped stages retain their own checks.
    """
    started_at = time.perf_counter() if started_at is None else started_at
    penalties = config.penalties or tuple(3.0 for _ in config.betas)
    if (
        len(penalties) != len(config.betas)
        or not penalties
        or penalties[-1] != 3.0
        or any(p < 1.0 for p in penalties)
    ):
        raise ValueError("One penalty per beta is required, ending at p=3")
    if config.stopping_rule not in ("joint", "density_change", "physical"):
        raise ValueError("Unknown OC stopping rule")
    if config.continuation_stage_cap is not None and config.continuation_stage_cap < 1:
        raise ValueError("Continuation stage cap must be positive")
    owned_physics = physics is None
    physics = (
        ScikitFEMElasticity.from_problem(problem, solver=config.solver)
        if physics is None
        else physics
    )
    mapping = PhysicalDensityMap(problem.centroids, problem.filter_radius, config.rho_min)
    # A distance hat is a quadrature approximation of a spatial integral.
    # Weight contributing cells by their measure on nonuniform meshes.
    mapping.A = mapping.A @ sparse.diags(physics.element_volumes)
    mapping.A = sparse.diags(1.0 / np.asarray(mapping.A.sum(axis=1)).ravel()) @ mapping.A
    n = len(physics.cells)
    weights = physics.element_volumes / physics.element_volumes.sum()
    x = (
        np.full(n, problem.volume_fraction)
        if initial_design is None
        else np.asarray(initial_design, dtype=float).copy()
    )
    if x.shape != (n,) or np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("Initial design must contain one value in [0,1] per cell")

    def restore_volume(design, beta):
        from scipy.special import expit, logit

        z = logit(np.clip(design, 1e-12, 1 - 1e-12))
        left = float(z.min()) - 48.0
        right = float(z.max()) + 48.0
        for _ in range(120):
            shift = (left + right) / 2
            candidate = expit(z - shift)
            rho, _ = mapping.physical(candidate, beta)
            if float(weights @ rho) > problem.volume_fraction:
                left = shift
            else:
                right = shift
        return expit(z - (left + right) / 2)

    setup_s = time.perf_counter() - started_at
    history, snapshots, stages = [], [], []
    stopping = PhysicalStopping(
        StopConfig(
            objective_tolerance=config.compliance_tolerance,
            density_tolerance=config.change_tolerance,
            patience=config.stable_steps,
        )
    )
    for stage, beta in enumerate(config.betas):
        physics.penalty = float(penalties[stage])
        x = restore_volume(x, beta)
        stable, values = 0, []
        stage_budget = (
            min(config.max_stage_steps, config.continuation_stage_cap)
            if config.continuation_stage_cap is not None and stage < len(config.betas) - 1
            else config.max_stage_steps
        )
        for iteration in range(stage_budget):
            rho, dprojection = mapping.physical(x, beta)
            compliance, gradient, _ = physics.evaluate(rho)
            dc = mapping.pullback(gradient, dprojection)
            dv = mapping.pullback(weights, dprojection)
            candidate, rho_candidate = weighted_oc_update(
                x,
                dc,
                dv,
                mapping,
                problem.volume_fraction,
                beta,
                weights,
                config.move / max(1.0, beta),
            )
            design_change = float(np.max(np.abs(candidate - x)))
            change = float(np.max(np.abs(rho_candidate - rho)))
            values.append(compliance)
            relative = (
                (max(values[-10:]) - min(values[-10:])) / values[-1]
                if len(values) >= 10
                else float("inf")
            )
            stable = (
                stable + 1
                if iteration >= config.min_stage_steps - 1
                and design_change < config.change_tolerance
                and change < config.change_tolerance
                and (
                    config.stopping_rule == "density_change"
                    or relative < config.compliance_tolerance
                )
                and abs(float(weights @ rho) - problem.volume_fraction) <= 1e-7
                and physics.last_residual <= 1e-8
                else 0
            )
            stop_check = {}
            if config.stopping_rule == "physical":
                stop_check = stopping.observe(
                    compliance,
                    rho,
                    parameters=(physics.penalty, float(beta)),
                    continuation_complete=stage == len(config.betas) - 1,
                    volume_error=float(weights @ rho) - problem.volume_fraction,
                    residual=physics.last_residual,
                )
                stable = stop_check["stable_checks"]
            elapsed = time.perf_counter() - started_at
            row = dict(
                iteration=len(history),
                stage=stage,
                beta=float(beta),
                penalty=physics.penalty,
                C=compliance,
                volume=float(weights @ rho),
                design_change=design_change,
                max_density_change=change,
                relative_C_window=relative,
                elapsed_s=elapsed,
                equilibrium_residual=physics.last_residual,
            )
            row.update(stop_check)
            history.append(row)
            if row["iteration"] % config.snapshot_interval == 0:
                snapshots.append((row["iteration"], elapsed, rho.copy()))
            if callback is not None:
                callback(row)
            if stable >= config.stable_steps or iteration == stage_budget - 1:
                break
            x = candidate
        stages.append(
            dict(
                beta=float(beta),
                penalty=physics.penalty,
                steps=iteration + 1,
                converged=stable >= config.stable_steps,
                design_change=design_change,
                max_density_change=change,
                relative_C_window=relative,
                stage_budget=stage_budget,
                continuation_stage=stage < len(config.betas) - 1,
            )
        )
        if stage < len(config.betas) - 1:
            x = candidate
    elapsed = time.perf_counter() - started_at
    snapshots.append((history[-1]["iteration"], elapsed, rho.copy()))
    final_converged = (
        stages[-1]["converged"]
        if config.continuation_stage_cap is not None
        else all(s["converged"] for s in stages)
    )
    summary = dict(
        method="SIMP-OC",
        case=problem.name,
        description=problem.description,
        C_raw=compliance,
        volume=float(weights @ rho),
        target_volume=problem.volume_fraction,
        gray_fraction=float(weights @ ((rho > 0.05) & (rho < 0.95))),
        grayness_percent=float(400 * weights @ (rho * (1 - rho))),
        converged=final_converged,
        stages=stages,
        termination="converged" if final_converged else "max_stage_steps",
        state_evaluations=len(history),
        setup_s=setup_s,
        optimization_s=elapsed - setup_s,
        wall_s=elapsed,
        max_state_residual=physics.max_residual,
        filter_radius=problem.filter_radius,
        config=asdict(config),
        fem_backend=physics.metadata(),
        processing="Element-measure-weighted hat density filter; tanh projection; post-projection rho_min floor; final p=3 (per-stage penalties recorded); physical weighted volume; stage-wise convergence",
        timing_scope="Problem supplied by caller; includes library basis setup, optimization, all FEM assembly and solves; excludes imports, serialization and plotting",
    )
    if owned_physics:
        physics.close()
    return dict(
        rho=rho.copy(),
        history=history,
        snapshots=snapshots,
        summary=summary,
        design_variables=x.copy(),
    )
