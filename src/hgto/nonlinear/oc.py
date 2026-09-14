"""SIMP optimality criteria with the same nonlinear states used by HGTO.

Design variables are pre-filter element densities. The physical filter,
projection, density floor, volume measure and constitutive equations match
``nonlinear.optimization``. Nonlinear compliance uses the actual implicit
adjoint, never the linear-elastic self-adjoint sensitivity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import json
import time
import traceback

import numpy as np
import torch
from scipy.special import expit, logit

from hgto._runtime import preserve_default_dtype
from hgto.nonlinear.optimization import case, operator, solve, diagnostics
from hgto.topopt.pipeline.filter import DensityFilter


@dataclass(frozen=True)
class NonlinearOCConfig:
    objective: str = "terminal_work"
    continuation_updates: int = 180
    max_updates: int = 600
    min_fixed_updates: int = 20
    move: float = 0.2
    beta_final: float = 8.0
    p_initial: float = 1.0
    p_final: float = 3.0
    filter_radius: float = 1.0
    rho_min: float = 0.001
    design_floor: float = 1e-9
    load_steps: int = 12
    yield_stress: float = 0.0015
    hardening: float = 0.1
    change_tolerance: float = 0.005
    objective_tolerance: float = 0.0005
    stable_steps: int = 8
    objective_window: int = 10
    snapshot_interval: int = 20
    positive_gradient_relative_tolerance: float = 1e-10
    mixed_sign_update: str = "reject"
    max_candidate_backtracks: int = 10
    acceptance: str = "state_solvable"
    acceptance_relative_tolerance: float = 1e-8
    stopping_rule: str = "joint"
    tangent_backend: str = "scipy"

    def validate(self):
        if self.tangent_backend not in ("scipy", "pypardiso"):
            raise ValueError("Unknown tangent backend")
        if self.stopping_rule not in ("joint", "density_change", "physical"):
            raise ValueError("Unknown OC stopping rule")
        if self.acceptance not in ("state_solvable", "fixed_parameter_descent"):
            raise ValueError("Unknown OC candidate acceptance rule")
        if (
            not np.isfinite(self.acceptance_relative_tolerance)
            or self.acceptance_relative_tolerance < 0
            or self.max_candidate_backtracks < 0
        ):
            raise ValueError("Invalid candidate acceptance tolerance or backtracking budget")
        if self.mixed_sign_update not in ("reject", "reciprocal_linear"):
            raise ValueError("Unknown mixed-sign OC update")
        if self.objective not in ("terminal_work", "complementary_work"):
            raise ValueError("Unknown nonlinear objective")
        if (
            self.continuation_updates < 0
            or self.max_updates < 1
            or self.max_updates < self.continuation_updates
            or self.min_fixed_updates < 0
            or self.load_steps < 1
        ):
            raise ValueError("Invalid update/load-step budgets")
        if not (
            0 < self.move <= 1
            and 1 <= self.beta_final
            and 1 <= self.p_initial <= self.p_final
            and 0 < self.rho_min < 1
            and 0 < self.design_floor < 0.5
        ):
            raise ValueError("Invalid OC material/projection parameters")
        if self.filter_radius <= 0 or self.yield_stress <= 0 or self.hardening <= 0:
            raise ValueError("Use positive filter radius, yield stress and hardening")
        if self.stable_steps < 1 or self.objective_window < 2 or self.snapshot_interval < 1:
            raise ValueError("Invalid convergence or snapshot configuration")

    def parameters(self, iteration):
        if self.continuation_updates == 0:
            return float(self.p_final), float(self.beta_final)
        p_fraction = min(1.0, iteration / max(1, int(0.7 * self.continuation_updates)))
        beta_fraction = min(1.0, iteration / max(1, int(0.8 * self.continuation_updates)))
        return (
            self.p_initial + (self.p_final - self.p_initial) * p_fraction,
            self.beta_final**beta_fraction,
        )


class NonlinearDensityMap:
    """Area-weighted hat filter and the HGTO normalized tanh projection."""

    def __init__(self, mesh, radius=1.0, rho_min=0.001, design_floor=1e-9):
        self.filter = DensityFilter(mesh, radius)
        self.A = self.filter.normalized_matrix
        self.weights = np.asarray(mesh.element_volumes(), dtype=float)
        self.weights /= self.weights.sum()
        self.rho_min = float(rho_min)
        self.design_floor = float(design_floor)

    def physical(self, x, beta):
        bar = self.A @ np.asarray(x, dtype=float)
        if beta:
            t = np.tanh(beta * (bar - 0.5))
            denominator = 2 * np.tanh(beta / 2)
            h = 0.5 + t / denominator
            dh = beta * (1 - t * t) / denominator
        else:
            h, dh = bar, np.ones_like(bar)
        return self.rho_min + (1 - self.rho_min) * h, (1 - self.rho_min) * dh

    def pullback(self, physical_gradient, projection_derivative):
        return np.asarray(self.A.T @ (physical_gradient * projection_derivative)).ravel()

    def enforce_volume(self, x, beta, target):
        """Uniform logit shift, only to initialize or re-center a changed beta.

        This is the same physical volume root used by the HGTO density map.
        It is not an objective-directed repair or a topology refinement.
        """
        z = logit(np.clip(x, self.design_floor, 1 - self.design_floor))
        left, right = float(z.min()) - 48.0, float(z.max()) + 48.0
        for _ in range(120):
            shift = 0.5 * (left + right)
            value = expit(z - shift)
            rho, _ = self.physical(value, beta)
            error = float(self.weights @ rho) - target
            if abs(error) < 2e-13:
                return value, rho
            if error > 0:
                left = shift
            else:
                right = shift
        raise RuntimeError("Physical volume root failed")


def oc_update(
    x, dc, dv, mapping, target, beta, move, positive_tolerance=1e-10, mixed_sign_update="reject"
):
    """OC with exact physical mass and an explicit optional signed branch.

    ``reciprocal_linear`` uses a reciprocal approximation for negative
    objective derivatives and a linear approximation for nonnegative ones.
    For a nonnegative volume multiplier, the latter minimize at the move
    lower bound. A signed equality multiplier supplies additional mass if
    this restricted branch cannot reach the target. No gradient is negated.
    """
    if mixed_sign_update not in ("reject", "reciprocal_linear"):
        raise ValueError("Unknown mixed-sign OC update")
    x, dc, dv = (np.asarray(a, dtype=float) for a in (x, dc, dv))
    if not all(np.isfinite(a).all() for a in (x, dc, dv)):
        raise ValueError("Non-finite OC inputs")
    if np.any(dv <= 0):
        raise ValueError("OC requires strictly positive physical-volume derivatives")
    tolerance = positive_tolerance * max(float(np.max(np.abs(dc))), np.finfo(float).tiny)
    positive = dc > tolerance
    info = {
        "gradient_min": float(dc.min()),
        "gradient_max": float(dc.max()),
        "positive_gradient_tolerance": float(tolerance),
        "positive_gradient_count": int(positive.sum()),
        "roundoff_positive_gradient_count": int(((dc > 0) & ~positive).sum()),
        "mixed_sign_update": mixed_sign_update,
        "linear_branch_count": int((dc >= 0).sum()),
        "signed_multiplier_used": False,
        "volume_multiplier": None,
        "linear_tie_fraction": None,
    }
    if positive.any() and mixed_sign_update == "reject":
        raise ValueError(
            "Nonlinear compliance has materially positive filtered sensitivities; "
            "classical multiplicative OC is not applicable. Audit: " + json.dumps(info)
        )
    lower = np.maximum(mapping.design_floor, x - move)
    upper = np.minimum(1 - mapping.design_floor, x + move)
    negative = dc < 0
    # A nonnegative derivative is minimized at its lower bound for lambda>=0.
    # With all negative derivatives this is exactly the classic OC path.
    branch_upper = np.where(negative, upper, lower)
    lower_rho, _ = mapping.physical(lower, beta)
    upper_rho, _ = mapping.physical(branch_upper, beta)
    if (
        not float(mapping.weights @ lower_rho) - 1e-11
        <= target
        <= float(mapping.weights @ upper_rho) + 1e-11
    ):
        from .signed_oc import negative_multiplier_update

        candidate, signed_info = negative_multiplier_update(
            x,
            dc,
            dv,
            lower,
            upper,
            target,
            lambda design: float(mapping.weights @ mapping.physical(design, beta)[0]),
        )
        rho, _ = mapping.physical(candidate, beta)
        return candidate, rho, info | signed_info
    scale = np.zeros_like(dc)
    scale[negative] = np.sqrt(-dc[negative] / dv[negative])
    left, right = 0.0, max(1.0, float(np.max(scale * scale)))
    for _ in range(140):
        multiplier = 0.5 * (left + right)
        candidate = np.clip(x * scale / np.sqrt(multiplier), lower, upper)
        candidate[~negative] = lower[~negative]
        rho, _ = mapping.physical(candidate, beta)
        error = float(mapping.weights @ rho) - target
        if abs(error) <= 2e-12:
            return candidate, rho, info
        if error > 0:
            left = multiplier
        else:
            right = multiplier
    raise RuntimeError("OC volume bisection failed; volume error=" + str(error))


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


@preserve_default_dtype
def run(name, physics, load, output, config=NonlinearOCConfig(), *, setup=None, spec=None):
    """Optimize and archive a nonlinear SIMP–OC run, including failures.

    An explicit ``setup/spec`` pair allows exactly the same custom problem
    to be passed to the graph and density optimizers. Continuation parameters
    match the HGTO time schedule; OC then continues at fixed parameters until
    its own convergence criterion or the declared budget is reached.
    """
    config.validate()
    if load <= 0 or physics not in ("linear", "nh", "j2", "elastic_j2"):
        raise ValueError("Use a supported physics model and a positive load")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    if setup is None:
        setup, spec = case(name)
    elif spec is None:
        spec = setup.raw
    if np.any(setup.passive_solid) or np.any(setup.passive_void):
        raise ValueError("This runner requires no passive elements")
    op = operator(setup, tangent_backend=config.tangent_backend)
    mapping = NonlinearDensityMap(
        setup.mesh, config.filter_radius, config.rho_min, config.design_floor
    )
    target = float(setup.volume_fraction)
    if not config.rho_min < target < 1:
        raise ValueError("Volume fraction must lie between density floor and one")
    force = torch.as_tensor(setup.f, dtype=torch.float64) * load
    p, beta = config.parameters(0)
    x, _ = mapping.enforce_volume(np.full(setup.mesh.n_elements, target), beta, target)
    setup_seconds = time.perf_counter() - started
    protocol = dict(spec) | dict(
        method="SIMP-OC",
        physics=physics,
        force_resultant=float(load),
        config=asdict(config),
        architecture=None,
        filter_radius_physical=config.filter_radius,
        rho_min=config.rho_min,
        load_steps=config.load_steps,
        yield_stress=config.yield_stress,
        hardening=config.hardening,
        target_volume=target,
        device="cpu",
        threads=torch.get_num_threads(),
        objective=config.objective,
        optimizer=(
            "Multiplicative optimality criteria; checked nonpositive sensitivities"
            if config.mixed_sign_update == "reject"
            else "Optimality criteria with reciprocal negative-gradient terms and linear nonnegative-gradient terms"
        ),
        processing="Element-measure-weighted hat filter, normalized tanh projection and physical density floor",
        positive_sensitivity_policy=(
            "Reject material positives; record numerically negligible positive terms"
            if config.mixed_sign_update == "reject"
            else "Explicit linear approximation at move lower bound for nonnegative gradients; reject infeasible physical-volume brackets"
        ),
        candidate_acceptance=config.acceptance,
        timing_scope="Case/operator/filter setup, all state/adjoint/update work and rejected candidates plus one final reanalysis; excludes imports and later plotting; incremental diagnostic serialization is included",
    )
    _dump(out / "protocol.json", protocol)
    np.save(out / "coords.npy", setup.mesh.coords)
    np.save(out / "econn.npy", setup.mesh.econn)
    np.save(out / "unit_force.npy", setup.f)
    np.savez_compressed(
        out / "geometry.npz",
        coords=setup.mesh.coords,
        cells=setup.mesh.econn,
        fixed_dofs=setup.fixed_dofs,
        forces=force.numpy(),
    )
    source_root = Path(__file__).parents[1]
    _dump(
        out / "source_manifest.json",
        {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_root.rglob("*.py"))
        },
    )
    rows, failures, snapshots, snapshot_steps = [], [], [], []
    state_calls = gradient_calls = successful_states = successful_gradients = 0
    total_newton = total_completed_load_increments = 0
    stable = fixed_count = 0
    fixed_objectives = []
    previous_x = previous_rho = None
    previous_beta = beta
    previous_p = p
    previous_solution = None
    line_search_stop = None
    converged = False
    rho = mapping.physical(x, beta)[0]
    try:
        for iteration in range(config.max_updates + 1):
            p, beta = config.parameters(iteration)
            if beta != previous_beta:
                x, _ = mapping.enforce_volume(x, beta, target)
            op.p.fill_(p)
            candidate_origin = x.copy()
            reductions = 0
            check_descent = (
                config.acceptance == "fixed_parameter_descent"
                and previous_solution is not None
                and p == previous_p
                and beta == previous_beta
            )
            reference_value = float(previous_solution[0]) if check_descent else None
            acceptance_tolerance = (
                config.acceptance_relative_tolerance * max(abs(reference_value), 1e-30)
                if check_descent
                else None
            )
            state_started = time.perf_counter()
            while True:
                rho, derivative = mapping.physical(x, beta)
                state_calls += 1
                gradient_calls += 1
                state_error = None
                try:
                    value, gradient, details = solve(
                        op,
                        torch.as_tensor(rho),
                        force,
                        physics,
                        True,
                        config.load_steps,
                        config.yield_stress,
                        config.hardening,
                        objective=config.objective,
                    )
                except Exception as exc:
                    state_error = exc
                if state_error is None:
                    successful_states += 1
                    successful_gradients += 1
                    total_newton += int(details.get("newton", 0))
                    total_completed_load_increments += (
                        config.load_steps if physics in ("nh", "j2") else 1
                    )
                    if not check_descent or value <= reference_value + acceptance_tolerance:
                        break
                    rejection = dict(
                        reason="objective_increase",
                        trial_objective=float(value),
                        reference_objective=reference_value,
                        tolerance=acceptance_tolerance,
                    )
                else:
                    rejection = dict(reason="state_failure", error=str(state_error))
                failures.append(
                    dict(
                        iteration=iteration,
                        p=p,
                        beta=beta,
                        candidate_backtracks=reductions,
                        elapsed_s=time.perf_counter() - started,
                        state_call=state_calls,
                        **rejection,
                    )
                )
                _dump(out / "candidate_rejections.json", failures)
                if previous_x is None or reductions >= config.max_candidate_backtracks:
                    if check_descent:
                        # This cached state was accepted at these exact parameters.
                        # Keep its actual iteration index; the rejected attempt is
                        # not a completed design update or a convergence event.
                        line_search_stop = dict(
                            reason="candidate_backtracking_exhausted",
                            attempted_iteration=iteration,
                            last_accepted_iteration=rows[-1]["iteration"],
                            p=p,
                            beta=beta,
                            last_rejection=rejection,
                            elapsed_s=time.perf_counter() - started,
                        )
                        x, rho = previous_x.copy(), previous_rho.copy()
                        value, gradient, details = previous_solution
                        break
                    raise state_error
                reductions += 1
                x = previous_x + 0.5**reductions * (candidate_origin - previous_x)
                x, _ = mapping.enforce_volume(x, beta, target)
            if line_search_stop is not None:
                row = rows[-1]
                iteration = row["iteration"]
                break
            dc = mapping.pullback(gradient.detach().numpy(), derivative)
            dv = mapping.pullback(mapping.weights, derivative)
            next_x, next_rho, gradient_info = oc_update(
                x,
                dc,
                dv,
                mapping,
                target,
                beta,
                config.move / max(1.0, beta),
                config.positive_gradient_relative_tolerance,
                config.mixed_sign_update,
            )
            change = float(np.max(np.abs(next_rho - rho)))
            design_change = float(np.max(np.abs(next_x - x)))
            fixed_parameters = p == config.p_final and beta == config.beta_final
            if fixed_parameters:
                fixed_count += 1
                fixed_objectives.append(float(value))
            window = fixed_objectives[-config.objective_window :]
            relative_window = (
                (max(window) - min(window)) / max(abs(value), 1e-30)
                if len(window) == config.objective_window
                else None
            )
            is_stable = (
                fixed_count >= config.min_fixed_updates
                and change < config.change_tolerance
                and design_change < config.change_tolerance
                and (
                    config.stopping_rule == "density_change"
                    or (
                        relative_window is not None and relative_window < config.objective_tolerance
                    )
                )
                and reductions == 0
                and details["residual"] <= 1e-8
                and abs(float(mapping.weights @ rho) - target) <= 1e-7
            )
            if config.stopping_rule == "physical":
                accepted_change = (
                    float(np.max(np.abs(rho - previous_rho)))
                    if previous_rho is not None
                    else float("inf")
                )
                is_stable = (
                    fixed_count >= config.min_fixed_updates
                    and accepted_change < config.change_tolerance
                    and relative_window is not None
                    and relative_window < config.objective_tolerance
                    and details["residual"] <= 1e-8
                    and abs(float(mapping.weights @ rho) - target) <= 1e-7
                )
            stable = stable + 1 if is_stable else 0
            diag = diagnostics(details, spec)
            row = dict(
                iteration=iteration,
                p=p,
                beta=beta,
                C=float(value),
                volume=float(mapping.weights @ rho),
                Mnd=float(400 * mapping.weights @ (rho * (1 - rho))),
                design_change=design_change,
                max_density_change=change,
                accepted_density_change=(
                    None if previous_rho is None else float(np.max(np.abs(rho - previous_rho)))
                ),
                relative_C_window=relative_window,
                fixed_parameter_states=fixed_count,
                consecutive_stable_states=stable,
                candidate_step_reductions=reductions,
                descent_checked=check_descent,
                descent_reference_objective=reference_value,
                descent_tolerance=acceptance_tolerance,
                state_seconds=time.perf_counter() - state_started,
                elapsed=time.perf_counter() - started,
                state_calls=state_calls,
                gradient_calls=gradient_calls,
                successful_states=successful_states,
                successful_gradient_sweeps=successful_gradients,
                total_newton_iterations=total_newton,
                completed_load_increments=total_completed_load_increments,
                **gradient_info,
                **diag,
            )
            rows.append(row)
            with (out / "history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            if iteration % config.snapshot_interval == 0:
                snapshots.append(rho.copy())
                snapshot_steps.append(iteration)
                np.savez_compressed(
                    out / "snapshots.npz", rho=np.stack(snapshots), steps=snapshot_steps
                )
                print(
                    json.dumps(
                        dict(
                            case=name,
                            method="SIMP-OC",
                            physics=physics,
                            iteration=iteration,
                            C=value,
                            volume=row["volume"],
                            p=p,
                            beta=beta,
                            residual=diag.get("residual"),
                            elapsed=row["elapsed"],
                        )
                    ),
                    flush=True,
                )
            if stable >= config.stable_steps or iteration == config.max_updates:
                converged = stable >= config.stable_steps
                break
            previous_x, previous_rho, previous_beta = x.copy(), rho.copy(), beta
            previous_p = p
            previous_solution = (value, gradient, details)
            x = next_x
        optimization_seconds = time.perf_counter() - started
        check_started = time.perf_counter()
        state_calls += 1
        check_value, _, check_details = solve(
            op,
            torch.as_tensor(rho),
            force,
            physics,
            False,
            config.load_steps,
            config.yield_stress,
            config.hardening,
            objective=config.objective,
        )
        successful_states += 1
        check_error = abs(check_value - value) / max(abs(value), 1e-30)
        if check_error > 1e-7:
            raise RuntimeError(f"Independent repeated final-state check failed: {check_error}")
        final_check_s = time.perf_counter() - check_started
        wall_s = time.perf_counter() - started
        if not snapshot_steps or snapshot_steps[-1] != iteration:
            snapshots.append(rho.copy())
            snapshot_steps.append(iteration)
        np.save(out / "rho.npy", rho)
        np.save(out / "design.npy", x)
        np.savez_compressed(out / "snapshots.npz", rho=np.stack(snapshots), steps=snapshot_steps)
        np.savez_compressed(
            out / "states.npz",
            **{
                key: val.detach().numpy()
                for key, val in check_details.items()
                if isinstance(val, torch.Tensor)
            },
        )
        result = dict(
            protocol=protocol,
            final=row,
            method="SIMP-OC",
            converged=converged,
            termination=(
                "stalled"
                if line_search_stop is not None
                else "convergence"
                if converged
                else "budget"
            ),
            line_search_stop=line_search_stop,
            design_updates=iteration,
            optimization_state_evaluations=len(rows),
            total_state_calls=state_calls,
            successful_states=successful_states,
            successful_gradient_sweeps=successful_gradients,
            failed_state_attempts=sum(entry["reason"] == "state_failure" for entry in failures),
            objective_rejected_candidates=sum(
                entry["reason"] == "objective_increase" for entry in failures
            ),
            rejected_candidates=len(failures),
            total_completed_newton_iterations=total_newton + int(check_details.get("newton", 0)),
            setup_s=setup_seconds,
            optimization_wall_s=optimization_seconds,
            final_check_s=final_check_s,
            wall_s=wall_s,
            final_check_relative_error=check_error,
            final_check=diagnostics(check_details, spec),
            max_state_residual=max(row["residual"] for row in rows),
        )
        _dump(out / "record.json", result)
        return result
    except Exception as exc:
        np.save(out / "failed_density.npy", rho)
        np.save(out / "failed_design.npy", x)
        if "dc" in locals() and "dv" in locals():
            np.savez_compressed(
                out / "failed_sensitivities.npz",
                filtered_objective_gradient=dc,
                filtered_volume_gradient=dv,
                physical_objective_gradient=gradient.detach().numpy(),
            )
        _dump(
            out / "failure.json",
            dict(
                message=str(exc),
                traceback=traceback.format_exc(),
                protocol=protocol,
                completed_states=len(rows),
                state_calls=state_calls,
                successful_states=successful_states,
                elapsed_s=time.perf_counter() - started,
                candidate_rejections=failures,
                last_completed_state=(
                    dict(objective=float(value), **diagnostics(details, spec))
                    if "details" in locals()
                    else None
                ),
            ),
        )
        raise
