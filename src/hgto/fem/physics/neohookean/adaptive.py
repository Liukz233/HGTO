"""Load cutbacks along the prescribed zero-to-full-force loading path."""

from dataclasses import replace
import time
import torch
from .newton import solve_nh_state


@torch.no_grad()
def solve_nh_adaptive(
    operator,
    rho,
    f_ext,
    *,
    n_ramp=48,
    max_subdivisions=6,
    max_failed_attempts=8,
    max_extra_increments=32,
    record_ramp_history=True,
    **kwargs,
):
    """Bisect a failed load increment, always from its last converged state.

    No state from another density is used. Returned histories keep the
    requested load stations; internal cutbacks are separately counted.
    A successful coarse increment is never replaced by an energy-selected
    alternative. Load-discretization studies remain required.
    """
    if f_ext.shape[0] != 1:
        raise ValueError("Adaptive nonlinear runner currently supports one load case")
    if n_ramp < 1 or max_subdivisions < 0 or max_failed_attempts < 1 or max_extra_increments < 0:
        raise ValueError("Invalid adaptive load budget")
    started = time.perf_counter()
    previous = torch.zeros_like(f_ext)
    histories, residuals, counts = [], [], []
    totals = dict(newton=0, inner=0, backtracks=0, fallback=0, inexact=0)
    failed_attempts = successful_increments = 0

    def advance(left, right, seed, depth):
        nonlocal failed_attempts, successful_increments
        if successful_increments >= n_ramp + max_extra_increments:
            raise RuntimeError("Adaptive NH load path exceeded its extra-increment budget")
        try:
            state = solve_nh_state(
                operator,
                rho,
                f_ext * right,
                u0=seed,
                n_ramp=1,
                record_ramp_history=False,
                **kwargs,
            )
        except RuntimeError:
            failed_attempts += 1
            if depth >= max_subdivisions or failed_attempts >= max_failed_attempts:
                raise
            middle = (left + right) / 2
            half = advance(left, middle, seed, depth + 1)
            return advance(middle, right, half.u, depth + 1)
        successful_increments += 1
        for key, field in [
            ("newton", "newton_iterations"),
            ("inner", "pcg_iterations"),
            ("backtracks", "backtracks"),
            ("fallback", "indefinite_fallbacks"),
            ("inexact", "inexact_directions"),
        ]:
            totals[key] += int(getattr(state, field).sum())
        return state

    for index in range(n_ramp):
        before = totals["newton"]
        state = advance(index / n_ramp, (index + 1) / n_ramp, previous, 0)
        previous = state.u
        histories.append(state.u[0].clone())
        residuals.append(state.residual_rel[0])
        counts.append(totals["newton"] - before)

    def integer(value):
        return torch.tensor([value], dtype=torch.long, device=operator.device)

    result = replace(
        state,
        newton_iterations=integer(totals["newton"]),
        pcg_iterations=integer(totals["inner"]),
        iterations=integer(totals["inner"]),
        backtracks=integer(totals["backtracks"]),
        indefinite_fallbacks=integer(totals["fallback"]),
        inexact_directions=integer(totals["inexact"]),
        load_factors=torch.arange(1, n_ramp + 1, dtype=operator.dtype, device=operator.device)
        / n_ramp,
        ramp_newton_iterations=torch.tensor([counts], dtype=torch.long, device=operator.device),
        ramp_residual_rel=torch.stack(residuals)[None],
        ramp_u=torch.stack(histories)[None] if record_ramp_history else None,
        t_state_total=time.perf_counter() - started,
    )
    result.adaptive_failed_attempts = failed_attempts
    result.adaptive_successful_increments = successful_increments
    return result
