"""First-order checks independent of the optimizer's current step size."""

import numpy as np


def projected_gradient(x, dc, dv, weights, objective, floor=1e-9):
    """Mass-metric projected gradient for a box and linearized volume equality.

    The subproblem minimizes ``dc/J @ d + .5 * sum(weights*d*d)``
    subject to ``dv @ d=0`` and the design bounds. Its unique solution is
    zero exactly at a first-order KKT point (for the current physical map).
    This certificate must not be scaled by a learning rate or move limit.
    """
    x, dc, dv, weights = (np.asarray(v, dtype=float) for v in (x, dc, dv, weights))
    if not all(np.isfinite(v).all() for v in (x, dc, dv, weights)):
        raise ValueError("Non-finite stationarity inputs")
    if np.any(weights <= 0) or np.any(dv <= 0) or not np.isfinite(objective):
        raise ValueError("Stationarity requires positive volume derivatives and weights")
    g = dc / (max(abs(float(objective)), np.finfo(float).tiny) * weights)
    a = dv / weights
    lower, upper = floor - x, 1 - floor - x
    # These finite brackets force every coordinate to the upper/lower bound.
    left = float(np.min((-g - upper) / a)) - 1.0
    right = float(np.max((-g - lower) / a)) + 1.0
    for _ in range(100):
        multiplier = 0.5 * (left + right)
        direction = np.clip(-g - multiplier * a, lower, upper)
        error = float(dv @ direction)
        if abs(error) < 1e-13:
            break
        if error > 0:
            left = multiplier
        else:
            right = multiplier
    return direction, dict(
        projected_gradient_inf=float(np.max(np.abs(direction))),
        projected_gradient_rms=float(np.sqrt(weights @ direction**2)),
        projected_gradient_volume_residual=abs(float(dv @ direction)),
        projected_gradient_slope=float(dc @ direction),
    )
