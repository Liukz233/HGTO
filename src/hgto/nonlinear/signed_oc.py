"""Feasible signed-multiplier branch of a reciprocal/linear OC subproblem.

This module is NumPy-only so the FEM and TensorFlow adapters use the same rule.
It is used only when a nonnegative equality multiplier cannot supply the
requested mass. The existing positive-multiplier OC path is unchanged.
"""

import numpy as np


def negative_multiplier_update(
    old, gradient, volume_gradient, lower, upper, target, mass, tolerance=2e-12
):
    old, gradient, volume_gradient, lower, upper = [
        np.asarray(x, dtype=float) for x in (old, gradient, volume_gradient, lower, upper)
    ]
    if any(
        x.shape != old.shape or not np.isfinite(x).all()
        for x in (gradient, volume_gradient, lower, upper)
    ) or np.any(volume_gradient <= 0):
        raise ValueError("Invalid signed OC arrays")
    minimum, maximum = float(mass(lower)), float(mass(upper))
    if not minimum - tolerance <= target <= maximum + tolerance:
        raise RuntimeError("OC move bounds cannot bracket the target volume")
    negative = gradient < 0
    base = np.where(negative, upper, lower)
    if float(mass(base)) > target + tolerance:
        raise ValueError("A negative multiplier is unnecessary for this target")
    if abs(float(mass(base)) - target) <= tolerance:
        return base, dict(volume_multiplier=0.0, signed_multiplier_used=True)
    positive_indices = np.flatnonzero(~negative)
    if not len(positive_indices):
        raise RuntimeError("No linear OC branch can supply additional mass")
    ratios = gradient[positive_indices] / volume_gradient[positive_indices]
    levels, groups = np.unique(ratios, return_inverse=True)

    def candidate(count):
        result = base.copy()
        indices = positive_indices[groups < count]
        result[indices] = upper[indices]
        return result

    # For lambda<0 all reciprocal branches are at their upper bounds. Linear
    # branches switch at lambda=-g/dV. Locate the one tied group spanning mass.
    lo, hi = 0, len(levels)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(mass(candidate(mid))) < target:
            lo = mid
        else:
            hi = mid
    a, b = candidate(lo), candidate(hi)
    left, right = 0.0, 1.0
    for _ in range(100):
        fraction = (left + right) / 2
        result = a + fraction * (b - a)
        value = float(mass(result))
        if abs(value - target) <= tolerance:
            return result, dict(
                volume_multiplier=-float(levels[lo]),
                signed_multiplier_used=True,
                linear_tie_fraction=fraction,
            )
        if value < target:
            left = fraction
        else:
            right = fraction
    raise RuntimeError("Signed OC tie interpolation did not satisfy physical mass")
