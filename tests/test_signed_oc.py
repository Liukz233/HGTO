import numpy as np
import pytest
from hgto.nonlinear.signed_oc import negative_multiplier_update


def test_linear_volume_surrogate_has_known_optimal_solution():
    old = np.full(4, 0.45)
    gradient = np.array([-1.0, 2.0, 4.0, 8.0])
    result, info = negative_multiplier_update(
        old, gradient, np.ones(4), old - 0.2, old + 0.2, 0.44, np.mean
    )
    np.testing.assert_allclose(result, [0.65, 0.61, 0.25, 0.25], atol=1e-10)
    assert info["volume_multiplier"] == -2.0


def test_equal_cost_linear_branches_preserve_symmetry():
    old = np.full(4, 0.45)
    gradient = np.array([-1.0, 2.0, 2.0, 2.0])
    result, info = negative_multiplier_update(
        old, gradient, np.ones(4), old - 0.2, old + 0.2, 0.45, np.mean
    )
    np.testing.assert_allclose(result, [0.65, *([1.15 / 3] * 3)], atol=1e-10)
    assert info["volume_multiplier"] == -2.0


def test_actual_move_bound_infeasibility_remains_an_error():
    old = np.full(4, 0.45)
    with pytest.raises(RuntimeError, match="move bounds"):
        negative_multiplier_update(old, np.ones(4), np.ones(4), old - 0.2, old + 0.2, 0.9, np.mean)
