"""Runtime helpers shared by callable optimization examples."""

from functools import wraps
import torch


def preserve_default_dtype(function):
    """Restore the caller's Torch default, including when optimization fails."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = torch.get_default_dtype()
        try:
            return function(*args, **kwargs)
        finally:
            torch.set_default_dtype(previous)

    return wrapped


def json_values(value):
    """Convert undefined convergence-window scalars to JSON null."""
    import math

    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
