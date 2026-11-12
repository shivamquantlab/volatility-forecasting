"""Exponentially weighted moving-average variance recursion."""

import numpy as np


def ewma_variance(
    returns: np.ndarray,
    *,
    decay: float = 0.94,
    initial_variance: float = 1e-4,
) -> np.ndarray:
    """Return variance state before each return plus one next-step forecast.

    Output element i is known before observing returns[i]. The final element
    is the forecast after all supplied returns. The default initial variance
    is fixed rather than estimated from the full sample to avoid leakage.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("returns must be a non-empty finite one-dimensional array")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be strictly between 0 and 1")
    initial = float(initial_variance)
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("initial_variance must be finite and positive")
    result = np.empty(values.size + 1, dtype=float)
    result[0] = initial
    try:
        with np.errstate(over="raise", invalid="raise"):
            for index, value in enumerate(values):
                result[index + 1] = decay * result[index] + (1.0 - decay) * value**2
    except FloatingPointError as error:
        raise ValueError("variance recursion exceeds the finite float range") from error
    return result

