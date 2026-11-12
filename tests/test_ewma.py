import numpy as np
import pytest

from volatility_forecasting.ewma import ewma_variance


def test_recursion_is_one_step_ahead_and_matches_formula():
    values = np.array([0.1, -0.2])
    state = ewma_variance(values, decay=0.5, initial_variance=0.04)
    np.testing.assert_allclose(state, [0.04, 0.025, 0.0325])
    changed_last = ewma_variance(np.array([0.1, -0.9]), decay=0.5, initial_variance=0.04)
    assert state[1] == changed_last[1]


def test_rejects_invalid_decay_and_zero_initial_variance():
    with pytest.raises(ValueError, match="decay"):
        ewma_variance(np.array([0.1, 0.2]), decay=1.0, initial_variance=0.01)
    with pytest.raises(ValueError, match="positive"):
        ewma_variance(np.array([0.1, 0.2]), initial_variance=0.0)

