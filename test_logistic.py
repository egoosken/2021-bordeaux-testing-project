from logistic import logistic_map
import numpy as np
import pytest

@pytest.mark.parametrize("r, x, expected", [[2.2, 0.1, 0.198], [3.4,0.2, 0.544], [1.7,0.75, 0.31875]])
def test_logistic_map(r, x, expected):
    x = logistic_map(r, x)

    assert np.isclose(x, expected)