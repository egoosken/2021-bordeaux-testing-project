from logistic import logistic_map, iterate_f
import numpy as np
import pytest
from numpy.testing import assert_allclose

@pytest.mark.parametrize("r, x, expected", [[2.2, 0.1, 0.198], [3.4,0.2, 0.544], [1.7,0.75, 0.31875]])
def test_logistic_map(r, x, expected):
    x = logistic_map(x, r)

    assert np.isclose(x, expected)

@pytest.mark.parametrize("it, r, x, expected", [[1, 2.2, 0.1, [0.198]], [4, 3.4,0.2, [0.544, 0.843418, 0.449019, 0.841163]], [2, 1.7,0.75, [0.31875,0.369152]]])
def test_iterate_f(it, r, x, expected):

    lst = iterate_f(it, x, r)

    assert np.allclose(lst, expected)

def test_iterate_f_fuzzing():

    num_x_values = 100

    x_values = np.random.random(num_x_values)
    x_values.sort()
    r = 1.5
    it = 100

    end_values = []
    for x in x_values:
        end_values.append(iterate_f(it,x,r)[-1])

    assert_allclose(end_values, [1/3]*num_x_values)
