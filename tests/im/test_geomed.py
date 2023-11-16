from functools import partial

import numdifftools as nd
import numpy as np
import pytest
from scipy import stats

from arim.im import geomed

np.random.seed(123)
dist = stats.multivariate_normal([0.5, 2.0], [[2.0, 0.3], [0.3, 0.5]])
data = dist.rvs(100)
zval = np.array([0.5, 2.0])


def test_gradf_and_inv_hessf():
    f2 = partial(geomed._f, data)
    z = np.array((0.5, 2.0))
    gradf_ref = nd.Gradient(f2)(z)
    gx, gy, h11, h12, h22 = geomed._gradf_and_inv_hessf(data, z)
    gradf_val = (gx, gy)
    np.testing.assert_allclose(gradf_val, gradf_ref)

    hessref = np.linalg.inv(nd.Hessian(f2)(z))
    tol = dict(rtol=1e-4)
    np.testing.assert_allclose(h11, hessref[0, 0], **tol)
    np.testing.assert_allclose(h12, hessref[1, 0], **tol)
    np.testing.assert_allclose(h12, hessref[0, 1], **tol)
    np.testing.assert_allclose(h22, hessref[1, 1], **tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_geomed(dtype):
    data2 = data.copy().astype(dtype)
    geomed.geomed(data2)
