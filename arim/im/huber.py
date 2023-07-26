"""
Huber M-estimate in 2D

"""

import numba
import math
import numpy as np
import os

# enable SIMD!
use_parallel = os.environ.get("ARIM_USE_PARALLEL", not numba.core.config.IS_32BITS)
_numba_opts = dict(nogil=True, parallel=use_parallel, fastmath=True, error_model="numpy")


@numba.njit(**_numba_opts)
def _huber_iter(data, tau, x0, y0):
    sum_w = 0.0
    x = 0.0
    y = 0.0
    for i in range(len(data)):
        x_i = data[i][0]
        y_i = data[i][1]
        w_i = min(1, tau / math.sqrt((x0 - x_i) ** 2 + (y0 - y_i) ** 2))
        sum_w += w_i
        x += x_i * w_i
        y += y_i * w_i
    inv_sum_w = 1 / sum_w
    x *= inv_sum_w
    y *= inv_sum_w
    return x, y


@numba.njit(**_numba_opts)
def huber_m_estimate(data, tau, xtol=1e-9, maxiter=600):
    """
    M-estimate of location using Huber's loss

    Reference: 

    Parameters
    ----------
    data : (n, 2)
    tau : threshold
    xtol
    maxiter
    
    Returns
    -------
    xsol
    numiter

    Notes
    -----
    Maronna 2004, chapter "Location and scale": algorithm: (2.75) p 39
    with weights for Huber function: w_tau(x) = min(1, k/abs(x)))  (2.32)
    """
    k = 0  # iteration counter
    l1_update = 2 * xtol  # init only

    # initial guess
    xk = 0.0
    yk = 0.0

    while l1_update > xtol:
        if k >= maxiter:
            raise Exception("max iter reached")

        xk1, yk1 = _huber_iter(data, tau, xk, yk)
        l1_update = abs(xk - xk1) + abs(yk - yk1)
        xk = xk1
        yk = yk1
        k += 1
    return np.array((xk, yk)), k
