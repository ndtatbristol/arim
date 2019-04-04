"""
Geometric median of 2D points using Newton's descent with backtracking line-search
"""

import numba
import math
import numpy as np

# enable SIMD!
_numba_opts = dict(nogil=True, parallel=True, fastmath=True, error_model="numpy")


@numba.njit(**_numba_opts)
def _f(data, z):
    """
    Function to minimise
    
    data : (n, d)
    z: (d, )
    out : 1
    """
    # optimized enough, don't touch!
    out = 0.0
    x = z[0]
    y = z[1]
    for i in range(len(data)):
        out += math.sqrt((x - data[i][0]) ** 2 + (y - data[i][1]) ** 2)
    return out


@numba.njit(**_numba_opts)
def _gradf_and_inv_hessf(data, z):
    """
    Gradient and inverse of the Hessian of f

    Parameters
    ----------
    data : (n, d)
    z: (d, )

    Returns
    -------
    gx : float
        First coefficient of the gradient
    gy : float
        Second coefficient of the gradient
    a11 : float
        ``invhess[0, 0]``
    a12 : float
        ``invhess[0, 1]``
    a22 : float
        ``invhess[1, 1]``
    """
    x = z[0]
    y = z[1]
    gx = 0.0
    gy = 0.0
    a11 = 0.0
    a12 = 0.0
    a22 = 0.0
    for i in range(len(data)):
        tx = x - data[i][0]
        ty = y - data[i][1]
        inv_l2_t = 1 / math.sqrt(tx ** 2 + ty ** 2)
        gx += inv_l2_t * tx
        gy += inv_l2_t * ty
        inv_l2_t3 = inv_l2_t ** 3
        a11 += inv_l2_t - inv_l2_t3 * tx ** 2
        a12 -= (tx * ty) * inv_l2_t3
        a22 += inv_l2_t - inv_l2_t3 * ty ** 2

    # Calculate invert:
    invdet = 1 / (a11 * a22 - a12 * a12)

    return gx, gy, a22 * invdet, -a12 * invdet, a11 * invdet


@numba.njit(**_numba_opts)
def _backtracking_line_search(data, x, gradval, p, rho, c):
    """
    Backtracking line search
    
    Find largest alpha that satisfies the sufficient decrease criterion::
        
        f(x + alpha * p) <= f(x) + c * alpha * dot(grad(f)(x), p)
        
    using backtracking method described in algorithm 3.1 p 37 from Nocedal & Wright.
    
    Parameters
    ----------
    
    x : ndarray
        Parameter of f
    gradval : ndarray
        grad(f)(x)
    p : ndarray
        Descent direction vector 
    rho : 
        contraction factor, 0 < p < 1
    c : float
        factor, 0 < c < 1.
        Recommended value: 10^-4 (Nocebal & Wright p 33)
        
    Returns
    -------
    alpha : float
        Maximum alpha which satisfies the sufficient decrease criterion

    """
    maxiter = 1000

    alpha = 1.0

    fval = _f(data, x)

    for k in range(maxiter):
        if _f(data, (x[0] + alpha * p[0], x[1] + alpha * p[1])) > (
            fval + c * alpha * (gradval[0] * p[0] + gradval[1] * p[1])
        ):
            # new alpha:
            alpha = alpha * rho
        else:
            return alpha
    else:
        raise Exception("cannot find suitable alpha")


@numba.njit(**_numba_opts)
def geomed(data, xtol=1e-5, maxiter=200, c=1e-4, rho=0.5):
    """
    Calculate geometric median using Newton's descent
    
    Parameters
    ----------
    data : (n, 2)
    xtol
    maxiter
    c
    rho
    
    Returns
    -------
    xsol
    numiter
    
    """
    # x0 has to be a ndarray
    #    if maxiter is None:
    #        maxiter = 200 * len(x0)

    k = 0  # iteration counter
    l1_update = 2 * xtol  # init only

    # initial guess
    xk = np.array((0.0, 0.0))

    while l1_update > xtol:
        if k >= maxiter:
            raise Exception("max iter reached")

        gx, gy, invh11, invh12, invh22 = _gradf_and_inv_hessf(data, xk)

        # Newton's descent direction:
        # Solve Hess @ p = -gradval
        #        p = np.linalg.solve(hessf(data, xk), -gradval)
        # p = -invh @gradval
        p = (-invh11 * gx - invh12 * gy, -invh12 * gx - invh22 * gy)

        alpha = _backtracking_line_search(data, xk, (gx, gy), p, rho, c)

        # update:
        update = (alpha * p[0], alpha * p[1])
        xk[0] += update[0]
        xk[1] += update[1]
        l1_update = abs(update[0]) + abs(update[1])
        k += 1
    return xk, k
