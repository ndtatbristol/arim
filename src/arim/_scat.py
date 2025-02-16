"""
Contains common functions to arim.scat and arim.model, to avoid always importing
arim.scat from arim.model.
"""

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _interpolate_scattering_matrix_kernel(scattering_matrix, inc_theta, out_theta):
    # This is a kernel which takes one incident angle and one scattered angle.
    numpoints = scattering_matrix.shape[0]
    dtheta = 2 * np.pi / numpoints

    # Returns indices in [0, ..., numpoints - 1]
    # -pi <-> 0
    # pi - eps <-> numpoints - 1
    # -pi + 2 k pi <-> 0
    inc_theta_idx = int((inc_theta + np.pi) // dtheta % numpoints)
    out_theta_idx = int((out_theta + np.pi) // dtheta % numpoints)

    # Returns the fraction in [0., 1.[ of the distance to the next point to the distance
    # to the last point.
    inc_theta_frac = ((inc_theta + np.pi) % dtheta) / dtheta
    out_theta_frac = ((out_theta + np.pi) % dtheta) / dtheta

    # if we are on the border, wrap around (360° = 0°)
    if inc_theta_idx != (numpoints - 1):
        inc_theta_idx_plus1 = inc_theta_idx + 1
    else:
        inc_theta_idx_plus1 = 0

    if out_theta_idx != (numpoints - 1):
        out_theta_idx_plus1 = out_theta_idx + 1
    else:
        out_theta_idx_plus1 = 0

    # use cardinal direction: sw for south west, etc
    sw = scattering_matrix[out_theta_idx, inc_theta_idx]
    ne = scattering_matrix[out_theta_idx_plus1, inc_theta_idx_plus1]
    se = scattering_matrix[out_theta_idx, inc_theta_idx_plus1]
    nw = scattering_matrix[out_theta_idx_plus1, inc_theta_idx]

    # https://en.wikipedia.org/wiki/Bilinear_interpolation
    f1 = sw + (se - sw) * inc_theta_frac
    f2 = nw + (ne - nw) * inc_theta_frac
    return f1 + (f2 - f1) * out_theta_frac


@numba.guvectorize(
    ["void(f8[:,:], f8[:], f8[:], f8[:])", "void(c16[:,:], f8[:], f8[:], c16[:])"],
    "(s,s),(),()->()",
    nopython=True,
    target="parallel",
    cache=True,
)
def _interpolate_scattering_matrix_ufunc(scattering_matrix, inc_theta, out_theta, res):
    res[0] = _interpolate_scattering_matrix_kernel(
        scattering_matrix, inc_theta[0], out_theta[0]
    )
