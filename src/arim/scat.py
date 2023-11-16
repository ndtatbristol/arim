"""
Scattering functions and helpers

In a nutshell: use :func:`scat_factory` to create a :class:`Scattering2d` object.

Single frequency scattering matrices are defined formally as::

    S_LL[j, i] = scat_func_LL(phi_in[i], phi_out[j], frequency) for i, j in 0..numangles-1

where `phi_in` and `phi_out` are linearly spaced 1d array between `-pi` (included) and `pi`
(excluded), as returned by :func:`make_angles`. NB: ``S_LL[phi_out_idx, phi_in_idx]``

Multiple frequency scattering matrices are defined formally as::

    S_LL[k, j, i] = scat_func_LL(phi_in[i], phi_out[j], frequencies[k])

.. data:: SCAT_KEYS
   :annotation: = frozenset(('LL', 'LT', 'TL', 'TT'))

   Keys for the different kinds of scattering.
   The first letter is the mode of the
   incident wave; the second letter is the mode of the scattered wave.
   In this module, functions that take an argument ``to_compute`` expects a subset of
   these keys.


"""
import abc
import contextlib
import ctypes
import functools
import math
import warnings

import numba
import numpy as np
import scipy.integrate
from numpy.core.umath import cos, pi, sin
from scipy import interpolate
from scipy.special._ufuncs import hankel1, hankel2

from . import _scat, _scat_crack, exceptions, ut

SCAT_KEYS = frozenset(("LL", "LT", "TL", "TT"))


def make_angles(numpoints):
    """Return angles for scattering matrices. Linearly spaced vector in [-pi, pi[."""
    return np.linspace(-np.pi, np.pi, numpoints, endpoint=False)


def make_angles_grid(numpoints):
    """Return angles for scattering matrices as a grid of incident and outgoing angles."""
    theta = make_angles(numpoints)
    inc_theta, out_theta = np.meshgrid(theta, theta, indexing="xy")
    return inc_theta, out_theta


def interpolate_matrix(scattering_matrix):
    """
    Returns a function that takes as input the incident angles and the scattering angles.
    This returned function returns the scattering amplitudes, obtained by bilinear
    interpolation of the scattering matrix.

    Parameters
    ----------
    scattering_matrix : ndarray

    Returns
    -------
    func

    """
    assert scattering_matrix.ndim == 2
    assert scattering_matrix.shape[0] == scattering_matrix.shape[1]

    return functools.partial(
        _scat._interpolate_scattering_matrix_ufunc, scattering_matrix
    )


def interpolate_matrices(scattering_matrices):
    """
    Convert a dictionary containing scattering matrices to a dictionary containing
    functions that interpolate the values of the scattering matrices.

    Parameters
    ----------
    scattering_matrices : dict[str, ndarray]

    Returns
    -------
    dict[str, function]
    """
    return {key: interpolate_matrix(mat) for key, mat in scattering_matrices.items()}


def sdh_2d_scat(
    inc_theta,
    out_theta,
    frequency,
    radius,
    longitudinal_vel,
    transverse_vel,
    min_terms=10,
    term_factor=4,
    to_compute=SCAT_KEYS,
):
    """
    Scattering coefficients for a side-drilled hole in 2D

    The scattered field is given by::

        u_scat(r, theta) = u0 * sqrt(1 / r) * exp(-i k r + i omega i ray) *
                           (sqrt(lambda_L) A(theta) e_r +
                            sqrt(lambda_T) B(theta) e_theta)

    where A(theta) and B(theta) are the scattering coefficients for respectively L and
    T scattered waves and where e_r and e_theta are the two vectors of the cylindrical
    coordinate system.

    The coefficient for LL, LT, TL and TT are obtained from Lopez-Sanchez's paper,
    equations 33, 34, 39, 40. See also Brind's paper.

    Another difference with these papers is the definition of theta. We use the NDT
    convention where pulse-echo corresponds to theta=0. For Brind, Lopez-Sanchez et al.
    pulse-echo corresponds to theta=pi.

    The number of factor in the sum is::

        maxn = max(min_terms, ceil(term_factor * alpha), ceil(term_factor * beta))


    Parameters
    ----------
    inc_theta : ndarray
        Angle in radians. Pulse echo case corresponds to inc_theta = out_theta
    out_theta : ndarray
        Angle in radians.
    frequency : float
    radius : float
    longitudinal_vel : float
    transverse_vel : float
    min_terms : int
    term_factor : int
    to_compute : set
        See :data:`SCAT_KEYS`

    Returns
    -------
    result : dict
        Keys corresponds to 'to_compute' argument. Values have the shape of theta.

    References
    ----------
    [Lopez-Sanchez] Lopez-Sanchez, Ana L., Hak-Joon Kim, Lester W. Schmerr, and Alexander
    Sedov. 2005. ‘Measurement Models and Scattering Models for Predicting the Ultrasonic
    Pulse-Echo Response From Side-Drilled Holes’. Journal of Nondestructive Evaluation 24
    3): 83–96. doi:10.1007/s10921-005-7658-4.

    [Brind] Brind, R. J., J. D. Achenbach, and J. E. Gubernatis. 1984. ‘High-Frequency
    Scattering of Elastic Waves from Cylindrical Cavities’. Wave Motion 6 (1):
    41–60. doi:10.1016/0165-2125(84)90022-2.

    [Zhang] Zhang, Jie, B.W. Drinkwater, and P.D. Wilcox. 2008. ‘Defect Characterization
    Using an Ultrasonic Array to Measure the Scattering Coefficient Matrix’. IEEE
    Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 55 (10): 2254–65.
    doi:10.1109/TUFFC.924.


    """
    theta = out_theta - inc_theta

    if not SCAT_KEYS.issuperset(to_compute):
        raise ValueError(
            f"Valid 'to_compute' arguments are {SCAT_KEYS} (got {to_compute})"
        )

    # wavenumber

    kl = 2 * pi * frequency / longitudinal_vel
    kt = 2 * pi * frequency / transverse_vel

    # Brind eq 2.8
    alpha = kl * radius
    beta = kt * radius
    beta2 = beta * beta

    # sum from n=0 to n=maxn (inclusive)
    # The larger maxn, the better the axppromixation
    maxn = max(
        [int(min_terms), math.ceil(term_factor * alpha), math.ceil(term_factor * beta)]
    )
    n = np.arange(0, maxn + 1)
    n2 = n * n

    # Brind eq 2.8
    epsilon = np.full(n.shape, 2.0)
    epsilon[0] = 1.0

    # Definition of C_n^(i)(x) and D_n^(i)(x)
    # Brind, eq 31
    c1 = lambda x: (n2 + n - beta2 / 2) * hankel1(n, x) - x * hankel1(n - 1, x)
    c2 = lambda x: (n2 + n - beta2 / 2) * hankel2(n, x) - x * hankel2(n - 1, x)
    d1 = lambda x: (n2 + n) * hankel1(n, x) - n * x * hankel1(n - 1, x)
    d2 = lambda x: (n2 + n) * hankel2(n, x) - n * x * hankel2(n - 1, x)
    c1_alpha = c1(alpha)
    c2_alpha = c2(alpha)
    d1_alpha = d1(alpha)
    d2_alpha = d2(alpha)
    c1_beta = c1(beta)
    c2_beta = c2(beta)
    d1_beta = d1(beta)
    d2_beta = d2(beta)

    # in angle
    phi = theta + pi

    # n_phi[i1, ..., id, j] := phi[i1, ..., id] * n[j]
    n_phi = np.einsum("...,j->...j", phi, n)
    cos_n_phi = cos(n_phi)
    sin_n_phi = sin(n_phi)
    del n_phi

    result = dict()

    # NB: sqrt(2j/(pi * k)) = sqrt(i) / pi

    if "LL" in to_compute:
        # Lopez-Sanchez eq (29)
        A_n = (
            1j
            / (2 * alpha)
            * (
                1
                + (c2_alpha * c1_beta - d2_alpha * d1_beta)
                / (c1_alpha * c1_beta - d1_alpha * d1_beta)
            )
        )

        # Brind (2.9) without:
        #   - u0, the amplitude of the incident wave,
        #   - 'exp(i k r)'  which in Bristol LTI model is in the propagation term,
        #   - 'lambda/sqrt(r)' which in Bristol LTI model is the 2D beamspread term,
        #
        # This is consistent with Lopez-Sanchez eq (33).
        #
        # NB: exp(i pi /4) = sqrt(i)
        #
        # The line:
        #   out = np.einsum('...j,j->...', n_phi, coeff)
        # gives the result:
        #   out[i1, ..., id] = sum_j (n_phi[i1, ..., id, j] * coeff[j])
        r = (np.sqrt(1j) / pi * alpha) * np.einsum(
            "...j,j->...", cos_n_phi, epsilon * A_n
        )

        result["LL"] = r

    if "LT" in to_compute:
        # Lopez-Sanchez eq (30)
        B_n = (
            2
            * n
            / (pi * alpha)
            * ((n2 - beta2 / 2 - 1) / (c1_alpha * c1_beta - d1_alpha * d1_beta))
        )

        # Lopez-Sanchez (34)
        # Warning: there is a minus sign in Brind (2.10). We trust LS here.
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * beta) * np.einsum(
            "...j,j->...", sin_n_phi, epsilon * B_n
        )
        result["LT"] = r

    if "TL" in to_compute:
        # Lopez-Sanchez eq (41)
        A_n = (
            2
            * n
            / (pi * beta)
            * (n2 - beta2 / 2 - 1)
            / (c1_alpha * c1_beta - d1_alpha * d1_beta)
        )

        # Lopez-Sanchez eq (39)
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * alpha) * np.einsum(
            "...j,j->...", sin_n_phi, epsilon * A_n
        )
        result["TL"] = r

    if "TT" in to_compute:
        # Lopez-Sanchez eq (42)
        B_n = (
            1j
            / (2 * beta)
            * (
                1
                + (c2_beta * c1_alpha - d2_beta * d1_alpha)
                / (c1_alpha * c1_beta - d1_alpha * d1_beta)
            )
        )

        # Lopez-Sanchez eq (40)
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * beta) * np.einsum(
            "...j,j->...", cos_n_phi, epsilon * B_n
        )
        result["TT"] = r

    return result


def crack_2d_scat(
    inc_theta,
    out_theta,
    frequency,
    crack_length,
    longitudinal_vel,
    transverse_vel,
    density,
    nodes_per_wavelength=20,
    assume_safe_for_opt=False,
    to_compute={"LL", "LT", "TL", "TT"},
):
    """
    Scattering matrix of the centre of a crack.

    The model assumes there is little interaction with the walls, ie the crack is deep
    enough (not a surface crack). Resolution with a Galerkin method.

    Parameters
    ----------
    inc_theta : ndarray
    out_theta : ndarray
    frequency : float
    crack_length : float
    longitudinal_vel : float
    transverse_vel : float
    density : float
    nodes_per_wavelength : int
        Default: 20
    assume_safe_for_opt : bool
        Default: False. If True, use an optimised implementation which requires that
        ``inc_theta[i, j] = inc_theta[i, 0]`` for all i and j. If False, use slower but
        more general implementation. Warning: no check is  performed to ensure the
        assumption holds.
    to_compute : set[str]
        See :data:`SCAT_KEYS`

    Returns
    -------
    dict of ndarray

    Notes
    -----
    Original code: function ``fn_s_matrices_for_crack_2d`` by Alexander Velichko and
    Paul D. Wilcox from the University of Bristol NDT library.
    Python code by Nicolas Budyn.

    References
    ----------
    [Glushkov] Glushkov, Evgeny, Natalia Glushkova, Alexander Ekhlakov, and
    Elena Shapar. 2006. ‘An Analytically Based Computer Model for Surface
    Measurements in Ultrasonic Crack Detection’. Wave Motion 43 (6): 458–73.
    doi:10.1016/j.wavemoti.2006.03.002.

    Unpublished work from Alexander Velichko

    """
    valid_keys = {"LL", "LT", "TL", "TT"}

    if not valid_keys.issuperset(to_compute):
        raise ValueError(
            f"Valid 'to_compute' arguments are {valid_keys} (got {to_compute})"
        )

    final_broadcast = np.broadcast(inc_theta, out_theta)
    if final_broadcast.ndim > 2:
        raise NotImplementedError

    inc_theta, out_theta = np.atleast_2d(inc_theta, out_theta)
    inc_theta, out_theta = np.broadcast_arrays(inc_theta, out_theta)
    # Explicitly mark the broadcasted arrays as read-only, to prevent
    # a FutureWarning and anticipate the future behaviour of numpy.
    inc_theta.flags.writeable = False
    out_theta.flags.writeable = False
    comp_broadcast = np.broadcast(inc_theta, out_theta)

    v_L = longitudinal_vel
    v_T = transverse_vel
    use_incident_L = "LL" in to_compute or "LT" in to_compute
    use_incident_T = "TL" in to_compute or "TT" in to_compute

    lambda_L = v_L / frequency
    xi2 = 2 * pi * frequency / v_T
    xi = v_T / v_L

    # mesh definition
    num_nodes = int(np.ceil(crack_length / lambda_L * nodes_per_wavelength))
    p = 0.113_340_798_6  # magic constant
    h_nodes = crack_length / (num_nodes + 2 * p)
    x_nodes = np.arange(num_nodes) * h_nodes + (
        h_nodes * (1 / 2 + p) - crack_length / 2
    )

    # get matrices for the linear system
    A_x = _scat_crack.A_x(xi, xi2, h_nodes, num_nodes)
    A_z = _scat_crack.A_z(xi, xi2, h_nodes, num_nodes)

    # init output (always in 2D)
    S_LL = np.zeros(comp_broadcast.shape, np.complex128, order="F")
    S_LT = np.zeros(comp_broadcast.shape, np.complex128, order="F")
    S_TL = np.zeros(comp_broadcast.shape, np.complex128, order="F")
    S_TT = np.zeros(comp_broadcast.shape, np.complex128, order="F")

    if assume_safe_for_opt:
        inc_theta_vect = inc_theta[0]
        matrices = _scat_crack.crack_2d_scat_matrix(
            inc_theta_vect,
            out_theta,
            v_L,
            v_T,
            density,
            frequency,
            use_incident_L,
            use_incident_T,
            x_nodes,
            h_nodes,
            A_x,
            A_z,
            S_LL,
            S_LT,
            S_TL,
            S_TT,
        )
    else:
        matrices = _scat_crack.crack_2d_scat_general(
            inc_theta,
            out_theta,
            v_L,
            v_T,
            density,
            frequency,
            use_incident_L,
            use_incident_T,
            x_nodes,
            h_nodes,
            A_x,
            A_z,
            S_LL,
            S_LT,
            S_TL,
            S_TT,
        )

    # reshape output to the requested shape
    final_matrices = [m.reshape(final_broadcast.shape) for m in matrices]

    return dict(zip(("LL", "LT", "TL", "TT"), final_matrices))


@numba.cfunc("f8(f8, voidptr)", cache=True)
def _crack_tip_integrand(x, data):
    alpha, k_p, k_s = numba.carray(data, 3, dtype=numba.float64)
    return -math.atan(
        (4 * x**2 * math.sqrt(x**2 - k_p**2) * math.sqrt(k_s**2 - x**2))
        / (2 * x**2 - k_s**2) ** 2
    ) / ((x + alpha) * math.pi)


def _crack_tip_k_plus_integral(alpha, k_p, k_s, eps=1e-3, **quad_kwargs):
    # integral in equation 2b
    # includes the -1/pi factor
    integrand_params = np.array([alpha, k_p, k_s])
    integrand_params_ptr = ctypes.cast(integrand_params.ctypes, ctypes.c_void_p)
    integrand_c = scipy.LowLevelCallable(
        _crack_tip_integrand.ctypes, integrand_params_ptr
    )

    # singularity at x = -alpha = beta
    beta = -alpha

    assert k_p < k_s
    assert eps > 0

    if k_p < beta < k_s:
        # enforce k_p <= beta - eps/2
        #   and  beta + eps/2 <= k_s
        max_allowable_eps = min(2 * (beta - k_p), 2 * (k_s - beta))
        assert max_allowable_eps > 0
        if eps >= max_allowable_eps:
            # overwrite eps if too big
            eps = max_allowable_eps / 2

        integral_left, error_left = scipy.integrate.quad(
            integrand_c, k_p, beta - eps / 2, **quad_kwargs
        )
        integral_right, error_right = scipy.integrate.quad(
            integrand_c, beta + eps / 2, k_s, **quad_kwargs
        )

        integral = integral_left + integral_right
        error = np.sqrt(error_left**2 + error_right**2)
    else:
        integral, error = scipy.integrate.quad(integrand_c, k_p, k_s)

    return integral, error


def _crack_tip_k_plus_integral_arr(alpha_arr, k_p, k_s, **quad_kwargs):
    out = np.empty_like(alpha_arr, float)
    it = np.nditer([alpha_arr, out], op_flags=(["readonly"], ["writeonly", "allocate"]))
    for alpha, res in it:
        res[...] = _crack_tip_k_plus_integral(alpha, k_p, k_s, **quad_kwargs)[0]
    return it.operands[1]


def _crack_tip_k_plus(alpha_arr, k_p, k_s, **quad_kwargs):
    return np.exp(_crack_tip_k_plus_integral_arr(alpha_arr, k_p, k_s, **quad_kwargs))


def crack_tip_2d(
    inc_theta,
    out_theta,
    longitudinal_vel,
    transverse_vel,
    rayleigh_vel=None,
    to_compute=("LL", "LT", "TL", "TT"),
    **quad_kwargs,
):
    """
    Analytical model of the diffraction of elastic waves by a crack tip. The crack length is infinite.

    Parameters
    ----------
    inc_theta
    out_theta
    longitudinal_vel
    transverse_vel
    rayleigh_vel : float or None
        If None, use :func:`arim.ut.rayleigh_vel`
    to_compute
    quad_kwargs

    Returns
    -------
    res

    Notes
    -----
    [Ogilvy83] Ogilvy, J. A., and J. A. G. Temple. 1983. ‘Diffraction of Elastic Waves by Cracks: Application to
    Time-of-Flight Inspection’. Ultrasonics 21 (6):259–69. https://doi.org/10.1016/0041-624X(83)90058-6.

    """
    res = dict()

    # use paper notations
    # 1e7 is a scaling factor. The theoretical result does not depend on the frequency
    # but the numerical result because of finite numerical precision.
    k_p = 1e7 / longitudinal_vel
    k_s = 1e7 / transverse_vel

    if rayleigh_vel is None:
        rayleigh_vel = ut.rayleigh_vel(longitudinal_vel, transverse_vel)
    k_0 = 1e7 / rayleigh_vel

    k_p2 = k_p**2
    k_s2 = k_s**2

    beta = inc_theta
    theta = out_theta

    e_ipi4 = np.sqrt(1j)  # exp(1j pi / 4)

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    pi = np.pi

    cos_theta = cos(theta)
    cos_beta = cos(beta)

    if "LT" in to_compute or "TT" in to_compute:
        k_plus_ks_cos_theta = _crack_tip_k_plus(
            -k_s * cos_theta, k_p, k_s, **quad_kwargs
        )
    else:
        k_plus_ks_cos_theta = None
    if "LL" in to_compute or "TL" in to_compute:
        k_plus_kp_cos_theta = _crack_tip_k_plus(
            -k_p * cos_theta, k_p, k_s, **quad_kwargs
        )
    else:
        k_plus_kp_cos_theta = None
    if "TL" in to_compute or "TT" in to_compute:
        k_plus_ks_cos_beta = _crack_tip_k_plus(-k_s * cos_beta, k_p, k_s, **quad_kwargs)
    else:
        k_plus_ks_cos_beta = None
    if "LL" in to_compute or "LT" in to_compute:
        k_plus_kp_cos_beta = _crack_tip_k_plus(-k_p * cos_beta, k_p, k_s, **quad_kwargs)
    else:
        k_plus_kp_cos_beta = None

    if "LL" in to_compute:
        # Gp(theta, beta)
        res["LL"] = (
            e_ipi4
            * sin(beta / 2)
            * (
                sin(theta / 2)
                * (2 * k_p2 * cos_beta**2 - k_s2)
                * (2 * k_p2 * cos_theta**2 - k_s2)
                + 2
                * k_p**3
                * cos(beta / 2)
                * cos_beta
                * sin(2 * theta)
                * sqrt(k_s - k_p * cos_theta)
                * sqrt(k_s - k_p * cos_beta)
            )
            / (
                2
                * pi
                * (k_s2 - k_p2)
                * (cos_theta + cos_beta)
                * (k_0 - k_p * cos_theta)
                * (k_0 - k_p * cos_beta)
                * k_plus_kp_cos_theta
                * k_plus_kp_cos_beta
            )
        )
    if "LT" in to_compute:
        # G_s(theta, beta)
        res["LT"] = (
            e_ipi4
            * sqrt(k_p / k_s)
            * (
                k_s2
                * sin(beta / 2)
                * (
                    sqrt(2 * k_p)
                    * (2 * k_p2 * cos_beta**2 - k_s2)
                    * sin(2 * theta)
                    * sqrt((k_p - k_s * cos_theta).astype(complex))
                    - 4
                    * k_p2
                    * sqrt(2 * k_s)
                    * cos(beta / 2)
                    * cos_beta
                    * sin(theta / 2)
                    * cos(2 * theta)
                    * sqrt(k_s - k_p * cos_beta)
                )
            )
            / (
                4
                * pi
                * (k_s2 - k_p2)
                * (k_s * cos_theta + k_p * cos_beta)
                * (k_0 - k_s * cos_theta)
                * (k_0 - k_p * cos_beta)
                * k_plus_ks_cos_theta
                * k_plus_kp_cos_beta
            )
        )
    if "TL" in to_compute:
        # F_p(theta, beta)
        res["TL"] = (
            e_ipi4
            * sqrt(k_s / k_p)
            * (
                k_s2
                * sin(beta / 2)
                * (
                    -k_p2
                    * sqrt(2 * k_s)
                    * cos(2 * beta)
                    * sin(2 * theta)
                    * sqrt(k_s - k_p * cos(theta))
                    + 4
                    * sqrt(2 * k_p)
                    * cos(beta / 2)
                    * cos(beta)
                    * sin(theta / 2)
                    * (2 * k_p2 * cos_theta**2 - k_s2)
                    * sqrt((k_p - k_s * cos_beta).astype(complex))
                )
            )
            / (
                4
                * pi
                * (k_s2 - k_p2)
                * (k_p * cos_theta + k_s * cos_beta)
                * (k_0 - k_p * cos_theta)
                * (k_0 - k_s * cos_beta)
                * k_plus_kp_cos_theta
                * k_plus_ks_cos_beta
            )
        )
    if "TT" in to_compute:
        # F_s(theta, beta)
        res["TT"] = (
            e_ipi4
            * k_s**3
            * sin(beta / 2)
            * (
                k_s * cos(2 * beta) * cos(2 * theta) * sin(theta / 2)
                + 2
                * cos(beta / 2)
                * cos(beta)
                * sin(2 * theta)
                * sqrt((k_p - k_s * cos_theta).astype(complex))
                * sqrt((k_p - k_s * cos_beta).astype(complex))
            )
            / (
                2
                * pi
                * (k_s2 - k_p2)
                * (cos_theta + cos_beta)
                * (k_0 - k_s * cos_theta)
                * (k_0 - k_s * cos_beta)
                * k_plus_ks_cos_theta
                * k_plus_ks_cos_beta
            )
        )

    return res


def rotate_matrix(scat_matrix, phi):
    """
    Return the scattering matrix S' of the scatterer rotated by an angle phi,
    knowing the scattering matrix S of the unrotated scatterer::

        S'(theta_1, theta_2) = S(theta_1 - phi, theta_2 - phi)

    Use FFT internally.

    Parameters
    ----------
    scat_matrix : ndarray
        Shape (numangles, numangles)
    phi : float
        Defect's rotation angle in radian.

    Returns
    ------
    roated_scat_matrix: ndarray
        Shape : (numangles, numangles)

    """
    n, _ = scat_matrix.shape

    freq = np.fft.fftfreq(n, 2 * np.pi / n)

    freq_x, freq_y = np.meshgrid(freq, freq, indexing="ij")

    freqshift = np.exp(-2j * np.pi * (freq_x + freq_y) * phi)
    scat_matrix_f = np.fft.fft2(scat_matrix)
    return np.fft.ifft2(freqshift * scat_matrix_f)


def rotate_matrices(scat_matrices, phi):
    """
    Call :func:`rotate_matrix` on each matrix in a dict.

    Parameters
    ----------
    scat_matrices : dict
    phi : float

    Returns
    -------
    dict

    """
    return {
        scat_key: rotate_matrix(scat_matrix, phi)
        for scat_key, scat_matrix in scat_matrices.items()
    }


def _partial_one_scat_key(scat_func, scat_key, *args, **kwargs):
    """
    Returns a dict of functions.

        >>> assert scat_func(x, y, z, to_compute=['LT'])['LT'] == _partial_one_scat_key(scat_func, 'LT', z)(x, y)

    """
    # Remark: do not try to replace this by a lambda function, a proper closure is needed
    # here.
    # See https://stackoverflow.com/questions/3252228/python-why-is-functools-partial-necessary
    # Inspired by https://docs.python.org/3/library/functools.html#functools.partial
    to_compute = {scat_key}

    def new_scat_func(*fargs, **fkeywords):
        newkeywords = kwargs.copy()
        newkeywords.update(fkeywords)
        return scat_func(*args, *fargs, to_compute=to_compute, **newkeywords)[scat_key]

    return new_scat_func


def scat_factory(kind, material, *args, **kwargs):
    """
    Creates a Scattering2d object in a simple way

    Parameters
    ----------
    kind : str
    material : Material
    args
    kwargs

    Returns
    -------
    Scattering2d

    Examples
    --------
    >>> material = arim.Material(6300., 3120., 2700., 'solid', metadata={'long_name': 'Aluminium'})

    Creating the scattering object:

    >>> scat_obj = scat_factory('file', material, 'scattering_data.mat')

    >>> scat_obj = scat_factory('crack_centre', material, crack_length=2.0e-3)

    >>> scat_obj = scat_factory('crack_tip', material)

    >>> scat_obj = scat_factory('sdh', material, radius=0.5e-3)

    >>> scat_obj = scat_factory('point', material) # unphysical, debug only

    Each ``scat_obj`` is a :class:`Scattering2d` object.

    """
    kind = kind.lower()  # ignore case
    if kind == "file":
        from . import io

        return io.scat.load_scat(*args, **kwargs)
    elif kind == "crack_centre":
        return CrackCentreScat(
            *args,
            longitudinal_vel=material.longitudinal_vel,
            transverse_vel=material.transverse_vel,
            density=material.density,
            **kwargs,
        )
    elif kind == "crack_tip":
        return CrackTipScat(
            material.longitudinal_vel, material.transverse_vel, *args, **kwargs
        )
    elif kind == "sdh":
        return SdhScat(
            *args,
            longitudinal_vel=material.longitudinal_vel,
            transverse_vel=material.transverse_vel,
            **kwargs,
        )
    elif kind == "point":
        return PointSourceScat(
            material.longitudinal_vel, material.transverse_vel, *args, **kwargs
        )
    else:
        raise NotImplementedError(f"no strategy for kind='{kind}'")


class Scattering2d(abc.ABC):
    """
    Base object for computing the scattering functions in 2D.

    Examples
    --------
    >>> material = arim.Material(6300., 3120., 2700., 'solid', metadata={'long_name': 'Aluminium'})
    >>> scat_obj = scat_factory('sdh', material, radius=0.5e-3)

    A :class:`Scattering2d` can be used a function of the incident angles, the scattered
    angles and the frequency:

    >>> inc_theta = np.deg2rad([0., 0., 0.])
    >>> out_theta = np.deg2rad([0., 10., 20])
    >>> frequency = 5e6  # Hz
    >>> result = scat_obj(inc_theta, out_theta, frequency)

    ``result`` is a dict with keys 'LL', 'LT', 'TL', 'TT'. Each value of the dict is an
    array of shape (3, ).

    >>> result2 = scat_obj(inc_theta, out_theta, frequency, to_compute=['LL'])


    ``result2`` is a dict which contains the key 'LL'. Use this feature to reduce the amount
    of computation. Depending on how the function is written, other keys may be returned.

    To generate scattering matrices:

    >>> numangles = 10  # number of angles between -pi (included) and +pi (excluded)
    >>> single_freq_matrices = scat_obj.as_single_freq_matrices(numangles, frequency)

    ``single_freq_matrices['LL']`` is here a 2d array of shape (10, 10).

    >>> frequencies = [1e6, 2e6, 3e6, 4e6, 5e6]  # Hz
    >>> multi_freq_matrices = scat_obj.as_multi_freq_matrices(numangles, frequencies)

    ``multi_freq_matrices['LL']`` is here a 3d array of shape (5, 10, 10).

    For convenience, functions that return an array instead of a dict of array are
    provided.

    >>> func_dict = scat_obj.as_freq_angles_funcs()
    >>> scat_LL_func = func_dict['LL']
    >>> scat_LL_func(phi_in, phi_out, frequency)
    ...  # return an array

    """

    @abc.abstractmethod
    def __call__(self, inc_theta, out_theta, frequency, to_compute=SCAT_KEYS):
        """
        Returns the scattering values for the given angles and frequency.

        Parameters
        ----------
        inc_theta : ndarray
        out_theta : ndarray
        frequency : float
        to_compute : set[str]
            See :data:`SCAT_KEYS`

        Returns
        -------
        scat_values : dict[ndarray]
            Keys: at least the ones given in ``to_compute``.

        """

    def as_freq_angles_funcs(self):
        """
        Returns a dict of scattering functions that take as input the incident angle,
        the outgoing angle and the frequency.
        """
        scat_funcs = {}

        for scat_key in SCAT_KEYS:
            scat_funcs[scat_key] = _partial_one_scat_key(self, scat_key)
        return scat_funcs

    def as_angles_funcs(self, frequency):
        """
        Returns a dict of scattering functions that take as input the incident angle
        and the outgoing angle.
        """
        scat_funcs = {}

        for scat_key in SCAT_KEYS:
            scat_funcs[scat_key] = _partial_one_scat_key(
                self, scat_key, frequency=frequency
            )
        return scat_funcs

    def as_multi_freq_matrices(self, frequencies, numangles, to_compute=SCAT_KEYS):
        """
        Returns scattering matrices at different frequencies.

        Parameters
        ----------
        frequency : ndarray
            Shape: (numfreq, )
        numangles : int
        to_compute
            See :data:`SCAT_KEYS`

        Returns
        -------
        dict[str, ndarray]
            Shape of each matrix: ``(numfreq, numpoints, numpoints)``

        """
        inc_theta, out_theta = make_angles_grid(numangles)

        out = None

        for i, frequency in enumerate(frequencies):
            matrices = self(inc_theta, out_theta, frequency, to_compute)
            if out is None:
                # Late initialisation for getting the datatype of matrices
                out = {
                    scat_key: np.zeros(
                        (len(frequencies), numangles, numangles),
                        matrices[scat_key].dtype,
                    )
                    for scat_key in to_compute
                }
            for scat_key in to_compute:
                out[scat_key][i] = matrices[scat_key]
        return out

    def as_single_freq_matrices(self, frequency, numangles, to_compute=SCAT_KEYS):
        """
        Returns scattering matrices at a given frequency.

        Parameters
        ----------
        frequency : float
        numangles : int
        to_compute : set[str]


        Returns
        -------
        dict[str, ndarray]
            Shape of each matrix: ``(numpoints, numpoints)``

        """
        inc_theta, out_theta = make_angles_grid(numangles)
        return self(inc_theta, out_theta, frequency, to_compute)

    def as_multi_freq_matrices_obj(self, frequencies, numangles, to_compute=SCAT_KEYS):
        """
        Returns scattering matrices at different frequencies as a ScatFromData object.

        Parameters
        ----------
        frequencies : ndarray
        numangles : int
        to_compute : set[str]

        Returns
        -------
        ScatFromData

        """
        matrices = self.as_multi_freq_matrices(frequencies, numangles, to_compute)
        return ScatFromData.from_dict(frequencies, matrices)


class Scattering2dFromFunc(Scattering2d):
    """
    Wrapper for scattering functions that take as three first arguments 'inc_theta',
    'out_theta' and 'frequency', and that accepts an argument 'to_compute'.

    To wrap a scattering function with this class:

    - create a class that inherit this class,
    - set the wrapped function as the '_scat_func' attribute,
    - populate the ``_scat_kwargs`` attribute with the extra arguments to pass to ``_scat_func``,
      ie any argument but ``inc_theta``, ``out_theta``, ``frequency`` and ``to_compute``.

    This class is abstract.
    """

    _scat_kwargs = None  # placeholder

    @staticmethod
    @abc.abstractmethod
    def _scat_func(*args, **kwargs):
        """Wrapped function."""
        raise NotImplementedError

    def __call__(self, inc_theta, out_theta, frequency, to_compute=SCAT_KEYS):
        return self._scat_func(
            inc_theta, out_theta, frequency, to_compute=to_compute, **self._scat_kwargs
        )

    def __repr__(self):
        # Returns something like 'Scattering(x=1, y=2)'
        arg_str = ", ".join([f"{key}={val}" for key, val in self._scat_kwargs.items()])
        return self.__class__.__qualname__ + "(" + arg_str + ")"


class SdhScat(Scattering2dFromFunc):
    """
    Scattering for side-drilled hole

    This class provides the :class:`Scattering2d` interface for :func:`sdh_2d_scat`.
    """

    _scat_func = staticmethod(sdh_2d_scat)

    def __init__(
        self, radius, longitudinal_vel, transverse_vel, min_terms=10, term_factor=4
    ):
        self._scat_kwargs = dict(
            radius=radius,
            longitudinal_vel=longitudinal_vel,
            transverse_vel=transverse_vel,
            min_terms=min_terms,
            term_factor=term_factor,
        )


class CrackCentreScat(Scattering2dFromFunc):
    """
    Scattering of a crack at its centre.

    This class provides the :class:`Scattering2d` interface for :func:`crack_2d_scat`.

    """

    _scat_func = staticmethod(crack_2d_scat)

    def __init__(
        self,
        crack_length,
        longitudinal_vel,
        transverse_vel,
        density,
        nodes_per_wavelength=20,
    ):
        self._scat_kwargs = dict(
            crack_length=crack_length,
            longitudinal_vel=longitudinal_vel,
            transverse_vel=transverse_vel,
            density=density,
            nodes_per_wavelength=nodes_per_wavelength,
        )
        self._in_matrix_calculation = False

    def __call__(self, inc_theta, out_theta, frequency, to_compute=SCAT_KEYS):
        return self._scat_func(
            inc_theta,
            out_theta,
            frequency,
            to_compute=to_compute,
            assume_safe_for_opt=self._in_matrix_calculation,
            **self._scat_kwargs,
        )

    @contextlib.contextmanager
    def _scat_matrix_calculation(self):
        """context manager to flag we are doing a scattering matrix calculation"""
        self._in_matrix_calculation = True
        yield
        self._in_matrix_calculation = False

    def as_multi_freq_matrices(self, frequencies, numangles, to_compute=SCAT_KEYS):
        with self._scat_matrix_calculation():
            return super().as_multi_freq_matrices(frequencies, numangles, to_compute)

    def as_single_freq_matrices(self, frequency, numangles, to_compute=SCAT_KEYS):
        with self._scat_matrix_calculation():
            assert self._in_matrix_calculation
            result = super().as_single_freq_matrices(frequency, numangles, to_compute)
        assert not self._in_matrix_calculation
        return result


class PointSourceScat(Scattering2dFromFunc):
    """
    Scattering of an unphysical point source. For debug only.

    For any incident and scattered angles, the scattering is defined as::

        S_LL = 1
        S_LT = v_L / v_T
        S_TL = -v_T / v_L
        S_TT = 1

    Remark: these scattering functions could have been defined as::

        S_LL = a
        S_LT = b * v_L / v_T
        S_TL = -b * v_T / v_L
        S_TT = c

    with any a, b, c. These coefficients were chosen arbitrarily in the present function.
    Therefore drawing quantitative conclusions from a model using this function must be
    done with care.

    """

    @staticmethod
    def _scat_func(
        phi_in,
        phi_out,
        frequency,
        longitudinal_vel,
        transverse_vel,
        to_compute=SCAT_KEYS,
    ):
        shape = np.broadcast(phi_in, phi_out).shape

        v_L = longitudinal_vel
        v_T = transverse_vel

        out = dict()
        if "LL" in to_compute:
            out["LL"] = np.full(shape, 1.0)
        if "LT" in to_compute:
            out["LT"] = np.full(shape, v_L / v_T)
        if "TL" in to_compute:
            out["TL"] = np.full(shape, -v_T / v_L)
        if "TT" in to_compute:
            out["TT"] = np.full(shape, 1.0)
        return out

    def __init__(self, longitudinal_vel, transverse_vel):
        self._scat_kwargs = dict(
            longitudinal_vel=longitudinal_vel, transverse_vel=transverse_vel
        )


class CrackTipScat(Scattering2dFromFunc):
    """
    Crack tip diffraction

    Wrapper for :func:`crack_tip_2d`
    """

    _scat_func = staticmethod(crack_tip_2d)

    def __call__(self, inc_theta, out_theta, frequency, to_compute=SCAT_KEYS):
        # drop frequency argument
        return self._scat_func(
            inc_theta, out_theta, to_compute=to_compute, **self._scat_kwargs
        )

    def __init__(
        self, longitudinal_vel, transverse_vel, rayleigh_vel=None, **quad_args
    ):
        self._scat_kwargs = dict(
            longitudinal_vel=longitudinal_vel,
            transverse_vel=transverse_vel,
            rayleigh_vel=rayleigh_vel,
            **quad_args,
        )


class ScatFromData(Scattering2d):
    """
    Scattering functions from a set of data.

    This class provides the regular :class:`Scattering2d` interface for a dataset of
    scattering matrices. The typical usage is to use this class for wrapping
    data generated with another software in arim.

    By default, this class uses a **linear interpolation** for generating values
    at new angles.

    By default, this class uses a **linear interpolation** for generating values at new
    frequencies when at least two frequencies point are given. Outside the frequency range
    of the data, values are extrapolated. This can be changed by modifying
    :attr:`interp_freq_kwargs`, the arguments passed to `scipy.interpolate.interp1d`.
    If the data contains only one frequency, this data will be used at any new frequency.

    Attributes
    ----------
    numfreq : int
    numangles : int
    orig_matrices : dict
        Shape of each value: (numfreq, numangles, numangles)
    frequencies : ndarray
        1d array
    interp_freq_kwargs : dict
        Passed to ``scipy.interpolate.interp1d``.
        Default: ``bounds_error=False, fill_value='extrapolate'``


    """

    def __init__(
        self,
        frequencies,
        scat_matrix_LL=None,
        scat_matrix_LT=None,
        scat_matrix_TL=None,
        scat_matrix_TT=None,
    ):
        frequencies = np.asarray(frequencies)
        if frequencies.ndim == 0:
            frequencies = np.array([frequencies])
        elif frequencies.ndim > 1:
            raise ValueError("'frequencies' must be 1d")
        numfreq = len(frequencies)

        shapes = {
            np.shape(scat_matrix_LL) if scat_matrix_LL is not None else None,
            np.shape(scat_matrix_LT) if scat_matrix_LT is not None else None,
            np.shape(scat_matrix_TL) if scat_matrix_TL is not None else None,
            np.shape(scat_matrix_TT) if scat_matrix_TT is not None else None,
        }
        shapes.discard(None)

        if len(shapes) == 0:
            raise ValueError("at least one scattering matrix must be passed")
        elif len(shapes) > 1:
            raise ValueError("scattering matrices must have the same shape")

        shape = shapes.pop()
        wrong_shape_err = (
            "Scattering matrices' shape must be (numfreq, numangles, numangles)"
        )
        if len(shape) != 3:
            raise ValueError(wrong_shape_err)
        if shape[1] != shape[2]:
            raise ValueError(wrong_shape_err)
        if shape[0] != numfreq:
            raise ValueError(wrong_shape_err)

        self.frequencies = frequencies
        self.numfreq = numfreq
        self.numangles = shape[1]

        self.orig_matrices = dict()
        if scat_matrix_LL is not None:
            self.orig_matrices["LL"] = np.ascontiguousarray(scat_matrix_LL)
        if scat_matrix_LT is not None:
            self.orig_matrices["LT"] = np.ascontiguousarray(scat_matrix_LT)
        if scat_matrix_TL is not None:
            self.orig_matrices["TL"] = np.ascontiguousarray(scat_matrix_TL)
        if scat_matrix_TT is not None:
            self.orig_matrices["TT"] = np.ascontiguousarray(scat_matrix_TT)

        self.interp_freq_kwargs = dict(bounds_error=False, fill_value="extrapolate")

    @classmethod
    def from_dict(cls, frequencies, scat_matrix_dict):
        """
        Alternative constructor: takes as input a dict of scattering matrices
        (keys: LL, LT, TL, TT)

        Parameters
        ----------
        frequencies
        scat_matrix_dict

        Returns
        -------
        obj : ScatFromData

        """
        scat_matrix_LL = scat_matrix_dict.get("LL")
        scat_matrix_LT = scat_matrix_dict.get("LT")
        scat_matrix_TL = scat_matrix_dict.get("TL")
        scat_matrix_TT = scat_matrix_dict.get("TT")
        return cls(
            frequencies, scat_matrix_LL, scat_matrix_LT, scat_matrix_TL, scat_matrix_TT
        )

    def __call__(self, inc_theta, out_theta, frequency, to_compute=SCAT_KEYS):
        # Compute first the scattering matrices at the desired frequency.
        # Then interpolate the angles.
        # A possible optimisation: perform the interpolation on frequency and angles
        # in one step instead of two. This would required extending interpolate_matrix.

        # perform frequency interpolation:
        matrices = self.freq_interp_matrices(
            self.frequencies, frequency, self.orig_matrices, **self.interp_freq_kwargs
        )

        # create angle interpolators:
        interpolators = interpolate_matrices(matrices)

        # perform angle interpolation:
        return {
            scat_key: interpolator(inc_theta, out_theta)
            for scat_key, interpolator in interpolators.items()
        }

    @staticmethod
    def freq_interp_matrices(
        frequencies, new_freq, multi_freq_matrices, **interp_freq_kwargs
    ):
        """
        Return the single-frequency scattering matrices from multi-frequency scattering
        matrices by interpolating at the desired frequency.

        Parameters
        ----------
        frequencies : ndarray
            1d
        new_freq : float
        multi_freq_scat_matrices : dict[str]
            Keys: frequencies (1d array), LL, LT, TL, TT
        interp1d_kwargs : kwargs
            Arguments for ``scipy.interpolate.interp1d``

        Returns
        -------
        single_freq_scat_matrices

        """
        out = {}

        interpolation_is_needed = len(frequencies) > 1
        if not interpolation_is_needed:
            # only one frequency, return the only scattering matrices
            if new_freq != frequencies[0]:
                warnings.warn(
                    "No available scattering data at f={}, use f={} instead".format(
                        new_freq, frequencies[0]
                    ),
                    exceptions.ArimWarning,
                )

        for key in SCAT_KEYS:
            try:
                matrix = multi_freq_matrices[key]
            except KeyError:
                continue
            else:
                if interpolation_is_needed:
                    out[key] = interpolate.interp1d(
                        frequencies, matrix, axis=0, **interp_freq_kwargs
                    )(new_freq)
                else:
                    out[key] = matrix[0]
        return out
