"""Nuts and bolts of arim.scat.crack_2d_scat"""

import numpy as np
import numba
import scipy.integrate as si
from numpy.core.umath import sin, cos, pi, exp, sqrt
import ctypes
import scipy


@numba.njit(cache=True)
def basis_function(k):
    k_00 = 1e-1
    if abs(k) <= k_00:
        return 1 - 1 / 18 * k**2 + 1 / 792 * k**4
    else:
        return 105.0 / k**7 * (k * (k * k - 15) * cos(k) - (6 * k * k - 15) * sin(k))


@numba.njit(cache=True)
def sigma(k, k0):
    return sqrt(np.complex_(k * k - k0 * k0)).conjugate()


@numba.njit(cache=True)
def F(xi, xi2, h, beta):
    # input: float, returns a complex
    k = xi2 * h * beta
    F = basis_function(k)
    sigma_1 = sigma(beta, xi)
    sigma_2 = sigma(beta, 1)

    L2 = -((beta**2 - 0.5) ** 2) + beta**2 * sigma_1 * sigma_2
    return F**2 * L2


@numba.njit(cache=True)
def P(k):
    # input: float, returns a complex
    k_00 = 1e-1
    if abs(k) <= k_00:
        F = 1 + 1j * k - 5 / 9 * k**2 - 2j / 9 * k**3
    else:
        sk = (exp(2j * k) - 1) / 2j
        ck = (exp(2j * k) + 1) / 2
        F = 105 / k**7 * (k * (k**2 - 15) * ck - (6 * k**2 - 15) * sk)
    return F**2


@numba.njit(cache=True)
def A_x_F1(x, xi, xi2, h_nodes, z):
    return F(xi, xi2, h_nodes, 1 - x**2) / sqrt(2 - x**2) * cos((1 - x**2) * z)


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_x_F1_real(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_x_F1(x, xi, xi2, h_nodes, z).real


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_x_F1_imag(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_x_F1(x, xi, xi2, h_nodes, z).imag


@numba.njit(cache=True)
def A_x_F2(x, xi, xi2, h_nodes, z):
    return F(xi, xi2, h_nodes, 1 + x**2) / sqrt(2 + x**2) * cos((1 + x**2) * z)


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_x_F2_real(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_x_F2(x, xi, xi2, h_nodes, z).real


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_x_F2_imag(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_x_F2(x, xi, xi2, h_nodes, z).imag


@numba.njit(cache=True)
def A_x_F(x, xi, xi2, h_nodes, z):
    return (
        -(((1 + 1j * x**2) ** 2 - 0.5) ** 2)
        * P(xi2 * h_nodes * (1 + 1j * x**2))
        / sqrt(2 + 1j * x**2)
        * exp(-1j * pi / 4)
        * exp(1j * (z - 2 * xi2 * h_nodes))
        + (xi + 1j * x**2) ** 2
        * P(xi2 * h_nodes * (xi + 1j * x**2))
        * sqrt(2 * xi + 1j * x**2)
        * x**2
        * exp(1j * pi / 4)
        * exp(1j * xi * (z - 2 * xi2 * h_nodes))
    ) * exp(-(z - 2 * xi2 * h_nodes) * x**2)


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_x_F_real(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_x_F(x, xi, xi2, h_nodes, z).real


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_x_F_imag(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_x_F(x, xi, xi2, h_nodes, z).imag


def A_x(xi, xi2, h_nodes, num_nodes):
    # From fn_Qx_matrix
    # Notes: the longest in this function is the numerical integration with scipy.integrate.quad.
    # To speed it up, the integrand is compiled with numba. The quad function requires
    # a function f8(f8, voidptr) so the arguments xi, xi2, h_nodes and z, needed for the
    # calculation of the integrand, are packed.

    # integral around branch point
    z_1 = xi2 * h_nodes * np.arange(num_nodes)

    # Pack arguments for LowLevelCallable.
    # data is updated at every loop. The main loop is NOT thread-safe. If the main loop
    # becomes parallel some day, make "data" local.
    data = np.array([xi, xi2, h_nodes, 0.0])
    data_ptr = ctypes.cast(data.ctypes, ctypes.c_void_p)

    quad_args = dict(limit=200)

    # For num_nodes = 4, I_12 looks like [a3, a2, a1, a0, a1, a2, a3] (size: 2*num_nodes-1)
    # Build the second half first, then copy it to the first half
    I_12 = np.zeros(2 * num_nodes - 1, complex)
    for i, z in enumerate(z_1):
        data[3] = z

        if i < 2:  # two first iterations, coefficients a0 and a1
            int_F1_real = scipy.LowLevelCallable(A_x_F1_real.ctypes, data_ptr)
            int_F1_imag = scipy.LowLevelCallable(A_x_F1_imag.ctypes, data_ptr)
            int_F2_real = scipy.LowLevelCallable(A_x_F2_real.ctypes, data_ptr)
            int_F2_imag = scipy.LowLevelCallable(A_x_F2_imag.ctypes, data_ptr)
            I_12[i + num_nodes - 1] = 4j * (
                si.quad(int_F1_real, 0, 1, **quad_args)[0]
                + 1j * si.quad(int_F1_imag, 0, 1, **quad_args)[0]
            ) + 4 * (
                si.quad(int_F2_real, 0, 50, **quad_args)[0]
                + 1j * si.quad(int_F2_imag, 0, 50, **quad_args)[0]
            )
        else:
            int_F_real = scipy.LowLevelCallable(A_x_F_real.ctypes, data_ptr)
            int_F_imag = scipy.LowLevelCallable(A_x_F_imag.ctypes, data_ptr)

            I_12[i + num_nodes - 1] = 4j * (
                si.quad(int_F_real, 0, 70, **quad_args)[0]
                + 1j * si.quad(int_F_imag, 0, 70, **quad_args)[0]
            )
    I_12[: num_nodes - 1] = I_12[: num_nodes - 1 : -1]

    v_ind = np.arange(num_nodes)
    m_ind = (
        np.full((num_nodes, num_nodes), num_nodes - 1) + v_ind[:, np.newaxis] - v_ind
    )
    return I_12[m_ind]


@numba.njit(cache=True)
def A_z_F1(x, xi, xi2, h_nodes, z):
    return (
        F(xi, xi2, h_nodes, xi - x**2)
        / sqrt(2 * xi - x**2)
        * cos((xi - x**2) * z)
    )


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_z_F1_real(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_z_F1(x, xi, xi2, h_nodes, z).real


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_z_F1_imag(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_z_F1(x, xi, xi2, h_nodes, z).imag


@numba.njit(cache=True)
def A_z_F2(x, xi, xi2, h_nodes, z):
    return (
        F(xi, xi2, h_nodes, xi + x**2)
        / sqrt(2 * xi + x**2)
        * cos((xi + x**2) * z)
    )


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_z_F2_real(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_z_F2(x, xi, xi2, h_nodes, z).real


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_z_F2_imag(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_z_F2(x, xi, xi2, h_nodes, z).imag


@numba.njit(cache=True)
def A_z_F(x, xi, xi2, h_nodes, z):
    return (
        -(((xi + 1j * x**2) ** 2 - 0.5) ** 2)
        * P(xi2 * h_nodes * (xi + 1j * x**2))
        / sqrt(2 * xi + 1j * x**2)
        * exp(-1j * pi / 4)
        * exp(xi * 1j * (z - 2 * xi2 * h_nodes))
        + (1 + 1j * x**2) ** 2
        * P(xi2 * h_nodes * (1 + 1j * x**2))
        * sqrt(2 + 1j * x**2)
        * x**2
        * exp(1j * pi / 4)
        * exp(1j * (z - 2 * xi2 * h_nodes))
    ) * exp(-(z - 2 * xi2 * h_nodes) * x**2)


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_z_F_real(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_z_F(x, xi, xi2, h_nodes, z).real


@numba.cfunc("f8(f8, voidptr)", cache=True)
def A_z_F_imag(x, data):
    xi, xi2, h_nodes, z = numba.carray(data, 4, dtype=numba.float64)
    return A_z_F(x, xi, xi2, h_nodes, z).imag


def A_z(xi, xi2, h_nodes, num_nodes):
    # from fn_Qz_matrix

    # integral around branch point
    z_1 = xi2 * h_nodes * np.arange(num_nodes)

    # Pack arguments for LowLevelCallable.
    # data is updated at every loop. The main loop is NOT thread-safe. If the main loop
    # becomes parallel some day, make "data" local.
    data = np.array([xi, xi2, h_nodes, 0.0])
    data_ptr = ctypes.cast(data.ctypes, ctypes.c_void_p)

    quad_args = dict(limit=200)

    # For num_nodes = 4, I_12 looks like [a3, a2, a1, a0, a1, a2, a3] (size: 2*num_nodes-1)
    # Build the second half first, then copy it to the first half
    I_12 = np.zeros(2 * num_nodes - 1, complex)
    for i, z in enumerate(z_1):
        data[3] = z
        if i < 2:  # two first iterations, coefficients a0 and a1
            int_F1_real = scipy.LowLevelCallable(A_z_F1_real.ctypes, data_ptr)
            int_F1_imag = scipy.LowLevelCallable(A_z_F1_imag.ctypes, data_ptr)
            int_F2_real = scipy.LowLevelCallable(A_z_F2_real.ctypes, data_ptr)
            int_F2_imag = scipy.LowLevelCallable(A_z_F2_imag.ctypes, data_ptr)

            I_12[i + num_nodes - 1] = 4j * (
                si.quad(int_F1_real, 0, sqrt(xi), **quad_args)[0]
                + 1j * si.quad(int_F1_imag, 0, sqrt(xi), **quad_args)[0]
            ) + 4 * (
                si.quad(int_F2_real, 0, 50, **quad_args)[0]
                + 1j * si.quad(int_F2_imag, 0, 50, **quad_args)[0]
            )
        else:
            int_F_real = scipy.LowLevelCallable(A_z_F_real.ctypes, data_ptr)
            int_F_imag = scipy.LowLevelCallable(A_z_F_imag.ctypes, data_ptr)

            I_12[i + num_nodes - 1] = 4j * (
                si.quad(int_F_real, 0, 70, **quad_args)[0]
                + 1j * si.quad(int_F_imag, 0, 70, **quad_args)[0]
            )
    I_12[: num_nodes - 1] = I_12[: num_nodes - 1 : -1]

    v_ind = np.arange(num_nodes)
    m_ind = (
        np.full((num_nodes, num_nodes), num_nodes - 1) + v_ind[:, np.newaxis] - v_ind
    )
    return I_12[m_ind]


@numba.jit(nopython=True, nogil=True, cache=True)
def crack_2d_scat_kernel(
    phi_in,
    phi_out_array,
    vel_L,
    vel_T,
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
):
    """
    work on one incident angle in order to cache the results of two to four
    linear solve
    """

    # Lamé coefficients, see http://subsurfwiki.org/wiki/Template:Elastic_modulus
    lame_lambda = density * (vel_L**2 - 2 * vel_T**2)
    lame_mu = density * vel_T**2

    omega = 2 * pi * frequency
    xi1 = 2 * pi * frequency / vel_L
    xi2 = 2 * pi * frequency / vel_T
    lambda_L = vel_L / frequency
    lambda_T = vel_T / frequency
    xi = vel_T / vel_L
    k_L = xi1  # alias
    k_T = xi2  # alias
    a_L = -1j * k_L * pi / xi2**2  # incident L wave
    a_T = -1j * k_T * pi / xi2**2  # incident S wave

    # normal vector to the crack
    nv = np.array([0.0, 1.0], np.complex128)  # force to complex to please numba

    sv = np.array(
        [-sin(phi_in), -cos(phi_in)], np.complex128
    )  # force to complex to please numba
    tv = np.array([sv[1], -sv[0]], np.complex128)
    if use_incident_L:
        b_L = exp(1j * k_L * x_nodes * sv[0]) * basis_function(-k_L * h_nodes * sv[0])
        b_x = -2 * sv[0] * sv[1] * b_L
        b_z = -(1 / xi**2 - 2 * sv[0] ** 2) * b_L
        vxL = np.linalg.solve(A_x, b_x)
        vzL = np.linalg.solve(A_z, b_z)
    if use_incident_T:
        b_T = exp(1j * k_T * x_nodes * sv[0]) * basis_function(-k_T * h_nodes * sv[0])
        b_x = -(tv[0] * sv[1] + tv[1] * sv[0]) * b_T
        b_z = -2 * tv[1] * sv[1] * b_T
        vxT = np.linalg.solve(A_x, b_x)
        vzT = np.linalg.solve(A_z, b_z)

    for j, phi_out in enumerate(phi_out_array):
        ev = np.array([sin(phi_out), cos(phi_out)], np.complex128)
        tv = np.array([ev[1], -ev[0]], np.complex128)
        c_L = basis_function(xi1 * h_nodes * ev[0]) * exp(-1j * xi1 * ev[0] * x_nodes)
        c_T = basis_function(xi2 * h_nodes * ev[0]) * exp(-1j * xi2 * ev[0] * x_nodes)

        if use_incident_L:
            v_L = np.array([a_L * np.dot(vxL, c_L), a_L * np.dot(vzL, c_L)])
            v_T = np.array([a_L * np.dot(vxL, c_T), a_L * np.dot(vzL, c_T)])

            S_LL[j] = (
                1
                / 4
                * sqrt(2 / pi)
                * exp(-1j * pi / 4)
                * xi1 ** (5 / 2)
                * (
                    lame_lambda / (density * omega**2) * (np.dot(v_L, nv))
                    + 2
                    * lame_mu
                    / (density * omega**2)
                    * np.dot(v_L, ev)
                    * np.dot(ev, nv)
                )
                / sqrt(lambda_L)
            )
            S_LT[j] = (
                1
                / 4
                * sqrt(2 / pi)
                * exp(-1j * pi / 4)
                * xi2 ** (5 / 2)
                * lame_mu
                / (density * omega**2)
                * (np.dot(v_T, tv) * np.dot(ev, nv) + np.dot(v_T, ev) * np.dot(tv, nv))
                / sqrt(lambda_T)
            )
        if use_incident_T:
            v_L = np.array([a_T * np.dot(vxT, c_L), a_T * np.dot(vzT, c_L)])
            v_T = np.array([a_T * np.dot(vxT, c_T), a_T * np.dot(vzT, c_T)])

            # This is the same expression as for LL and LT but v_L and v_T are
            # different.
            # Add a minus sign compared to the original code because change of
            # polarisation.
            S_TL[j] = -(
                1
                / 4
                * sqrt(2 / pi)
                * exp(-1j * pi / 4)
                * xi1 ** (5 / 2)
                * (
                    lame_lambda / (density * omega**2) * (np.dot(v_L, nv))
                    + 2
                    * lame_mu
                    / (density * omega**2)
                    * np.dot(v_L, ev)
                    * np.dot(ev, nv)
                )
                / sqrt(lambda_L)
            )
            S_TT[j] = -(
                1
                / 4
                * sqrt(2 / pi)
                * exp(-1j * pi / 4)
                * xi2 ** (5 / 2)
                * lame_mu
                / (density * omega**2)
                * (np.dot(v_T, tv) * np.dot(ev, nv) + np.dot(v_T, ev) * np.dot(tv, nv))
                / sqrt(lambda_T)
            )
    return S_LL, S_LT, S_TL, S_TT


@numba.jit(nopython=True, nogil=True, cache=True, parallel=False)
def crack_2d_scat_matrix(
    phi_in_vect,
    phi_out_array,
    vel_L,
    vel_T,
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
):
    """
    call the kernel in the case where there is one phi_in for many phi_out
    (use optimised kernel)
    """
    # todo: set parallel=True if one day numba supports cache=True with this flag.
    assert phi_in_vect.ndim == 1
    assert phi_out_array.ndim == 2
    for i in range(phi_in_vect.shape[0]):
        crack_2d_scat_kernel(
            phi_in_vect[i],
            phi_out_array[:, i],
            vel_L,
            vel_T,
            density,
            frequency,
            use_incident_L,
            use_incident_T,
            x_nodes,
            h_nodes,
            A_x,
            A_z,
            S_LL[:, i],
            S_LT[:, i],
            S_TL[:, i],
            S_TT[:, i],
        )
    return S_LL, S_LT, S_TL, S_TT


@numba.jit(nopython=True, nogil=True, cache=True, parallel=False)
def crack_2d_scat_general(
    phi_in_array,
    phi_out_array,
    vel_L,
    vel_T,
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
):
    """
    call the kernel in the case where there is one phi_in for one phi_out
    (no optimisation available)
    """
    # todo: set parallel=True if one day numba supports cache=True with this flag.
    for i in range(phi_in_array.shape[0]):
        for j in range(phi_out_array.shape[1]):
            # pass a slice which is writeable
            crack_2d_scat_kernel(
                phi_in_array[i, j],
                phi_out_array[i, j : j + 1],
                vel_L,
                vel_T,
                density,
                frequency,
                use_incident_L,
                use_incident_T,
                x_nodes,
                h_nodes,
                A_x,
                A_z,
                S_LL[i, j : j + 1],
                S_LT[i, j : j + 1],
                S_TL[i, j : j + 1],
                S_TT[i, j : j + 1],
            )
    return S_LL, S_LT, S_TL, S_TT
