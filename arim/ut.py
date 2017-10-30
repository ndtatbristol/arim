"""
Toolbox of functions for ultrasonic testing/acoustics.
"""
# Only function that does not require any arim-specific logic should be put here.
# This module must be kept free of any arim dependencies because so that it could be used
# without arim.

import numpy as np
import warnings
import numba
import math
from scipy.special import hankel1, hankel2
from functools import partial

from numpy.core.umath import sin, cos, pi, exp


class UtWarning(UserWarning):
    pass


def fmc(numelements):
    """
    Return all pairs of elements for a FMC.
    HMC as performed by Brain.

    Returns
    -------
    tx : ndarray [numelements^2]
        Transmitter for each scanline: 0, 0, ..., 1, 1, ...
    rx : ndarray
        Receiver for each scanline: 1, 2, ..., 1, 2, ...
    """
    numelements = int(numelements)
    elements = np.arange(numelements)

    # 0 0 0    1 1 1    2 2 2
    tx = np.repeat(elements, numelements)

    # 0 1 2    0 1 2    0 1 2
    rx = np.tile(elements, numelements)
    return tx, rx


def hmc(numelements):
    """
    Return all pairs of elements for a HMC.
    HMC as performed by Brain (rx >= tx)

    Returns
    -------
    tx : ndarray [numelements^2]
        Transmitter for each scanline: 0, 0, 0, ..., 1, 1, 1, ...
    rx : ndarray
        Receiver for each scanline: 0, 1, 2, ..., 1, 2, ...
    """
    numelements = int(numelements)
    elements = np.arange(numelements)

    # 0 0 0    1 1    2
    tx = np.repeat(elements, range(numelements, 0, -1))

    # 0 1 2    0 1    2
    rx = np.zeros_like(tx)
    take_n_last = np.arange(numelements, 0, -1)
    start = 0
    for n in take_n_last:
        stop = start + n
        rx[start:stop] = elements[-n:]
        start = stop
    return tx, rx


def infer_capture_method(tx, rx):
    """
    Infers the capture method from the indices of transmitters and receivers.

    Returns: 'hmc', 'fmc', 'unsupported'

    Parameters
    ----------
    tx : list
        One per scanline
    rx : list
        One per scanline

    Returns
    -------
    capture_method : string
    """
    numelements = max(np.max(tx), np.max(rx)) + 1
    assert len(tx) == len(rx)

    # Get the unique combinations tx/rx of the input.
    # By using set, we ignore the order of the combinations tx/rx.
    combinations = set(zip(tx, rx))

    # Could it be a HMC? Most frequent case, go first.
    # Remark: HMC can be made with tx >= rx or tx <= rx. Check both.
    tx_hmc, rx_hmc = hmc(numelements)
    combinations_hmc1 = set(zip(tx_hmc, rx_hmc))
    combinations_hmc2 = set(zip(rx_hmc, tx_hmc))

    if (len(tx_hmc) == len(tx)) and ((combinations == combinations_hmc1) or
                                         (combinations == combinations_hmc2)):
        return 'hmc'

    # Could it be a FMC?
    tx_fmc, rx_fmc = fmc(numelements)
    combinations_fmc = set(zip(tx_fmc, rx_fmc))
    if (len(tx_fmc) == len(tx)) and (combinations == combinations_fmc):
        return 'fmc'

    # At this point we are hopeless
    return 'unsupported'


def default_scanline_weights(tx, rx):
    """
    Scanline weights for TFM.

    Consider a scanline obtained by the transmitter i and the receiver j; this
    scanline is denoted (i,j). If the response matrix contains both (i, j) and (j, i),
    the corresponding scanline weight is 1. Otherwise, the scanline weight is 2.

    Example: for a FMC, all scanline weights are 1.
    Example: for a HMC, scanline weights for the pulse-echo scanlines are 1,
    scanline weights for the non-pulse-echo scanlines are 2.

    Remark: the function does not check if there are duplicated signals.

    Parameters
    ----------
    tx : list[int] or ndarray
        tx[i] is the index of the transmitter (between 0 and numelements-1) for
        the i-th scanline.
    rx : list[int] or ndarray
        rx[i] is the index of the receiver (between 0 and numelements-1) for
        the i-th scanline.

    Returns
    -------
    scanline_weights : ndarray

    """
    if len(tx) != len(rx):
        raise ValueError('tx and rx must have the same lengths (numscanlines)')
    numscanlines = len(tx)

    # elements_pairs contains (tx[0], rx[0]), (tx[1], rx[1]), etc.
    elements_pairs = {*zip(tx, rx)}
    scanline_weights = np.ones(numscanlines)
    for this_tx, this_rx, scanline_weight in \
            zip(tx, rx, np.nditer(scanline_weights, op_flags=['readwrite'])):
        if (this_rx, this_tx) not in elements_pairs:
            scanline_weight[...] = 2.
    return scanline_weights


def decibel(arr, reference=None, neginf_value=-1000., return_reference=False):
    """
    Return 20*log10(abs(arr) / reference)

    If reference is None, use:

        reference := max(abs(arr))

    Parameters
    ----------
    arr : ndarray
        Values to convert in dB.
    reference : float or None
        Reference value for 0 dB. Default: None
    neginf_value : float or None
        If not None, convert -inf dB values to this parameter. If None, -inf
        dB values are not changed.
    return_max : bool
        Default: False.

    Returns
    -------
    arr_db
        Array in decibel.
    arr_max: float
        Return ``max(abs(arr))``. This value is returned only if return_max is true.

    """
    # Disable warnings messages for log10(0.0)
    arr_abs = np.abs(arr)

    if arr_abs.shape == ():
        orig_shape = ()
        arr_abs = arr_abs.reshape((1,))
    else:
        orig_shape = None

    if reference is None:
        reference = np.nanmax(arr_abs)
    else:
        assert reference > 0.

    with np.errstate(divide='ignore'):
        arr_db = 20 * np.log10(arr_abs / reference)

    if neginf_value is not None:
        arr_db[np.isneginf(arr_db)] = neginf_value

    if orig_shape is not None:
        arr_db = arr_db.reshape(orig_shape)

    if return_reference:
        return arr_db, reference
    else:
        return arr_db


def wrap_phase(phases):
    """Return a phase in [-pi, pi[

    http://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    """
    phases = np.asarray(phases)
    return (phases + np.pi) % (2 * np.pi) - np.pi


def instantaneous_phase_shift(analytic_sig, time_vect, carrier_frequency):
    """
    For a signal $x(ray) = A * exp(i (2 pi f_0 ray + phi(ray)))$, returns phi(ray) in [-pi, pi[.

    Parameters
    ----------
    analytic_sig: ndarray
    time_vect: ndarray
    carrier_frequency: float

    Returns
    -------
    phase_shift

    """
    analytic_sig = np.asarray(analytic_sig)
    dtype = analytic_sig.dtype
    if dtype.kind != 'c':
        warnings.warn('Expected an analytic (complex) signal, got {}. Use a Hilbert '
                      'transform to get the analytic signal.'.format(dtype), UtWarning,
                      stacklevel=2)
    phase_correction = 2 * np.pi * carrier_frequency * time_vect
    phase = wrap_phase(np.angle(analytic_sig) - phase_correction)
    return phase


def directivity_2d_rectangular_in_fluid(theta, element_width, wavelength):
    """
    Returns the directivity of an element based on the integration of uniformally radiating sources
    along a straight line in 2D.

    A element is modelled as 'rectangle' of finite width and infinite length out-of-plane.

    This directivity is based only on the element width: each source is assumed to radiate
    uniformally.

    Considering a points1 in the axis Ox in the cartesian basis (O, x, y, z),
    ``theta`` is the inclination angle, ie. the angle in the plane Oxz. Cf. Wooh's paper.

    The directivity is normalised by the its maximum value, obtained for
    theta=0°.

    Returns:

        sinc(pi*a*sin(theta)/lambda)

    where: sinc(x) = sin(x)/x


    Parameters
    ----------
    theta : ndarray
        Angles in radians.
    element_width : float
        In meter.
    wavelength : float
        In meter.

    Returns
    -------
    directivity : ndarray
        Signed directivity for each angle.

    References
    ----------

    Wooh, Shi-Chang, and Yijun Shi. 1999. ‘Three-Dimensional Beam Directivity of Phase-Steered Ultrasound’.
        The Journal of the Acoustical Society of America 105 (6): 3275–82. doi:10.1121/1.424655.

    See Also
    --------
    :func:`transmission_2d_rectangular_in_fluid`

    """
    if element_width < 0:
        raise ValueError('Negative width')
    if wavelength < 0:
        raise ValueError('Negative wavelength')

    # /!\ numpy.sinc defines sinc(x) := sin(pi * x)/(pi * x)
    x = (element_width / wavelength) * np.sin(theta)
    return np.sinc(x)


# backward compatibility
directivity_finite_width_2d = directivity_2d_rectangular_in_fluid


def radiation_2d_rectangular_in_fluid(theta, element_width, wavelength):
    """
    Piston model.

    Field is::

        p(r, theta, ray) = V0(omega) * Z * R(theta) * exp(i omega ray - k r) * sqrt(wavelength / r)

    where:
        - V0 is the (uniform) velocity on the piston
        - P(theta) is the output of the current function,
        - Z = rho c is the acoustic impedance

    Reference: Schmerr, Fundamentals of ultrasonic phased array, eq (2.38)

    R is dimensionless.

    Parameters
    ----------
    theta : ndarray
    element_width
    wavelength

    Returns
    -------
    radiation : ndarray


    """
    directivity = directivity_2d_rectangular_in_fluid(theta, element_width, wavelength)

    r = np.sqrt(1j) * (element_width / wavelength) * directivity
    return r


def radiation_2d_cylinder_in_fluid(source_radius, wavelength):
    """
    Radiation term in far field for a cylinder radiating uniform in a fluid.

    The field generated is::

        p(r, ray) = V0(omega) * Z * R * exp(i omega ray - k r) * sqrt(wavelength / r)

    where:
        - V0 is the (uniform) velocity on the piston
        - R is the output of the current function
        - Z = rho c is the acoustic impedance

    Reference: Theoretical Acoustics, Philip M. Morse & K. Uno Ingard, equation 7.3.3

    R is dimensionless.


    Parameters
    ----------
    source_radius : float
    wavelength : float

    Returns
    -------
    transmission_term : complex

    """
    rad = 2. * np.sqrt(1j) * (source_radius / wavelength)
    return rad


def snell_angles(incidents_angles, c_incident, c_refracted):
    """
    Returns the angles of the refracted rays according to Snell–Descartes law:

        c1/c2 = sin(alpha1)/sin(alpha2)

    In case of total internal reflection (incident angles above the critical angles), the output depends
    on the datatype of the incident angle.
    If the incident angle is real, the refracted angle is "not a number".
    If the incident angle is complex, the refracted angle is complex (imagery part not null).
    The reason is that either the real or the complex arcsine function is used.
    """
    return np.arcsin(c_refracted / c_incident * sin(incidents_angles))


def _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l,
                   c_t):
    """
    Coefficient N defined by Krautkrämer in equation (A8).
    """
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)
    N = (ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
         + cos_2_alpha_t * cos_2_alpha_t
         + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid))
    return N


def fluid_solid(alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_l=None,
                alpha_t=None):
    """
    Returns the transmission and reflection coefficients for an incident wave at a fluid-to-solid interface.

    The coefficients are expressed as pressure/stress ratio. All angles are in radians.
    The angles are relative to the normal of the interface.

    Cf. equations (A7), (A8), (A9) from Krautkrämer.

    Caveat: in case of total reflection, using complex angles should be considered.

    Parameters
    ----------
    alpha_fluid : ndarray
        Angles of the incident wave in the fluid.
    rho_fluid : float
        Density of the fluid.
    rho_solid : float
        Density of the solid.
    c_fluid : float
        Speed of sound in the fluid.
    c_l : float
        Speed of the longitudinal wave in the solid.
    c_t : float
        Speed of the transverse wave in the solid.
    alpha_l : ndarray or None
        Angles of the transmitted longitudinal wave in the solid. If None: compute it on the fly with Snell-Descartes laws.
    alpha_t : ndarray or None
        Angles of the transmitted transverse wave in the solid. If None: compute it on the fly with Snell-Descartes laws.

    Returns
    -------
    reflection : ndarray
        Reflection coefficient
    transmission_l : ndarray
        Transmission coefficient of the longitudinal wave
    transmission_l : ndarray
        Reflection coefficient of the longitudinal wave

    References
    ----------
    [KK]_


    """
    alpha_fluid = np.asarray(alpha_fluid)

    if alpha_l is None:
        alpha_l = snell_angles(alpha_fluid, c_fluid, c_l)
    if alpha_t is None:
        alpha_t = snell_angles(alpha_fluid, c_fluid, c_t)

    N = _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l,
                       c_t)

    # Eq A.7
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)

    reflection = (
                     ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
                     + cos_2_alpha_t * cos_2_alpha_t
                     - (rho_fluid * c_fluid * cos(alpha_l)) /
                     (rho_solid * c_l * cos(alpha_fluid))
                 ) / N

    # Eq A.8
    transmission_l = 2. * cos_2_alpha_t / N

    # Eq A.9
    transmission_t = -2. * ct_cl2 * sin(2 * alpha_l) / N

    return reflection, transmission_l, transmission_t


def solid_l_fluid(alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid=None,
                  alpha_t=None):
    """
    Returns the transmission and reflection coefficients for an incident longitudinal wave at a solid-to-fluid interface.

    The coefficients are expressed as pressure/stress ratio. All angles are in radians.
    The angles are relative to the normal of the interface.

    Cf. equations (A7), (A8), (A9) from Krautkrämer.

    Caveat: in case of total reflection, using complex angles should be considered.

    Parameters
    ----------
    alpha_l : ndarray
        Angles of the incident longitudinal wave in the solid.
    rho_fluid : float
        Density of the fluid.
    rho_solid : float
        Density of the solid.
    c_fluid : float
        Speed of sound in the fluid.
    c_l : float
        Speed of the longitudinal wave in the solid.
    c_t : float
        Speed of the transverse wave in the solid.
    alpha_fluid : ndarray or None
        Angles of the transmitted wave in the fluid. If None: compute it on the fly with Snell-Descartes laws.
    alpha_t : ndarray or None
        Angles of the incident transverse wave in the solid. If None: compute it on the fly with Snell-Descartes laws.

    Returns
    -------
    reflection_l : ndarray
        Reflection coefficient of the longitudinal wave
    reflection_t : ndarray
        Reflection coefficient of the transverse wave
    transmission : ndarray
        Transmission coefficient

    References
    ----------
    [KK]_


    """
    if alpha_fluid is None:
        alpha_fluid = snell_angles(alpha_l, c_l, c_fluid)
    if alpha_t is None:
        alpha_t = snell_angles(alpha_l, c_l, c_t)

    N = _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l,
                       c_t)

    # Eq A.10
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)
    reflection_l = (ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
                    - cos_2_alpha_t * cos_2_alpha_t
                    + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(
        alpha_fluid)) / N

    # Eq A.11
    reflection_t = (2 * ct_cl2 * sin(2 * alpha_l) * cos(2 * alpha_t)) / N

    # Eq A.12
    transmission = 2 * rho_fluid * c_fluid * cos(alpha_l) * cos(2 * alpha_t) / (
        N * rho_solid * c_l * cos(alpha_fluid))

    return reflection_l, reflection_t, transmission


def solid_t_fluid(alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid=None,
                  alpha_l=None):
    """
    Returns the transmission and reflection coefficients for an incident transverse wave at a solid-to-fluid interface.

    The coefficients are expressed as pressure/stress ratio. All angles are in radians.
    The angles are relative to the normal of the interface.

    Cf. equations (A7), (A8), (A9) from Krautkrämer.

    Caveat: in case of total reflection, using complex angles should be considered.

    Parameters
    ----------
    alpha_t : ndarray
        Angles of the incident transverse wave in the solid.
    rho_fluid : float
        Density of the fluid.
    rho_solid : float
        Density of the solid.
    c_fluid : float
        Speed of sound in the fluid.
    c_l : float
        Speed of the longitudinal wave in the solid.
    c_t : float
        Speed of the transverse wave in the solid.
    alpha_fluid : ndarray or None
        Angles of the transmitted wave in the fluid. If None: compute it on the fly with Snell-Descartes laws.
    alpha_l : ndarray or None
        Angles of the incident longitudinal wave in the solid. If None: compute it on the fly with Snell-Descartes laws.

    Returns
    -------
    reflection_l : ndarray
        Reflection coefficient of the longitudinal wave
    reflection_t : ndarray
        Reflection coefficient of the transverse wave
    transmission : ndarray
        Transmission coefficient

    References
    ----------
    [KK]_

    """
    if alpha_fluid is None:
        alpha_fluid = snell_angles(alpha_t, c_t, c_fluid)
    if alpha_l is None:
        alpha_l = snell_angles(alpha_t, c_t, c_l)

    N = _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l,
                       c_t)

    # Eq A.14
    reflection_l = -sin(4 * alpha_t) / N

    # Eq A.13
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)
    reflection_t = (ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
                    - cos_2_alpha_t * cos_2_alpha_t
                    - rho_fluid * c_fluid / (rho_solid * c_l)
                    * cos(alpha_l) / cos(alpha_fluid)) / N

    # Eq A.15
    transmission = 2 * rho_fluid * c_fluid * cos(alpha_l) * sin(2 * alpha_t) / (
        N * rho_solid * c_l * cos(alpha_fluid))

    # TODO: Rose in "Ultrasonic guided waves in solid media" gives the oppositeof these
    # coefficients. Fix this?
    return reflection_l, reflection_t, transmission


def scattering_angles(numpoints):
    """Return angles for scattering matrices. Linearly spaced vector in [-pi, pi[."""
    return np.linspace(-np.pi, np.pi, numpoints, endpoint=False)


def scattering_angles_grid(numpoints):
    """Return angles for scattering matrices as a grid of incident and outgoing angles.
    """
    theta = scattering_angles(numpoints)
    inc_theta, out_theta = np.meshgrid(theta, theta, indexing='xy')
    return inc_theta, out_theta


def make_scattering_matrix(scattering_func, numpoints):
    """
    Call the function 'scattering_func' for all incident and scattered angles.
    Returns the result as a matrix.

    Parameters
    ----------
    scattering_func :
        Function that returns the amplitudes of the scattered wave.
         Its first argument must be the angle of the incident wave, its second
         argument must be the angle of the scattered wave.
    numpoints : int
        Number of points to use.

    Returns
    -------
    scattering_matrix : ndarray
        Shape (numpoints, numpoints)
        scattering_matrix[i, j] is the coefficient for the incident angle theta[j] and
        the scattering angle theta[i].
    """
    inc_theta, out_theta = scattering_angles_grid(numpoints)
    scattering_matrix = scattering_func(inc_theta, out_theta)
    return scattering_matrix


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


@numba.guvectorize(['void(f8[:,:], f8[:], f8[:], f8[:])',
                    'void(c16[:,:], f8[:], f8[:], c16[:])'],
                   '(s,s),(),()->()',
                   nopython=True, target='parallel')
def _interpolate_scattering_matrix_ufunc(scattering_matrix, inc_theta, out_theta, res):
    res[0] = _interpolate_scattering_matrix_kernel(scattering_matrix, inc_theta[0],
                                                   out_theta[0])


def interpolate_scattering_matrix(scattering_matrix):
    """
    Returns a function that takes as input the incident angles and the scattering angles.
    This returned function returns the scattering amplitudes, obtained by bilinear
    interpolation of the scattering matrix.

    Parameters
    ----------
    scattering_matrix

    Returns
    -------
    func

    """
    scattering_matrix = np.asarray(scattering_matrix)
    if scattering_matrix.ndim != 2:
        raise ValueError('scattering matrix must have a shape (n, n)')
    if scattering_matrix.shape[0] != scattering_matrix.shape[1]:
        raise ValueError('scattering matrix must have a shape (n, n)')

    return partial(_interpolate_scattering_matrix_ufunc, scattering_matrix)


def scattering_matrices_to_interp_funcs(scattering_matrices):
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
    return {key: interpolate_scattering_matrix(mat) for key, mat
            in scattering_matrices.items()}


def scattering_2d_cylinder(inc_theta, out_theta, radius, longitudinal_wavelength,
                           transverse_wavelength,
                           min_terms=10, term_factor=4,
                           to_compute={'LL', 'LT', 'TL', 'TT'}):
    """
    Scattering coefficients for a 2D circle (or infinitely-long cylinder in 3D).

    The scattered field is given by::

        u_scat(r, theta) = u0 * sqrt(1 / r) * exp(-i k r + i omega i ray) *
                           (sqrt(lambda_L) A(theta) e_r +
                            sqrt(lambda_T) B(theta) e_theta)

    where A(theta) and B(theta) are the scattering coefficients for respectively L and
    T scattered waves and where e_r and e_theta are the two vectors of the cylindrical
    coordinate system.

    Keys for the coefficients: 'LL', 'LT', 'TL', 'TT'. The first letter refers to the kind
    of the incident wave, the second letter refers to the kind of the scattered wave.

    The coefficient for LL, LT, TL and TT are obtained from Lopez-Sanchez's paper,
    equations 33, 34, 39, 40. See also Brind's paper. Compared to these papers, the
    complex conjugate coefficients are returned because these papers use the
    convention ``u(ray) = u(omega) exp(-i omega ray)`` whereas we use the convention
    ``u(ray) = u(omega) exp(+i omega ray)``.

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
    radius : float
    longitudinal_wavelength : float
    transverse_wavelength : float
    min_terms : int
    term_factor : int
    to_compute : set
        Coefficients to compute. Default: compute all.

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
    valid_keys = {'LL', 'LT', 'TL', 'TT'}

    theta = out_theta - inc_theta

    if not valid_keys.issuperset(to_compute):
        raise ValueError("Valid 'to_compute' arguments are {} (got {})".format(valid_keys,
                                                                               to_compute))

    # wavenumber
    kl = 2 * pi / longitudinal_wavelength
    kt = 2 * pi / transverse_wavelength

    # Brind eq 2.8
    alpha = kl * radius
    beta = kt * radius
    beta2 = beta * beta

    # sum from n=0 to n=maxn (inclusive)
    # The larger maxn, the better the axppromixation
    maxn = max([int(min_terms),
                math.ceil(term_factor * alpha), math.ceil(term_factor * beta)])
    n = np.arange(0, maxn + 1)
    n2 = n * n

    # Brind eq 2.8
    epsilon = np.full(n.shape, 2.)
    epsilon[0] = 1.

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
    n_phi = np.einsum('...,j->...j', phi, n)
    cos_n_phi = cos(n_phi)
    sin_n_phi = sin(n_phi)
    del n_phi

    result = dict()

    # NB: sqrt(2j/(pi * k)) = sqrt(i) / pi

    if 'LL' in to_compute:
        # Lopez-Sanchez eq (29)
        A_n = 1j / (2 * alpha) * (
            1 + (c2_alpha * c1_beta - d2_alpha * d1_beta) /
            (c1_alpha * c1_beta - d1_alpha * d1_beta))

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
        r = (np.sqrt(1j) / pi * alpha) * \
            np.einsum('...j,j->...', cos_n_phi, epsilon * A_n)

        result['LL'] = r.conjugate()

    if 'LT' in to_compute:
        # Lopez-Sanchez eq (30)
        B_n = 2 * n / (pi * alpha) * ((n2 - beta2 / 2 - 1) /
                                      (c1_alpha * c1_beta - d1_alpha * d1_beta))

        # Lopez-Sanchez (34)
        # Warning: there is a minus sign in Brind (2.10). We trust LS here.
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * beta) * \
            np.einsum('...j,j->...', sin_n_phi, epsilon * B_n)
        result['LT'] = r.conjugate()

    if 'TL' in to_compute:
        # Lopez-Sanchez eq (41)
        A_n = 2 * n / (pi * beta) * (n2 - beta2 / 2 - 1) / (
            c1_alpha * c1_beta - d1_alpha * d1_beta)

        # Lopez-Sanchez eq (39)
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * alpha) * \
            np.einsum('...j,j->...', sin_n_phi, epsilon * A_n)
        result['TL'] = r.conjugate()

    if 'TT' in to_compute:
        # Lopez-Sanchez eq (42)
        B_n = 1j / (2 * beta) * (1 + (c2_beta * c1_alpha - d2_beta * d1_alpha) /
                                 (c1_alpha * c1_beta - d1_alpha * d1_beta))

        # Lopez-Sanchez eq (40)
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * beta) * \
            np.einsum('...j,j->...', cos_n_phi, epsilon * B_n)
        result['TT'] = r.conjugate()

    return result


def _scattering_2d_cylinder(inc_theta, out_theta, scat_key, **scat_params):
    return scattering_2d_cylinder(inc_theta, out_theta, to_compute={scat_key},
                                  **scat_params)[scat_key]


def scattering_2d_cylinder_funcs(radius, longitudinal_wavelength,
                                 transverse_wavelength, **kwargs):
    """
    Cf. :func:`scattering_2d_cylinder`

    Parameters
    ----------
    radius
    longitudinal_wavelength
    transverse_wavelength
    kwargs

    Returns
    -------
    scat_funcs: dict of func
        Usage: scat_funcs['LT'](inc_theta, out_theta)

    """
    scat_funcs = {}
    scat_params = dict(
        radius=radius,
        longitudinal_wavelength=longitudinal_wavelength,
        transverse_wavelength=transverse_wavelength,
        **kwargs
    )

    for scat_key in ['LL', 'LT', 'TL', 'TT']:
        scat_funcs[scat_key] = partial(_scattering_2d_cylinder, scat_key=scat_key,
                                       **scat_params)
    return scat_funcs


def scattering_2d_cylinder_matrices(numpoints, radius, longitudinal_wavelength,
                                    transverse_wavelength,
                                    to_compute={'LL', 'LT', 'TL', 'TT'}, **kwargs):
    """
    Cf. :func:`scattering_2d_cylinder`.

    Parameters
    ----------
    numpoints : int
        Number of points for discretising [-pi, pi[.
    radius
    longitudinal_wavelength
    transverse_wavelength
    min_terms
    term_factor
    to_compute

    Returns
    -------
    matrices : dict
    """
    matrices = {}
    scat_funcs = scattering_2d_cylinder_funcs(radius, longitudinal_wavelength,
                                              transverse_wavelength, **kwargs)
    for scat_key in to_compute:
        matrices[scat_key] = make_scattering_matrix(scat_funcs[scat_key], numpoints)
    return matrices


def scattering_point_source_funcs(longitudinal_velocity, transverse_velocity):
    """
    (Unphysical) scattering functions of a point source. For debug only.

    Parameters
    ----------
    longitudinal_velocity : float
    transverse_velocity : float

    Returns
    -------
    dict
    """
    vl = longitudinal_velocity
    vt = transverse_velocity
    return {
        'LL': lambda inc, out: np.full_like(inc, 1.),
        'LT': lambda inc, out: np.full_like(inc, vl / vt),
        'TL': lambda inc, out: -np.full_like(inc, vt / vl),
        'TT': lambda inc, out: np.full_like(inc, 1.),
    }


def make_timevect(num, step, start=0., dtype=None):
    """
    Return a linearly spaced time vector.

    Remark: using this method is preferable to ``numpy.arange(start, start + num * step, step``
    which may yield an incorrect number of samples due to numerical inaccuracy.

    Parameters
    ----------
    num : int
        Number of samples to generate.
    step : float, optional
        Time step (time between consecutive samples).
    start : scalar
        Starting value of the sequence. Default: 0.
    dtype : numpy.dtype
        Optional, the type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    samples : ndarray
        Linearly spaced vector ``[start, stop]`` where ``end = start + (num - 1)*step``

    Examples
    --------
    >>> make_timevect(10, .1)
    array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
    >>> make_timevect(10, .1, start=1.)
    array([ 1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9])

    Notes
    -----

    Adapted from ``numpy.linspace``
    (License: http://www.numpy.org/license.html ; 3 clause BSD)

    """
    if not isinstance(num, int):
        raise TypeError('num must be an integer (got {})'.format(type(num)))
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)

    # Convert float/complex array scalars to float
    start = start * 1.
    step = step * 1.

    dt = np.result_type(start, step)
    if dtype is None:
        dtype = dt

    y = np.arange(0, num, dtype=dt)

    if num > 1:
        y *= step

    y += start

    return y.astype(dtype, copy=False)


def rotate_scattering_matrix(scat_matrix, phi):
    """
    Return the scattering matrix S' of the scatterer rotated by an angle phi,
    knowing the scattering matrix S of the unrotated scatterer.

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

    freq_x, freq_y = np.meshgrid(freq, freq, indexing='ij')

    freqshift = np.exp(-2j * np.pi * (freq_x + freq_y) * phi)
    scat_matrix_f = np.fft.fft2(scat_matrix)
    return np.fft.ifft2(freqshift * scat_matrix_f)


def _rotate_array(arr, n):
    """
        >>> _rotate_array([1, 2, 3, 4, 5, 6, 7], 2)
        array([3, 4, 5, 6, 7, 1, 2])
        >>> _rotate_array([1, 2, 3, 4, 5, 6, 7], -2)
        array([6, 7, 1, 2, 3, 4, 5])

    """
    return np.concatenate([arr[n:], arr[:n]])


def make_toneburst(num_cycles, centre_freq, dt, num_samples=None, wrap=False,
                   analytical=False):
    """
    Returns a toneburst defined by centre frequency and a number of cycles.

    The signal is windowed by a Hann window (strictly zero outside the window). The
    toneburst is always symmetrical and its maximum is 1.0.

    Parameters
    ----------
    num_cycles : int
        Number of cycles of the toneburst.
    centre_freq : float
        Centre frequency
    dt : float
        Time step
    num_samples : int or None
        Number of time points. If None, returns a time vector that contains
        exactly the the toneburst. If larger, pads with zeros.
    wrap : bool, optional
        If False, the signal starts at n=0. If True, the signal is wrapped around such
         as its maximum is at n=0. The beginning of the signal is at the end of the vector.
         Default: False.
    analytical : bool, optional
        If True, returns the corresponding analytical signal (cos(...) + i sin(...)).
        Default: False.

    Returns
    -------
    toneburst : ndarray
        Array of length ``num_samples``

    """
    if dt <= 0.:
        raise ValueError('negative time step')
    if centre_freq <= 0.:
        raise ValueError('negative centre frequency')
    if num_cycles <= 0:
        raise ValueError('negative number of cycles')
    if num_samples is not None and num_samples <= 0:
        raise ValueError('negative number of time samples')

    len_pulse = int(np.ceil(num_cycles / centre_freq / dt))
    # force an odd length for pulse symmetry
    if len_pulse % 2 == 0:
        len_pulse += 1
    half_len_window = len_pulse // 2

    if num_samples is None:
        num_samples = len_pulse
    if len_pulse > num_samples:
        raise ValueError('time vector is too short for this pulse')

    t = np.arange(len_pulse)
    if analytical:
        sig = np.exp(2j * np.pi * dt * centre_freq * (t - half_len_window))
    else:
        sig = cos(2 * np.pi * dt * centre_freq * (t - half_len_window))
    window = np.hanning(len_pulse)

    toneburst = sig * window
    full_toneburst = np.zeros(num_samples, toneburst.dtype)
    full_toneburst[:len_pulse] = toneburst

    if wrap:
        full_toneburst = _rotate_array(full_toneburst, half_len_window)

    return full_toneburst
