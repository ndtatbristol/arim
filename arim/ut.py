"""
Toolbox of functions for ultrasonic testing/acoustics.
"""
# Only function that does not require any arim-specific logic should be put here.
# This module must be kept free of any arim dependencies because so that it could be used
# without arim.

import numpy as np
import warnings

from numpy.core.umath import sin, cos


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
    For a signal $x(t) = A * exp(i (2 pi f_0 t + phi(t)))$, returns phi(t) in [-pi, pi[.

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
                      'transform to get the analytic signal.'.format(dtype), UtWarning)
    phase_correction = 2 * np.pi * carrier_frequency * time_vect
    phase = wrap_phase(np.angle(analytic_sig) - phase_correction)
    return phase


def directivity_finite_width_2d(theta, element_width, wavelength):
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

    ..  [WO] Wooh, Shi-Chang, and Yijun Shi. 1999. ‘Three-Dimensional Beam Directivity of Phase-Steered Ultrasound’.
             The Journal of the Acoustical Society of America 105 (6): 3275–82. doi:10.1121/1.424655.

    """
    if element_width < 0:
        raise ValueError('Negative width')
    if wavelength < 0:
        raise ValueError('Negative wavelength')

    # /!\ numpy.sinc defines sinc(x) := sin(pi * x)/(pi * x)
    x = element_width * np.sin(theta) / wavelength
    return np.sinc(x)


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
    N = (c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) \
        + cos(2 * alpha_t) ** 2 \
        + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid)
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
    reflection = ((c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) \
                  + cos(2 * alpha_t) ** 2 \
                  - (rho_fluid * c_fluid * cos(alpha_l)) / (
                      rho_solid * c_l * cos(alpha_fluid))) / N

    # Eq A.8
    transmission_l = 2. * cos(2 * alpha_t) / N

    # Eq A.9
    transmission_t = -2. * (c_t / c_l) ** 2 * sin(2 * alpha_l) / N

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
    reflection_l = ((c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) - cos(
        2 * alpha_t) ** 2 \
                    + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(
        alpha_fluid)) / N

    # Eq A.11
    reflection_t = (2 * (c_t / c_l) ** 2 * sin(2 * alpha_l) * cos(2 * alpha_t)) / N

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
    reflection_t = ((c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) - cos(
        2 * alpha_t) ** 2 \
                    - rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(
        alpha_fluid)) / N

    # Eq A.15
    transmission = 2 * rho_fluid * c_fluid * cos(alpha_l) * sin(2 * alpha_t) / (
        N * rho_solid * c_l * cos(alpha_fluid))

    return reflection_l, reflection_t, transmission


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
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
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


def _rotate_array(arr, n):
    """
        >>> _rotate_array([1, 2, 3, 4, 5, 6, 7], 2)
        array([3, 4, 5, 6, 7, 1, 2])
        >>> _rotate_array([1, 2, 3, 4, 5, 6, 7], -2)
        array([6, 7, 1, 2, 3, 4, 5])

    """
    return np.concatenate([arr[n:], arr[:n]])


def make_toneburst(num_cycles, num_samples, dt, centre_freq, wrap=False,
                   analytical=False):
    """
    Returns a toneburst defined by centre frequency and a number of cycles.

    The signal is windowed by a Hann window (strictly zero outside the window). The
    toneburst is always symmetrical and its maximum is 1.0.

    Parameters
    ----------
    num_cycles : int
        Number of cycles of the toneburst.
    num_samples : int
        Number of the vector.
    dt : float
        Time step
    centre_freq : float
        Centre frequency
    wrap : bool, optional
        If False, the signal starts at n=0. If True, the signal is wrapped around such
         as its maximum is at n=0. The beginning of the signal is at the end of the vector.
         Default: False.
    analytical : bool, optional
        If True, returns the corresponding analytical signal (cos(...) + i sin(...)).

    Returns
    -------
    toneburst : ndarray
        Array of length ``num_samples``

    """
    if dt <= 0.:
        raise ValueError('negative time step')
    if dt <= 0.:
        raise ValueError('negative centre frequency')
    if num_cycles <= 0:
        raise ValueError('negative number of cycles')
    if num_samples <= 0:
        raise ValueError('negative number of time samples')

    len_pulse = int(round(num_cycles / centre_freq / dt))
    # force an odd length for pulse symmetry
    if len_pulse % 2 == 0:
        len_pulse += 1
    half_len_window = len_pulse // 2

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
