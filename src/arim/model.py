"""
Core functions of the forward models.

.. seealso::
    :mod:`arim.models`
    :mod:`arim.scat`
    :mod:`arim.ut`

"""
# This module is imported on demand. It should be imported only for modelling.
# Function that are not modelling-specific should go to arim.ut, which is always imported.

import abc
import logging
import math
import os
import warnings
from collections import namedtuple

import numba
import numpy as np
from numpy.core.umath import cos, sin

from . import _scat, helpers, signal
from . import core as c

logger = logging.getLogger(__name__)

use_parallel = os.environ.get("ARIM_USE_PARALLEL", not numba.core.config.IS_32BITS)


def make_toneburst(
    num_cycles, centre_freq, dt, num_samples=None, wrap=False, analytical=False
):
    """
    Returns a toneburst defined by centre frequency and a number of cycles.

    The signal is windowed by a Hann window (strictly zero outside the window). The
    toneburst is always symmetrical and its maximum is 1.0.

    With ``wrap=False``, the result is made up of (in this order) the toneburst then zeros
    (controlled by ``num_samples``).
    With ``wrap=True``, the result is made up of (in this order) the second half of the toneburst,
    then zeros, then the first half of the toneburst.

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

    See Also
    --------
    :func:`make_toneburst2`

    """
    if dt <= 0.0:
        raise ValueError("negative time step")
    if centre_freq <= 0.0:
        raise ValueError("negative centre frequency")
    if num_cycles <= 0:
        raise ValueError("negative number of cycles")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("negative number of time samples")

    len_pulse = int(np.ceil(num_cycles / centre_freq / dt))
    # force an odd length for pulse symmetry
    if len_pulse % 2 == 0:
        len_pulse += 1
    half_len_window = len_pulse // 2

    if num_samples is None:
        num_samples = len_pulse
    if len_pulse > num_samples:
        raise ValueError("time vector is too short for this pulse")

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


def _rotate_array(arr, n):
    """
    >>> _rotate_array([1, 2, 3, 4, 5, 6, 7], 2)
    array([3, 4, 5, 6, 7, 1, 2])
    >>> _rotate_array([1, 2, 3, 4, 5, 6, 7], -2)
    array([6, 7, 1, 2, 3, 4, 5])

    """
    return np.concatenate([arr[n:], arr[:n]])


def make_toneburst2(
    num_cycles,
    centre_freq,
    dt,
    num_before=2,
    num_after=1,
    analytical=False,
    use_fast_len=True,
):
    """
    Returns a toneburst defined by centre frequency and a number of cycles.

    The result array is made up of (in this order) zeros (number controlled by ``num_before``),
    then the toneburst, then zeros (number controlled by ``num_after``).

    Parameters
    ----------
    num_cycles : int
        Number of cycles of the toneburst.
    centre_freq : float
        Centre frequency
    dt : float
        Time step
    num_before : int, optional
        Amount of zeros before the toneburst (in toneburst length).
    num_after : int, optional
        Amount of zeros after the toneburst (in toneburst length).
    analytical : bool, optional
    use_fast_len : bool, optional
        Use a FFT-friendly length (the default is True).

    Returns
    -------
    toneburst_time : arim.core.Time
    toneburst : ndarray
    t0_idx : int
        Index of the time sample ``t=0``.

    See Also
    --------
    :func:`make_toneburst`
    """

    signal = make_toneburst(
        num_cycles, centre_freq, dt, num_samples=None, wrap=False, analytical=analytical
    )
    n = len(signal)
    m = num_before * n
    p = num_after * n

    toneburst_len = m + n + p
    if use_fast_len:
        import scipy.fftpack

        toneburst_len = scipy.fftpack.next_fast_len(toneburst_len)
    toneburst = np.zeros(toneburst_len, dtype=signal.dtype)
    toneburst[m : m + n] = signal

    t0_idx = m + n // 2
    toneburst_time = c.Time(-t0_idx * dt, dt, len(toneburst))

    return toneburst_time, toneburst, t0_idx


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
        raise ValueError("Negative width")
    if wavelength < 0:
        raise ValueError("Negative wavelength")

    # /!\ numpy.sinc defines sinc(x) := sin(pi * x)/(pi * x)
    x = (element_width / wavelength) * np.sin(theta)
    return np.sinc(x)


def directivity_2d_rectangular_in_fluid_for_path(
    ray_geometry, element_width, wavelength
):
    """
    Wrapper for :func:`directivity_2d_rectangular_in_fluid` that uses a
    :class:`RayGeometry` object.

    Parameters
    ----------
    ray_geometry : arim.ray.RayGeometry
    element_width : float
    wavelength : float

    Returns
    -------
    directivity : ndarray
        Signed directivity for each angle.
    """
    return directivity_2d_rectangular_in_fluid(
        ray_geometry.conventional_out_angle(0), element_width, wavelength
    )


def _f0(x, k2):
    # Miller and Pursey 1954 eq (74)
    x2 = x * x
    # Warning: sqrt(a) * sqrt(b) != sqrt(a * b) because of negative values
    return (2 * x2 - k2) ** 2 - 4 * x2 * np.sqrt(x2 - 1) * np.sqrt(x2 - k2)


def directivity_2d_rectangular_on_solid_l(
    theta, element_width, wavelength_l, wavelength_t
):
    """
    L-wave directivity of rectangular element on solid

    The element is modelled by an infinitely long strip of finite width
    vibrating in a direction normal to the surface of the solid medium.

    Parameters
    ----------
    theta : ndarray
        Angles in radians.
    element_width : float
    wavelength_l : float
    wavelength_t : float

    Returns
    -------
    directivity_l : ndarray
        Complex

    Notes
    -----
    Equations MP (93) and DW (2), (3), (6)

    The sinc results of the integration of MP (90) with far field
    approximation.

    Normalisation coefficients are ignored, but the values are consistent with
    :func:`directivity_2d_rectangular_on_solid_t`.

    References
    ----------
    Miller, G. F., and H. Pursey. 1954. ‘The Field and Radiation Impedance of
    Mechanical Radiators on the Free Surface of a Semi-Infinite Isotropic
    Solid’. Proceedings of the Royal Society of London A: Mathematical,
    Physical and Engineering Sciences 223 (1155): 521–41.
    https://doi.org/10.1098/rspa.1954.0134.

    Drinkwater, Bruce W., and Paul D. Wilcox. 2006. ‘Ultrasonic Arrays for
    Non-Destructive Evaluation: A Review’. NDT & E International 39 (7):
    525–41. https://doi.org/10.1016/j.ndteint.2006.03.006.

    See Also
    --------
    :func:`directivity_2d_rectangular_on_solid_t`

    """
    k = wavelength_l / wavelength_t
    k2 = k * k
    theta = np.asarray(theta).astype(np.complex_)
    S = sin(theta)
    C = cos(theta)
    return (
        ((k2 - 2 * S**2) * C)
        / _f0(S, k2)
        * np.sinc((element_width / wavelength_l) * S)
    )


def directivity_2d_rectangular_on_solid_t(
    theta, element_width, wavelength_l, wavelength_t
):
    """
    T-wave directivity of rectangular element on solid

    See :func:`directivity_2d_rectangular_on_solid_l` for further information.

    Parameters
    ----------
    theta : ndarray
        Angles in radians.
    element_width : float
    wavelength_l : float
    wavelength_t : float

    Returns
    -------
    directivity_t : ndarray
        Complex

    Notes
    -----
    Equations MP (94) and DW (2), (4), (6)

    See Also
    --------
    :func:`directivity_2d_rectangular_on_solid_t`

    """
    k = wavelength_l / wavelength_t
    k2 = k * k
    theta = np.asarray(theta).astype(np.complex_)
    S = sin(theta)
    return (
        k**2.5
        * (np.sqrt(k2 * S * S - 1) * sin(2 * theta))
        / _f0(k * S, k2)
        * np.sinc((element_width / wavelength_t) * S)
    )


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


def fluid_solid(
    alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_l=None, alpha_t=None
):
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

    N = _fluid_solid_n(
        alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t
    )

    # Eq A.7
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)

    reflection = (
        ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
        + cos_2_alpha_t * cos_2_alpha_t
        - (rho_fluid * c_fluid * cos(alpha_l)) / (rho_solid * c_l * cos(alpha_fluid))
    ) / N

    # Eq A.8
    transmission_l = 2.0 * cos_2_alpha_t / N

    # Eq A.9
    transmission_t = -2.0 * ct_cl2 * sin(2 * alpha_l) / N

    return reflection, transmission_l, transmission_t


def _fluid_solid_n(
    alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t
):
    """
    Coefficient N defined by Krautkrämer in equation (A8).
    """
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)
    N = (
        ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
        + cos_2_alpha_t * cos_2_alpha_t
        + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid)
    )
    return N


def solid_l_fluid(
    alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid=None, alpha_t=None
):
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

    N = _fluid_solid_n(
        alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t
    )

    # Eq A.10
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)
    reflection_l = (
        ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
        - cos_2_alpha_t * cos_2_alpha_t
        + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid)
    ) / N

    # Eq A.11
    reflection_t = (2 * ct_cl2 * sin(2 * alpha_l) * cos(2 * alpha_t)) / N

    # Eq A.12
    transmission = (
        2
        * rho_fluid
        * c_fluid
        * cos(alpha_l)
        * cos(2 * alpha_t)
        / (N * rho_solid * c_l * cos(alpha_fluid))
    )

    return reflection_l, reflection_t, transmission


def solid_t_fluid(
    alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid=None, alpha_l=None
):
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

    N = _fluid_solid_n(
        alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t
    )

    # Eq A.14
    reflection_l = -sin(4 * alpha_t) / N

    # Eq A.13
    ct_cl2 = (c_t * c_t) / (c_l * c_l)
    cos_2_alpha_t = cos(2 * alpha_t)
    reflection_t = (
        ct_cl2 * sin(2 * alpha_l) * sin(2 * alpha_t)
        - cos_2_alpha_t * cos_2_alpha_t
        - rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid)
    ) / N

    # Eq A.15
    transmission = (
        2
        * rho_fluid
        * c_fluid
        * cos(alpha_l)
        * sin(2 * alpha_t)
        / (N * rho_solid * c_l * cos(alpha_fluid))
    )

    # TODO: Rose in "Ultrasonic guided waves in solid media" gives the oppositeof these
    # coefficients. Fix this?
    return reflection_l, reflection_t, transmission


def transmission_at_interface(
    interface_kind,
    material_inc,
    material_out,
    mode_inc,
    mode_out,
    angles_inc,
    force_complex=True,
    unit="stress",
):
    """
    Compute the transmission coefficients for an interface.

    The angles of transmission or reflection are obtained using Snell-Descartes laws
    (:func:`snell_angles`).
    Warning: do not check whether the angles of incidence are physical.

    Warning: only fluid-to-solid interface is implemented yet.

    TODO: write test.

    Parameters
    ----------
    interface_kind : InterfaceKind
    material_inc : arim.Material
        Material of the incident ray legs.
    material_out : arim.Material
        Material of the transmitted ray legs.
    mode_inc : Mode
        Mode of the incidents ray legs.
    mode_out : Mode
        Mode of the transmitted ray legs.
    angles_inc : ndarray
        Angle of incidence of the ray legs.
    force_complex : bool
        If True, return complex coefficients. If not, return coefficients with the same datatype as ``angles_inc``. Default: True.
    unit : str
        'stress' or 'displacement'. Default: 'stress'

    Returns
    -------
    amps : ndarray
        ``amps[i, j]`` is the transmission coefficient for the ray (i, j) at the given interface.

    """
    if force_complex:
        angles_inc = np.asarray(angles_inc, dtype=complex)
    if unit.lower() == "stress":
        convert_to_displacement = False
    elif unit.lower() == "displacement":
        convert_to_displacement = True
    else:
        raise ValueError("Argument 'unit' must be 'stress' or 'displacement'")

    if interface_kind is c.InterfaceKind.fluid_solid:
        # Fluid-solid interface in transmission
        #   "in" is in the fluid
        #   "out" is in the solid
        assert mode_inc is c.Mode.L, "you've broken the physics"

        fluid = material_inc
        solid = material_out
        assert solid.state_of_matter == c.StateMatter.solid
        assert fluid.state_of_matter.name != c.StateMatter.solid

        alpha_fluid = angles_inc
        alpha_l = snell_angles(
            alpha_fluid, fluid.longitudinal_vel, solid.longitudinal_vel
        )
        alpha_t = snell_angles(
            alpha_fluid, fluid.longitudinal_vel, solid.transverse_vel
        )

        params = dict(
            alpha_fluid=alpha_fluid,
            alpha_l=alpha_l,
            alpha_t=alpha_t,
            rho_fluid=fluid.density,
            rho_solid=solid.density,
            c_fluid=fluid.longitudinal_vel,
            c_l=solid.longitudinal_vel,
            c_t=solid.transverse_vel,
        )

        refl, trans_l, trans_t = fluid_solid(**params)
        if convert_to_displacement:
            # u2/u1 = z tau2 / tau1 = -z tau2 / p1
            z = (material_inc.density * material_inc.velocity(mode_inc)) / (
                material_out.density * material_out.velocity(mode_out)
            )
            trans_l *= z
            trans_t *= z
        if mode_out is c.Mode.L:
            return trans_l
        elif mode_out is c.Mode.T:
            return trans_t
        else:
            raise ValueError("invalid mode")
    elif interface_kind is c.InterfaceKind.solid_fluid:
        # Fluid-solid interface in transmission
        #   "in" is in the solid
        #   "out" is in the fluid
        assert mode_out is c.Mode.L, "you've broken the physics"

        solid = material_inc
        fluid = material_out
        assert solid.state_of_matter == c.StateMatter.solid
        assert fluid.state_of_matter.name != c.StateMatter.solid

        params = dict(
            rho_fluid=fluid.density,
            rho_solid=solid.density,
            c_fluid=fluid.longitudinal_vel,
            c_l=solid.longitudinal_vel,
            c_t=solid.transverse_vel,
        )

        if mode_inc is c.Mode.L:
            alpha_l = angles_inc
            refl_l, refl_t, transmission = solid_l_fluid(alpha_l=alpha_l, **params)
        elif mode_inc is c.Mode.T:
            alpha_t = angles_inc
            refl_l, refl_t, transmission = solid_t_fluid(alpha_t=alpha_t, **params)
        else:
            raise RuntimeError
        if convert_to_displacement:
            # u2/u1 = z tau2 / tau1 = -z tau2 / p1
            z = (material_inc.density * material_inc.velocity(mode_inc)) / (
                material_out.density * material_out.velocity(mode_out)
            )
            transmission *= z
        return transmission
    else:
        raise NotImplementedError


def reflection_at_interface(
    interface_kind,
    material_inc,
    material_against,
    mode_inc,
    mode_out,
    angles_inc,
    force_complex=True,
    unit="stress",
):
    """
    Compute the reflection coefficients for an interface.

    The angles of transmission or reflection are obtained using Snell-Descartes laws
    (:func:`snell_angles`).
    Warning: do not check whether the angles of incidence are physical.

    Warning: only fluid-to-solid interface is implemented yet.

    TODO: write test.

    Parameters
    ----------
    interface_kind : InterfaceKind
    material_inc : Material
        Material of the incident ray legs.
    material_against : Material
        Material of the reflected ray legs.
    mode_inc : Mode
        Mode of the incidents ray legs.
    mode_out : Mode
        Mode of the transmitted ray legs.
    angles_inc : ndarray
        Angle of incidence of the ray legs.
    force_complex : bool
        If True, return complex coefficients. If not, return coefficients with the same
        datatype as ``angles_inc``. Default: True.
    unit : str
        'stress' or 'displacement'. Default: 'stress'

    Returns
    -------
    amps : ndarray
        ``amps[i, j]`` is the reflection coefficient for the ray (i, j) at the given interface.

    """
    if force_complex:
        angles_inc = np.asarray(angles_inc, dtype=complex)
    if unit.lower() == "stress":
        convert_to_displacement = False
    elif unit.lower() == "displacement":
        convert_to_displacement = True
    else:
        raise ValueError("Argument 'unit' must be 'stress' or 'displacement'")

    if interface_kind is c.InterfaceKind.solid_fluid:
        # Reflection against a solid-fluid interface
        #   "in" is in the solid
        #   "out" is also in the solid
        solid = material_inc
        fluid = material_against
        assert solid.state_of_matter == c.StateMatter.solid
        assert fluid.state_of_matter != c.StateMatter.solid

        if mode_inc is c.Mode.L:
            angles_l = angles_inc
            angles_t = None
            solid_fluid = solid_l_fluid
        elif mode_inc is c.Mode.T:
            angles_l = None
            angles_t = angles_inc
            solid_fluid = solid_t_fluid
        else:
            raise ValueError("invalid mode")

        params = dict(
            alpha_fluid=None,
            alpha_l=angles_l,
            alpha_t=angles_t,
            rho_fluid=fluid.density,
            rho_solid=solid.density,
            c_fluid=fluid.longitudinal_vel,
            c_l=solid.longitudinal_vel,
            c_t=solid.transverse_vel,
        )
        with np.errstate(invalid="ignore"):
            refl_l, refl_t, trans = solid_fluid(**params)

        z = material_inc.velocity(mode_inc) / material_inc.velocity(mode_out)
        if mode_out is c.Mode.L:
            if convert_to_displacement:
                return refl_l * z
            else:
                return refl_l
        elif mode_out is c.Mode.T:
            if convert_to_displacement:
                return refl_t * z
            else:
                return refl_t
        else:
            raise ValueError("invalid mode")
    elif interface_kind is c.InterfaceKind.fluid_solid:
        # Reflection against a fluid-solid interface
        #   "in" is in the liquid
        #   "out" is also in the liquid
        solid = material_against
        fluid = material_inc
        assert solid.state_of_matter == c.StateMatter.solid
        assert fluid.state_of_matter != c.StateMatter.solid

        angles_fluid = angles_inc

        params = dict(
            alpha_fluid=angles_fluid,
            alpha_l=None,
            alpha_t=None,
            rho_fluid=fluid.density,
            rho_solid=solid.density,
            c_fluid=fluid.longitudinal_vel,
            c_l=solid.longitudinal_vel,
            c_t=solid.transverse_vel,
        )
        with np.errstate(invalid="ignore"):
            reflection, transmission_l, transmission_t = fluid_solid(**params)

        z = material_inc.velocity(mode_inc) / material_inc.velocity(mode_out)
        if convert_to_displacement:
            return reflection * z
        else:
            return reflection
    else:
        raise NotImplementedError


def transmission_reflection_for_path(
    path, ray_geometry, force_complex=True, unit="stress"
):
    """
    Return the transmission-reflection coefficients for a given path.

    This function takes into account all relevant interfaces defined in ``path``.

    Requires to have computed the angles of incidence at each interface
    (cf. :meth:`RayGeometry.conventional_inc_angle`).
    Use internally :func:`transmission_at_interface`` and :func:``reflection_at_interface``.

    The angles of transmission or reflection are obtained using Snell-Descartes laws (:func:`snell_angles`).

    Warning: do not check whether the angles of incidence are physical.

    Parameters
    ----------
    path : Path
    ray_geometry : arim.ray.RayGeometry
    force_complex : bool
        If True, return complex coefficients. If not, return coefficients with the same datatype as ``angles_inc``.
        Default: True.

    Yields
    ------
    amps : ndarray or None
        Amplitudes of transmission-reflection coefficients. None if not defined for all interface.
    """
    # Requires the incoming angles for all interfaces except the probe and the scatterers
    # (first and last).
    transrefl = None

    # For all interfaces but the first and last:
    for i, interface in enumerate(path.interfaces[1:-1], start=1):
        assert interface.transmission_reflection is not None

        params = dict(
            interface_kind=interface.kind,
            material_inc=path.materials[i - 1],
            mode_inc=path.modes[i - 1],
            mode_out=path.modes[i],
            angles_inc=ray_geometry.conventional_inc_angle(i),
            force_complex=force_complex,
            unit=unit,
        )

        logger.debug(
            "compute {} coefficients at interface {}".format(
                interface.transmission_reflection.name, interface.points
            )
        )

        if interface.transmission_reflection is c.TransmissionReflection.transmission:
            params["material_out"] = path.materials[i]
            tmp = transmission_at_interface(**params)
        elif interface.transmission_reflection is c.TransmissionReflection.reflection:
            params["material_against"] = interface.reflection_against
            tmp = reflection_at_interface(**params)
        else:
            raise RuntimeError

        if transrefl is None:
            transrefl = tmp
        else:
            transrefl *= tmp
        del tmp

    return transrefl


def reverse_transmission_reflection_for_path(
    path, ray_geometry, force_complex=True, unit="stress"
):
    """
    Return the transmission-reflection coefficients of the reverse path.

    This function uses the same angles as :func:`transmission_reflection_for_path`.
    These angles are the incident angles in the direct path (but the reflected/transmitted
    angles in the reverse path).

    Parameters
    ----------
    path : Path
    ray_geometry : arim.ray.RayGeometry
    force_complex : bool
        Use complex angles. Default : True

    Returns
    -------
    rev_transrefl : ndarray
        Shape: (path.interfaces[0].numpoints, path.interfaces[-1].numpoints)

    """
    transrefl = None

    # For all interfaces but the first and last:
    for i, interface in enumerate(path.interfaces[1:-1], start=1):
        assert interface.transmission_reflection is not None

        mode_inc = path.modes[i]
        material_inc = path.materials[i]
        mode_out = path.modes[i - 1]

        params = dict(
            material_inc=material_inc,
            mode_inc=mode_inc,
            mode_out=mode_out,
            force_complex=force_complex,
            unit=unit,
        )

        # In the reverse path, transmitted or reflected angles coming out the i-th
        # interface
        trans_or_refl_angles = ray_geometry.conventional_inc_angle(i)

        if interface.transmission_reflection is c.TransmissionReflection.transmission:
            material_out = path.materials[i - 1]
            params["material_out"] = material_out
            params["interface_kind"] = interface.kind.reverse()

            # Compute the incident angles in the reverse path from the incident angles in the
            # direct path using Snell laws.
            if force_complex:
                trans_or_refl_angles = np.asarray(trans_or_refl_angles, complex)
            params["angles_inc"] = snell_angles(
                trans_or_refl_angles,
                material_out.velocity(mode_out),
                material_inc.velocity(mode_inc),
            )
            tmp = transmission_at_interface(**params)

        elif interface.transmission_reflection is c.TransmissionReflection.reflection:
            params["material_against"] = interface.reflection_against
            params["interface_kind"] = interface.kind
            params["angles_inc"] = snell_angles(
                trans_or_refl_angles,
                material_inc.velocity(mode_out),
                material_inc.velocity(mode_inc),
            )
            tmp = reflection_at_interface(**params)

        else:
            raise RuntimeError

        if transrefl is None:
            transrefl = tmp
        else:
            transrefl *= tmp
        del tmp

    return transrefl


def beamspread_2d_for_path(ray_geometry):
    """
    Compute the 2D beamspread for a path. This function supports rays which goes through
    several interfaces (with virtual source).

    Only the (conventional) incoming angles are used, it is assumed in that the outgoing
    angles follow Snell laws.

    The beamspread has the dimension of 1/sqrt(r).

    In an unbounded medium::

        beamspread := 1/sqrt(r)

    Through one interface::

        beamspread := 1/sqrt(r1 + r2/beta)

    where::

        beta := (c1 * cos(theta2)^2) / (c2 * cos(theta1)^2)


    Parameters
    ----------
    ray_geometry : arim.ray.RayGeometry

    Returns
    -------
    beamspread : ndarray

    References
    ----------
    Schmerr, Fundamentals of ultrasonic phased arrays (Springer), §2.5

    """
    velocities = ray_geometry.rays.fermat_path.velocities

    # Using notations from forward model, this function computes the beamspread at A_n
    # where n = ray_geometry.numinterfaces - 1
    # Case n=0: undefined
    # Case n=1: beamspread = 1/sqrt(r)

    # Precompute gamma (coefficient of conversion between actual source
    # and virtual source)
    n = ray_geometry.numinterfaces - 1
    gamma_list = []
    for k in range(1, n):
        # k varies in [1, n-1] (included)
        theta_inc = ray_geometry.conventional_inc_angle(k)

        nu = velocities[k - 1] / velocities[k]
        sin_theta = np.sin(theta_inc)
        cos_theta = np.cos(theta_inc)
        gamma_list.append(
            (nu * nu - sin_theta * sin_theta) / (nu * cos_theta * cos_theta)
        )

    # Between the probe and the first interface, beamspread of an unbounded medium.
    # Use a copy because the original may be a cached value and we don'ray want
    # to change it by accident.
    virtual_distance = ray_geometry.inc_leg_size(1).copy()  # distance A_0 A_1

    for k in range(1, n):
        # distance A_k A_{k+1}:
        r = ray_geometry.inc_leg_size(k + 1)
        gamma = 1.0
        for i in range(k):
            gamma *= gamma_list[i]
        virtual_distance += r / gamma

    return np.reciprocal(np.sqrt(virtual_distance))


def reverse_beamspread_2d_for_path(ray_geometry):
    """
    Reverse beamspread for a path.
    Uses the same angles as in beamspread_2d_for_path for consistency.

    For a ray (i, ..., j), the reverse beamspread is obtained by considering the point
    j is the source and i is the endpoint. The direct beamspread considers this is the
    opposite.

    This gives the same result as beamspread_2d_for_path(reversed_ray_geometry)
    assuming the rays perfectly follow Snell laws. Because of errors in the ray tracing,
    there is a small difference.

    Parameters
    ----------
    ray_geometry : arim.ray.RayGeometry

    Returns
    -------
    rev_beamspread : ndarray
        Shape: (numelements, numgridpoints)

    """
    velocities = ray_geometry.rays.fermat_path.velocities
    # import pdb; pdb.set_trace()

    # Using notations from forward model, this function computes the beamspread at A_n
    # where n = ray_geometry.numinterfaces - 1
    # Case n=0: undefined
    # Case n=1: beamspread = 1/sqrt(r)

    # Precompute gamma (coefficient of conversion between actual source
    # and virtual source)
    n = ray_geometry.numinterfaces - 1
    gamma_list = []
    for k in range(1, n):
        # k varies in [1, n-1] (included)
        theta_out = ray_geometry.conventional_inc_angle(n - k)
        nu = velocities[n - k] / velocities[n - k - 1]
        sin_theta = np.sin(theta_out)
        cos_theta = np.cos(theta_out)
        # gamma expressed with theta_out instead of theta_in
        gamma_list.append(
            (nu * cos_theta * cos_theta) / (1 - nu * nu * sin_theta * sin_theta)
        )

    # Between the probe and the first interface, beamspread of an unbounded medium.
    # Use a copy because the original may be a cached value and we don'ray want
    # to change it by accident.
    virtual_distance = ray_geometry.inc_leg_size(n).copy()

    for k in range(1, n):
        # distance A_k A_{k+1}:
        r = ray_geometry.inc_leg_size(n - k)
        gamma = 1.0
        for i in range(k):
            gamma *= gamma_list[i]
        virtual_distance += r / gamma

    return np.reciprocal(np.sqrt(virtual_distance))


def material_attenuation_for_path(path, ray_geometry, frequency):
    r"""
    Return material attenuation for each ray (between 0 and 1)

    .. math::

        M(\omega) = \exp(- \sum_i a_i(\omega) d_i)

    If no attenuation is provided, ignore silently.

    Reference: Schmerr chapter 9

    Parameters
    ----------
    path : Path
    ray_geometry : arim.ray.RayGeometry

    Returns
    -------
    attenuation : ndarray
        Shape: (numelements, numgridpoints)
    """
    log_att = np.zeros(
        (path.interfaces[0].points.numpoints, path.interfaces[-1].points.numpoints)
    )

    for k, (material, mode) in enumerate(zip(path.materials, path.modes), start=1):
        att_obj = material.attenuation(mode)
        if att_obj is None:
            continue
        else:
            att_coeff = att_obj(frequency)
            log_att -= att_coeff * ray_geometry.inc_leg_size(k)
    return np.exp(log_att)


def _nested_dict_to_flat_list(dictlike):
    if dictlike is None:
        return []
    else:
        try:
            values = dictlike.values()
        except AttributeError:
            # dictlike is a leaf:
            return [dictlike]
        # dictlike is not a leaf:
        all_values = []
        for value in values:
            # union of sets:
            all_values = _nested_dict_to_flat_list(value)
        return all_values


class RayWeights(
    namedtuple(
        "RayWeights",
        [
            "tx_ray_weights_dict",
            "rx_ray_weights_dict",
            "tx_ray_weights_debug_dict",
            "rx_ray_weights_debug_dict",
            "scattering_angles_dict",
        ],
    )
):
    """
    Data container for ray weights.

    Attributes
    ----------
    tx_ray_weights_dict : dict[arim.Path, ndarray]
        Each value has a shape of (numelements, numgridpoints)
    rx_ray_weights_dict : dict[arim.Path, ndarray]
        Each value has a shape of (numelements, numgridpoints)
    tx_ray_weights_debug_dict : dict
        See function tx_ray_weights
    rx_ray_weights_debug_dict : dict
        See function rx_ray_weights
    scattering_angles_dict : dict[arim.Path, ndarray]
        Each value has a shape of (numelements, numgridpoints)
    """

    @property
    def nbytes(self):
        all_arrays = []
        all_arrays += _nested_dict_to_flat_list(self.tx_ray_weights_dict)
        all_arrays += _nested_dict_to_flat_list(self.rx_ray_weights_dict)
        all_arrays += _nested_dict_to_flat_list(self.tx_ray_weights_debug_dict)
        all_arrays += _nested_dict_to_flat_list(self.rx_ray_weights_debug_dict)
        all_arrays += _nested_dict_to_flat_list(self.scattering_angles_dict)
        # an array is not hashable so we cheat a bit to get unique arrays
        unique_ids = set(id(x) for x in all_arrays)
        nbytes = 0
        for arr in all_arrays:
            if id(arr) in unique_ids:
                nbytes += arr.nbytes
                unique_ids.remove(id(arr))
        return nbytes


def model_amplitudes_factory(tx, rx, view, ray_weights, scattering, scat_angle=0.0):
    """
    Calculates the model coefficients once the ray weights are known.

    The effective scattering is ``scattering(inc_theta - scat_angle, out_theta - scat_angle)``

    Parameters
    ----------
    tx : ndarray
    rx : ndarray
    view : View
    ray_weights : RayWeights
    scattering : dict
        Dict of functions (slow but precise) or matrices(fast but precision depends on
        the angle sampling).
    scat_angle : float

    Returns
    ------
    model_amplitudes : ModelAmplitudes
        Object that is indexable with a grid point index or a slice of grid points. The
         values are computed on the fly.

        can be indexe as an array but that computes the
        Function that returns the model amplitudes and takes as argument a slice.
        ndarray
        Shape: (blocksize, numtimetraces)
        Yield until all grid points are processed.

    Examples
    --------
    >>> model_amplitudes = model_amplitudes_factory(tx, rx, view, ray_weights, scattering)
    >>> model_amplitudes[0]
    # returns the 'numtimetraces' amplitudes at the grid point 0
    >>> model_amplitudes[:10] # returns the amplitudes for the first 10 grid points
    array([ 0.27764253,  0.78863332,  0.83998295,  0.96811351,  0.57929045, 0.00935137,  0.8905348 ,  0.46976061,  0.08101099,  0.57615469])
    >>> model_amplitudes[...] # returns the amplitudes for all points. Warning: you may
    ... # run out of memory!
    array([...])

    """
    # Pick the right scattering matrix/function.
    # scat_key is LL, LT, TL or TT
    scattering_obj = scattering[view.scat_key()]

    try:
        scattering_obj.shape
    except AttributeError:
        is_scattering_func = True
    else:
        is_scattering_func = False

    tx_ray_weights = ray_weights.tx_ray_weights_dict[view.tx_path]
    rx_ray_weights = ray_weights.rx_ray_weights_dict[view.rx_path]
    tx_scattering_angles = ray_weights.scattering_angles_dict[view.tx_path]
    rx_scattering_angles = ray_weights.scattering_angles_dict[view.rx_path]

    assert (
        tx_ray_weights.shape
        == rx_ray_weights.shape
        == tx_scattering_angles.shape
        == rx_scattering_angles.shape
    )

    # the great transposition
    tx_ray_weights = tx_ray_weights.T
    rx_ray_weights = rx_ray_weights.T
    tx_scattering_angles = tx_scattering_angles.T
    rx_scattering_angles = rx_scattering_angles.T

    if is_scattering_func:
        return _ModelAmplitudesWithScatFunction(
            tx,
            rx,
            scattering_obj,
            tx_ray_weights,
            rx_ray_weights,
            tx_scattering_angles,
            rx_scattering_angles,
            scat_angle,
        )
    else:
        return _ModelAmplitudesWithScatMatrix(
            tx,
            rx,
            scattering_obj,
            tx_ray_weights,
            rx_ray_weights,
            tx_scattering_angles,
            rx_scattering_angles,
            scat_angle,
        )


class ModelAmplitudes(abc.ABC):
    """
    Class for on-the-fly calculation of model amplitudes.


    Pseudo-array of coefficients P_ij = Q_i Q'_j S_ij. Shape: (numpoints, numtimetraces)

    This object can be indexed almost like a regular Numpy array.
    When indexed, the values are computed on the fly.
    Otherwise an array of this size would be too large.

    .. warning::
        Only the first dimension must be indexed. See examples below.

    Examples
    --------
    >>> model_amplitudes = model_amplitudes_factory(tx, rx, view, ray_weights, scattering_dict)

    This object is not an array:
    >>> type(model_amplitudes)
    __main__.ModelAmplitudes

    But when indexed, it returns an array:
    >>> type(model_amplitudes[0])
    numpy.ndarray

    Get the P_ij for the first grid point (returns an array of size (numtimetraces,)):
    >>> model_amplitudes[0]

    Get the P_ij for the first ten grid points (returns an array of size
    (10, numtimetraces,)):
    >>> model_amplitudes[:10]

    Get all P_ij (may run out of memory):
    >>> model_amplitudes[...]

    To get the first Get all P_ij (may run out of memory):
    >>> model_amplitudes[...]

    Indexing the second dimension will fail. For example to model amplitude of
    the fourth point and the eigth timetrace, use:
    >>> model_amplitudes[3][7]  # valid usage
    >>> model_amplitudes[3, 7]  # invalid usage, raise an IndexError
    """

    @abc.abstractmethod
    def __getitem__(self, grid_slice):
        ...

    @property
    def shape(self):
        return (self.numpoints, self.numtimetraces)

    def sensitivity_uniform_tfm(self, timetrace_weights, **kwargs):
        # wrapper in general case, inherit and write a faster implementation if possible
        return sensitivity_uniform_tfm(self, timetrace_weights, **kwargs)

    def sensitivity_model_assisted_tfm(self, timetrace_weights, **kwargs):
        # wrapper in general case, inherit and write a faster implementation if possible
        return sensitivity_model_assisted_tfm(self, timetrace_weights, **kwargs)


class _ModelAmplitudesWithScatFunction(ModelAmplitudes):
    def __init__(
        self,
        tx,
        rx,
        scattering_fn,
        tx_ray_weights,
        rx_ray_weights,
        tx_scattering_angles,
        rx_scattering_angles,
        scat_angle=0.0,
    ):
        self.tx = tx
        self.rx = rx
        self.scattering_fn = scattering_fn
        self.tx_ray_weights = tx_ray_weights
        self.rx_ray_weights = rx_ray_weights
        self.tx_scattering_angles = tx_scattering_angles
        self.rx_scattering_angles = rx_scattering_angles
        self.numpoints, self.numelements = tx_ray_weights.shape
        self.numtimetraces = self.tx.shape[0]
        self.scat_angle = scat_angle
        self.dtype = np.complex_

    def __getitem__(self, grid_slice):
        # Nota bene: arrays' shape is (numpoints, numtimetrace), i.e. the transpose
        # of RayWeights. They are contiguous.
        if np.empty(self.numpoints)[grid_slice].ndim > 1:
            raise IndexError("Only the first dimension of the object is indexable.")

        scat_angle = self.scat_angle
        scattering_amplitudes = self.scattering_fn(
            np.take(self.tx_scattering_angles[grid_slice], self.tx, axis=-1)
            - scat_angle,
            np.take(self.rx_scattering_angles[grid_slice], self.rx, axis=-1)
            - scat_angle,
        )

        model_amplitudes = (
            scattering_amplitudes
            * np.take(self.tx_ray_weights[grid_slice], self.tx, axis=-1)
            * np.take(self.rx_ray_weights[grid_slice], self.rx, axis=-1)
        )
        return model_amplitudes


@numba.guvectorize(
    "void(int_[:], int_[:], complex128[:,:], complex128[:], complex128[:], float64[:], float64[:], float64[:], complex128[:])",
    "(n),(n),(s,s),(e),(e),(e),(e),()->(n)",
    nopython=True,
    target="parallel",
)
def _model_amplitudes_with_scat_matrix(
    tx,
    rx,
    scattering_matrix,
    tx_ray_weights,
    rx_ray_weights,
    tx_scattering_angles,
    rx_scattering_angles,
    scat_angle,
    res,
):
    # This is a kernel on a grid point.
    numtimetraces = tx.shape[0]
    # assert res.shape[0] == tx_ray_weights.shape[0]
    # assert tx_ray_weights.shape == rx_ray_weights.shape == tx_scattering_angles.shape == rx_scattering_angles.shape
    for scan in range(numtimetraces):
        inc_theta = tx_scattering_angles[tx[scan]] - scat_angle[0]
        out_theta = rx_scattering_angles[rx[scan]] - scat_angle[0]

        scattering_amp = _scat._interpolate_scattering_matrix_kernel(
            scattering_matrix, inc_theta, out_theta
        )
        res[scan] = scattering_amp * tx_ray_weights[tx[scan]] * rx_ray_weights[rx[scan]]


class _ModelAmplitudesWithScatMatrix(ModelAmplitudes):
    def __init__(
        self,
        tx,
        rx,
        scattering_mat,
        tx_ray_weights,
        rx_ray_weights,
        tx_scattering_angles,
        rx_scattering_angles,
        scat_angle=0.0,
    ):
        self.tx = tx
        self.rx = rx
        self.scattering_mat = scattering_mat
        self.tx_ray_weights = tx_ray_weights
        self.rx_ray_weights = rx_ray_weights
        self.tx_scattering_angles = tx_scattering_angles
        self.rx_scattering_angles = rx_scattering_angles
        self.numpoints, self.numelements = tx_ray_weights.shape
        self.numtimetraces = self.tx.shape[0]
        self.scat_angle = scat_angle
        self.dtype = np.result_type(tx_ray_weights, rx_ray_weights, scattering_mat)

    def __getitem__(self, grid_slice):
        # Nota bene: arrays' shape is (numpoints, numtimetrace), i.e. the transpose
        # of RayWeights. They are contiguous.
        return _model_amplitudes_with_scat_matrix(
            self.tx,
            self.rx,
            self.scattering_mat,
            self.tx_ray_weights[grid_slice],
            self.rx_ray_weights[grid_slice],
            self.tx_scattering_angles[grid_slice],
            self.rx_scattering_angles[grid_slice],
            self.scat_angle,
        )


def sensitivity_uniform_tfm(model_amplitudes, timetrace_weights, block_size=4000):
    """
    Return the sensitivity for uniform TFM.

    The sensitivity at a point is defined the predicted TFM amplitude that a sole
    scatterer centered on that point would have.

    Parameters
    ----------
    model_amplitudes : ndarray or ModelAmplitudes
        Coefficients P_ij. Shape: (numpoints, numtimetraces)
    timetrace_weights : ndarray
        Shape: (numtimetraces, )

    Returns
    -------
    predicted_intensities
        Shape: (numpoints, )
    """
    numpoints, numtimetraces = model_amplitudes.shape
    assert timetrace_weights.ndim == 1
    assert model_amplitudes.shape[1] == timetrace_weights.shape[0]

    sensitivity = None

    # chunk the array in case we have an array too big (ModelAmplitudes)
    for chunk in helpers.chunk_array((numpoints, numtimetraces), block_size):
        tmp = (timetrace_weights[np.newaxis] * model_amplitudes[chunk]).sum(axis=1)
        if sensitivity is None:
            sensitivity = np.zeros((numpoints,), dtype=tmp.dtype)
        sensitivity[chunk] = tmp
    sensitivity /= numtimetraces
    return sensitivity


def sensitivity_model_assisted_tfm(
    model_amplitudes, timetrace_weights, block_size=4000
):
    """
    Return the sensitivity for model assisted TFM (multiply TFM timetraces by conjugate
    of scatterer contribution).

    The sensitivity at a point is defined the predicted TFM amplitude that a sole
    scatterer centered on that point would have.

    Parameters
    ----------
    model_amplitudes : ndarray or ModelAmplitudes
        Coefficients P_ij. Shape: (numpoints, numtimetraces)
    timetrace_weights : ndarray
        Shape: (numtimetraces, )

    Returns
    -------
    predicted_intensities
        Shape: (numpoints, ).
    """
    numpoints, numtimetraces = model_amplitudes.shape
    assert timetrace_weights.ndim == 1
    assert model_amplitudes.shape[1] == timetrace_weights.shape[0]

    sensitivity = None

    # chunk the array in case we have an array too big (ModelAmplitudes)
    for chunk in helpers.chunk_array((numpoints, numtimetraces), block_size):
        absval = np.abs(model_amplitudes[chunk])
        tmp = (absval * absval * timetrace_weights[np.newaxis]).sum(axis=1)
        if sensitivity is None:
            sensitivity = np.zeros((numpoints,), dtype=tmp.dtype)
        sensitivity[chunk] = tmp
    sensitivity /= numtimetraces
    return sensitivity


@numba.njit(parallel=use_parallel, nogil=True)
def _timeshift_timedomain(unshifted_response, delays, dt, t0_idx, out):
    n = unshifted_response.shape[1]
    for idx in numba.prange(unshifted_response.shape[0]):
        delay_idx = math.floor(delays[idx] / dt)
        out[idx, delay_idx - t0_idx : delay_idx - t0_idx + n] += unshifted_response[idx]


def transfer_func_to_timetraces(
    unshifted_transfer_func,
    delays,
    timetraces_time,
    toneburst_time,
    toneburst_freq,
    toneburst_f,
    toneburst_t0_idx,
    timetraces=None,
):
    """Returns time-domain timetraces from the unshifted transfer function and the toneburst

    Parameters
    ----------
    unshifted_transfer_func : ndarray
        Transfer function to apply, without the "exp(-i omega tau)" term.
        Shape ``(numtimetraces, numfreq)`` or ``(numscatterers, numtimetraces, numfreq)``
    delays : ndarray
        Delay for each scatterer and timetrace.
        Shape ``(numtimetraces,)`` or ``(numscatterers, numtimetraces)``
    timetraces_time : arim.core.Time
    toneburst_time : arim.core.Time
    toneburst_freq : ndarray
        Frequency array of the toneburst, obtained typically with ``np.fft.rfftfreq``
    toneburst_f : ndarray
        Spectrum of the toneburst, obtained with ``np.fft.rfft``.
    toneburst_t0_idx : [type]
        Index so that ``toneburst_time.samples[t0_idx] = 0.``
    timetraces : ndarray
        Optional, write on this array if provided.

    Returns
    -------
    timetraces : ndarray
        timetraces

    """
    if unshifted_transfer_func.ndim == 2:
        unshifted_transfer_func = unshifted_transfer_func.reshape(
            (1, *unshifted_transfer_func.shape)
        )
    if delays.ndim == 1:
        delays = delays.reshape((1, *delays.shape))
    numscatterers, numtimetraces, _ = unshifted_transfer_func.shape
    assert delays.shape == (numscatterers, numtimetraces)

    if timetraces_time.step != toneburst_time.step:
        raise NotImplementedError
    dt = timetraces_time.step

    if timetraces is None:
        timetraces = np.zeros((numtimetraces, len(timetraces_time)), np.complex_)

    # Account for the timetraces t0
    delays = delays - timetraces_time.start
    assert np.all(delays >= 0.0)

    # Shift transfer func by the frac of the time step
    delays_remainder = delays % dt
    frac_shifted_transfer_func = signal.timeshift_spectra(
        unshifted_transfer_func, delays_remainder, toneburst_freq
    )

    # Calculate timedomain response
    frac_shifted_response = signal.rfft_to_hilbert(
        frac_shifted_transfer_func * toneburst_f, len(toneburst_time)
    )

    # Shift timedomain response by a multiple of the time step
    for scat_idx in range(numscatterers):
        _timeshift_timedomain(
            frac_shifted_response[scat_idx],
            delays[scat_idx],
            dt,
            toneburst_t0_idx,
            timetraces,
        )

    return timetraces


def transfer_func_to_scanlines(
    unshifted_transfer_func,
    delays,
    scanlines_time,
    toneburst_time,
    toneburst_freq,
    toneburst_f,
    toneburst_t0_idx,
    timetraces=None,
):
    warnings.warn(
        DeprecationWarning(
            "transfer_func_to_scanlines is deprecated. Use transfer_func_to_timetraces"
        )
    )
    return transfer_func_to_timetraces(
        unshifted_transfer_func,
        delays,
        scanlines_time,
        toneburst_time,
        toneburst_freq,
        toneburst_f,
        toneburst_t0_idx,
        timetraces,
    )
