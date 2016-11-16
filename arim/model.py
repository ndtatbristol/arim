"""
Formulas for modelling the physics of ultrasonic testing.

"""
import logging

import numpy as np
from numpy import sin, cos

from . import core
from .core import Mode, InterfaceKind, StateMatter, TransmissionReflection

__all__ = ['directivity_finite_width_2d', 'fluid_solid', 'solid_l_fluid', 'solid_t_fluid', 'snell_angles',
           'transmission_at_interface', 'reflection_at_interface',
           'transmission_reflection_per_interface_for_path', 'transmission_reflection_for_path'
           'beamspread_after_interface', 'beamspread_per_interface_for_path', 'beamspread_for_path']

logger = logging.getLogger(__name__)


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
    directivity
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


def _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t):
    """
    Coefficient N defined by Krautkrämer in equation (A8).
    """
    N = (c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) \
        + cos(2 * alpha_t) ** 2 \
        + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid)
    return N


def fluid_solid(alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_l=None, alpha_t=None):
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

    N = _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t)

    # Eq A.7
    reflection = ((c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) \
                  + cos(2 * alpha_t) ** 2 \
                  - (rho_fluid * c_fluid * cos(alpha_l)) / (rho_solid * c_l * cos(alpha_fluid))) / N

    # Eq A.8
    transmission_l = 2. * cos(2 * alpha_t) / N

    # Eq A.9
    transmission_t = -2. * (c_t / c_l) ** 2 * sin(2 * alpha_l) / N

    return reflection, transmission_l, transmission_t


def solid_l_fluid(alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid=None, alpha_t=None):
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

    N = _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t)

    # Eq A.10
    reflection_l = ((c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) - cos(2 * alpha_t) ** 2 \
                    + rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid)) / N

    # Eq A.11
    reflection_t = (2 * (c_t / c_l) ** 2 * sin(2 * alpha_l) * cos(2 * alpha_t)) / N

    # Eq A.12
    transmission = 2 * rho_fluid * c_fluid * cos(alpha_l) * cos(2 * alpha_t) / (N * rho_solid * c_l * cos(alpha_fluid))

    return reflection_l, reflection_t, transmission


def solid_t_fluid(alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid=None, alpha_l=None):
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

    N = _fluid_solid_n(alpha_fluid, alpha_l, alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t)

    # Eq A.14
    reflection_l = -sin(4 * alpha_t) / N

    # Eq A.13
    reflection_t = ((c_t / c_l) ** 2 * sin(2 * alpha_l) * sin(2 * alpha_t) - cos(2 * alpha_t) ** 2 \
                    - rho_fluid * c_fluid / (rho_solid * c_l) * cos(alpha_l) / cos(alpha_fluid)) / N

    # Eq A.15
    transmission = 2 * rho_fluid * c_fluid * cos(alpha_l) * sin(2 * alpha_t) / (N * rho_solid * c_l * cos(alpha_fluid))

    return reflection_l, reflection_t, transmission


def transmission_at_interface(interface_kind, material_inc, material_out, mode_inc, mode_out, angles_inc,
                              force_complex=True):
    """
    Compute the transmission coefficients for an interface.

    The angles of transmission or reflection are obtained using Snell-Descartes laws (:func:`snell_angles`).
    Warning: do not check whether the angles of incidence are physical.

    Warning: only fluid-to-solid interface is implemented yet.

    TODO: write test.

    Parameters
    ----------
    interface_kind : InterfaceKind
    material_inc : Material
        Material of the incident ray legs.
    material_out : Material
        Material of the transmitted ray legs.
    mode_inc : Mode
        Mode of the incidents ray legs.
    mode_out : Mode
        Mode of the transmitted ray legs.
    angles_inc : ndarray
        Angle of incidence of the ray legs.
    force_complex : bool
        If True, return complex coefficients. If not, return coefficients with the same datatype as ``angles_inc``. Default: True.

    Returns
    -------
    amps : ndarray
        ``amps[i, j]`` is the transmission coefficient for the ray (i, j) at the given interface.

    """
    if force_complex:
        angles_inc = np.asarray(angles_inc, dtype=complex)

    if interface_kind is core.InterfaceKind.fluid_solid:
        # Fluid-solid interface in transmission
        #   "in" is in the fluid
        #   "out" is in the solid
        assert mode_inc is core.Mode.L, "you've broken the physics"

        fluid = material_inc
        solid = material_out
        assert solid.state_of_matter == StateMatter.solid
        assert fluid.state_of_matter.name != StateMatter.solid

        alpha_fluid = angles_inc
        alpha_l = snell_angles(alpha_fluid, fluid.longitudinal_vel, solid.longitudinal_vel)
        alpha_t = snell_angles(alpha_fluid, fluid.longitudinal_vel, solid.transverse_vel)

        params = dict(
            alpha_fluid=alpha_fluid,
            alpha_l=alpha_l,
            alpha_t=alpha_t,
            rho_fluid=fluid.density,
            rho_solid=solid.density,
            c_fluid=fluid.longitudinal_vel,
            c_l=solid.longitudinal_vel,
            c_t=solid.transverse_vel)

        refl, trans_l, trans_t = fluid_solid(**params)
        if mode_out is Mode.L:
            return trans_l
        elif mode_out is Mode.T:
            return trans_t
        else:
            raise ValueError("invalid mode")
    else:
        raise NotImplementedError


def reflection_at_interface(interface_kind, material_inc, material_against, mode_inc, mode_out, angles_inc,
                            force_complex=True):
    """
    Compute the reflection coefficients for an interface.

    The angles of transmission or reflection are obtained using Snell-Descartes laws (:func:`snell_angles`).
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
        If True, return complex coefficients. If not, return coefficients with the same datatype as ``angles_inc``. Default: True.

    Returns
    -------
    amps : ndarray
        ``amps[i, j]`` is the reflection coefficient for the ray (i, j) at the given interface.

    """
    if force_complex:
        angles_inc = np.asarray(angles_inc, dtype=complex)

    if interface_kind is InterfaceKind.solid_fluid:
        # Reflection against a solid-fluid interface
        #   "in" is in the solid
        #   "out" is also in the solid
        solid = material_inc
        fluid = material_against
        assert solid.state_of_matter == StateMatter.solid
        assert fluid.state_of_matter != StateMatter.solid

        if mode_inc is Mode.L:
            angles_l = angles_inc
            angles_t = None
            solid_fluid = solid_l_fluid
        elif mode_inc is Mode.T:
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
            c_t=solid.transverse_vel)
        with np.errstate(invalid='ignore'):
            refl_l, refl_t, trans = solid_fluid(**params)

        if mode_out is Mode.L:
            return refl_l
        elif mode_out is Mode.T:
            return refl_t
        else:
            raise ValueError("invalid mode")
    else:
        raise NotImplementedError


def transmission_reflection_per_interface_for_path(path, angles_inc_list, force_complex=True):
    """
    Yield the transmission-reflection coefficients interface per interface for a given path.

    For non-relevant interfaces  (attribute ``transmission_reflection`` set to None), yield None.

    Requires to have computed the angles of incidence at each interface (cf. :meth:`Rays.get_incoming_angles`).
    Use internally :func:`transmission_at_interface`` and :func:``reflection_at_interface``.

    The angles of transmission or reflection are obtained using Snell-Descartes laws (:func:`snell_angles`).

    Warning: do not check whether the angles of incidence are physical.

    TODO: write test.

    Parameters
    ----------
    path : Path
    angles_inc_list : list of ndarray
    force_complex : bool
        If True, return complex coefficients. If not, return coefficients with the same datatype as ``angles_inc``.
        Default: True.

    Yields
    ------
    amps : ndarray or None
        Amplitudes of transmission-reflection coefficients. None if not relevant.

    """
    for i, interface in enumerate(path.interfaces):
        if interface.transmission_reflection is None:
            yield None
            continue

        assert i > 0, "cannot compute transmission/reflection at the first interface. Set 'transmission_reflection' to None in Interface."

        params = dict(
            interface_kind=interface.kind,
            material_inc=path.materials[i - 1],
            mode_inc=path.modes[i - 1],
            mode_out=path.modes[i],
            angles_inc=angles_inc_list[i],
            force_complex=force_complex,
        )

        logger.info("compute {} coefficients at interface {}".format(interface.transmission_reflection.name,
                                                                     interface.points))

        if interface.transmission_reflection is TransmissionReflection.transmission:
            params['material_out'] = path.materials[i]
            yield transmission_at_interface(**params)
        elif interface.transmission_reflection is TransmissionReflection.reflection:
            params['material_against'] = interface.reflection_against
            yield reflection_at_interface(**params)
        else:
            raise ValueError('invalid constant for transmission/reflection')


def transmission_reflection_for_path(path, angles_inc_list, force_complex=True):
    """
    Return the transmission-reflection coefficients for a given path.

    This function takes into account all relevant interfaces defined in ``path``.

    Requires to have computed the angles of incidence at each interface (cf. :meth:`Rays.get_incoming_angles`).
    Use internally :func:`transmission_at_interface`` and :func:``reflection_at_interface``.

    The angles of transmission or reflection are obtained using Snell-Descartes laws (:func:`snell_angles`).

    Warning: do not check whether the angles of incidence are physical.

    TODO: write test.

    Parameters
    ----------
    path : Path
    angles_inc_list : list of ndarray
    force_complex : bool
        If True, return complex coefficients. If not, return coefficients with the same datatype as ``angles_inc``.
        Default: True.

    Yields
    ------
    amps : ndarray or None
        Amplitudes of transmission-reflection coefficients. None if not defined for all interface.
    """
    amps = None

    for amps_per_interface in transmission_reflection_per_interface_for_path(path, angles_inc_list, force_complex=force_complex):
        if amps_per_interface is None:
            continue
        if amps is None:
            amps = amps_per_interface
        else:
            amps *= amps_per_interface
    return amps


def beamspread_after_interface(inc_angles_last_interface, out_angles_last_interface,
                               leg_sizes_before_last_interface,
                               leg_sizes_after_last_interface, p=0.5):
    """
    Compute the contribution to beamspread of an interface.


    Parameters
    ----------
    inc_angles_last_interface
        Angles of the incoming rays at the last interface.
    out_angles_last_interface
        Angles of the outgoing rays at the last interface.
    leg_sizes_before_last_interface
        For each ray, size of the leg between the second to last interface and the last interface.
    leg_sizes_after_last_interface : ndarray
        For each ray, size of the leg between the last interface and the current point.
    p : float
        Beamspread is ``1/distance**p`` (distance power p). Use 0.5 for 2D and 1.0 for 3D. Default: 0.5

    Returns
    -------
    beamspread : ndarray

    """
    beta = (np.cos(out_angles_last_interface) ** 2 * np.sin(inc_angles_last_interface)) \
           / (np.cos(inc_angles_last_interface) ** 2 * np.sin(out_angles_last_interface))
    distance_virtual = leg_sizes_before_last_interface * beta
    return (distance_virtual / (distance_virtual + leg_sizes_after_last_interface)) ** p


def beamspread_per_interface_for_path(inc_angles_list, out_angles_list, inc_leg_sizes_list, p=0.5):
    """
    Compute the beamspread for a path as a list of the contribution of each interface.

    Parameters
    ----------
    inc_angles_list : list of ndarray
        Each array of the list is the angle of the incoming ray to the interface. One array per interface.
        None if not relevant.
    out_angles_list : list of ndarray
        Each array of the list is the angle of the outgoing ray from the interface. One array per interface.
        None if not relevant.
    inc_leg_sizes_list : list of ndarray
        Each array of the list is the size of the incoming leg to the interface. Legs are assumed to be straight.
        One array per interface.
        None if not relevant.
    p : float
        Beamspread is ``1/distance**p`` (distance power p). Use 0.5 for 2D and 1.0 for 3D. Default: 0.5

    Returns
    -------
    beamspread_list : list of ndarray

    """
    # At the first interface, beamspread is not defined
    yield None

    assert len(inc_angles_list) == len(out_angles_list) == len(inc_leg_sizes_list)
    numinterfaces = len(inc_angles_list)

    for i in range(1, numinterfaces):
        if i == 1:
            # Leg size between the probe and the first interface
            yield inc_leg_sizes_list[1] ** (-p)
        else:
            leg_sizes_before_last_interface = inc_leg_sizes_list[i - 1]
            leg_sizes_after_last_interface = inc_leg_sizes_list[i]

            inc_angles_last_interface = inc_angles_list[i - 1]
            out_angles_last_interface = out_angles_list[i - 1]

            yield beamspread_after_interface(inc_angles_last_interface, out_angles_last_interface,
                                             leg_sizes_before_last_interface,
                                             leg_sizes_after_last_interface, p=p)


def beamspread_for_path(inc_angles_list, out_angles_list, inc_leg_sizes_list, p=0.5):
    """
    Compute the beamspread for a path.

    Parameters
    ----------
    inc_angles_list : list of ndarray
        Each array of the list is the angle of the incoming ray to the interface. One array per interface.
        None if not relevant.
    out_angles_list : list of ndarray
        Each array of the list is the angle of the outgoing ray from the interface. One array per interface.
        None if not relevant.
    inc_leg_sizes_list : list of ndarray
        Each array of the list is the size of the incoming leg to the interface. Legs are assumed to be straight.
        One array per interface.
        None if not relevant.
    p : float
        Beamspread is ``1/distance**p`` (distance power p). Use 0.5 for 2D and 1.0 for 3D. Default: 0.5

    Returns
    -------
    beamspread : ndarray

    """
    beamspread = None
    for relative_beamspread in beamspread_per_interface_for_path(inc_angles_list, out_angles_list, inc_leg_sizes_list, p=p):
        if relative_beamspread is None:
            continue
        if beamspread is None:
            beamspread = relative_beamspread
        else:
            beamspread *= relative_beamspread
    return beamspread
