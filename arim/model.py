"""
Functions related to the forward model.

.. seealso::
    :mod:`arim.ut`

"""
# Functions that rely on arim-specific structures (Path, Frame, Interface, etc.) should
# be put here.
# Functions that rely on simpler logic should be put in arim.ut and imported here.

import logging
import warnings

import numpy as np

from . import core as c
from . import ut
from .ut import snell_angles, fluid_solid, solid_l_fluid, solid_t_fluid, \
    directivity_2d_rectangular_in_fluid

# for backward compatiblity:
from .path import RayGeometry

logger = logging.getLogger(__name__)


def radiation_2d_rectangular_in_fluid_for_path(ray_geometry, element_width, wavelength,
                                               impedance):
    """
    Wrapper for :func:`radiation_2d_rectangular_in_fluid` that uses
    a :class:`RayGeometry` object.


    Parameters
    ----------
    ray_geometry : RayGeometry
    element_width
    wavelength
    impedance

    Returns
    -------

    """
    return ut.radiation_2d_rectangular_in_fluid(ray_geometry.conventional_out_angle(0),
                                                element_width, wavelength, impedance)


def directivity_finite_width_2d_for_path(ray_geometry, element_width, wavelength):
    """
    Wrapper for :func:`directivity_2d_rectangular_in_fluid` that uses a
    :class:`RayGeometry` object.

    Parameters
    ----------
    ray_geometry : RayGeometry
    element_width : float
    wavelength : float

    Returns
    -------
    directivity : ndarray
        Signed directivity for each angle.
    """
    return directivity_2d_rectangular_in_fluid(ray_geometry.conventional_out_angle(0),
                                               element_width, wavelength)


def transmission_at_interface(interface_kind, material_inc, material_out, mode_inc,
                              mode_out, angles_inc,
                              force_complex=True):
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
        alpha_l = snell_angles(alpha_fluid, fluid.longitudinal_vel,
                               solid.longitudinal_vel)
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
        if mode_out is c.Mode.L:
            return trans_l
        elif mode_out is c.Mode.T:
            return trans_t
        else:
            raise ValueError("invalid mode")
    else:
        raise NotImplementedError


def reflection_at_interface(interface_kind, material_inc, material_against, mode_inc,
                            mode_out, angles_inc,
                            force_complex=True):
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

    Returns
    -------
    amps : ndarray
        ``amps[i, j]`` is the reflection coefficient for the ray (i, j) at the given interface.

    """
    if force_complex:
        angles_inc = np.asarray(angles_inc, dtype=complex)

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
            c_t=solid.transverse_vel)
        with np.errstate(invalid='ignore'):
            refl_l, refl_t, trans = solid_fluid(**params)

        if mode_out is c.Mode.L:
            return refl_l
        elif mode_out is c.Mode.T:
            return refl_t
        else:
            raise ValueError("invalid mode")
    else:
        raise NotImplementedError


def transmission_reflection_per_interface_for_path(path, angles_inc_list,
                                                   force_complex=True):
    """
    Yield the transmission-reflection coefficients interface per interface for a given path.

    For non-relevant interfaces  (attribute ``transmission_reflection`` set to None), yield None.

    Requires to have computed the angles of incidence at each interface
    (cf. :meth:`RayGeometry.conventional_inc_angle`).
    Use internally :func:`transmission_at_interface`` and :func:``reflection_at_interface``.

    The angles of transmission or reflection are obtained using Snell-Descartes laws
    (:func:`snell_angles`).

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

        logger.debug("compute {} coefficients at interface {}".format(
            interface.transmission_reflection.name,
            interface.points))

        if interface.transmission_reflection is c.TransmissionReflection.transmission:
            params['material_out'] = path.materials[i]
            yield transmission_at_interface(**params)
        elif interface.transmission_reflection is c.TransmissionReflection.reflection:
            params['material_against'] = interface.reflection_against
            yield reflection_at_interface(**params)
        else:
            raise ValueError('invalid constant for transmission/reflection')


def transmission_reflection_for_path(path, ray_geometry, force_complex=True):
    """
    Return the transmission-reflection coefficients for a given path.

    This function takes into account all relevant interfaces defined in ``path``.

    Requires to have computed the angles of incidence at each interface
    (cf. :meth:`RayGeometry.conventional_inc_angle`).
    Use internally :func:`transmission_at_interface`` and :func:``reflection_at_interface``.

    The angles of transmission or reflection are obtained using Snell-Descartes laws (:func:`snell_angles`).

    Warning: do not check whether the angles of incidence are physical.

    TODO: write test.

    Parameters
    ----------
    path : Path
    ray_geometry : RayGeometry
    force_complex : bool
        If True, return complex coefficients. If not, return coefficients with the same datatype as ``angles_inc``.
        Default: True.

    Yields
    ------
    amps : ndarray or None
        Amplitudes of transmission-reflection coefficients. None if not defined for all interface.
    """
    amps = None

    inc_angles_list = [ray_geometry.conventional_inc_angle(k) for k in
                       range(ray_geometry.numinterfaces)]

    for amps_per_interface in transmission_reflection_per_interface_for_path(path,
                                                                             inc_angles_list,
                                                                             force_complex=force_complex):
        if amps_per_interface is None:
            continue
        if amps is None:
            amps = amps_per_interface
        else:
            amps *= amps_per_interface
    return amps


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

        beamspread := 1/(sqrt(r1) * (1 + r2 / (beta * r1))

    where::

        beta := (c1 * cos(theta2)^2) / (c2 * cos(theta1)^2)


    Parameters
    ----------
    ray_geometry : arim.path.RayGeometry

    Returns
    -------
    beamspread : ndarray

    References
    ----------
    Schmerr, Fundamentals of ultrasonic phased arrays (Springer), §2.5

    """
    inc_angles_list = [ray_geometry.conventional_inc_angle(i) for i in
                       range(ray_geometry.numinterfaces)]
    inc_leg_sizes_list = [ray_geometry.inc_leg_size(i) for i in
                          range(ray_geometry.numinterfaces)]

    refractive_indices = [None]
    velocities = ray_geometry.rays.fermat_path.velocities
    for inc_velocity, out_velocity in zip(velocities, velocities[1:]):
        refractive_indices.append(inc_velocity / out_velocity)
    refractive_indices.append(None)

    numinterfaces = len(inc_angles_list)
    beamspread = None
    for i in range(1, numinterfaces):
        if i == 1:
            # Leg size between the probe and the first interface
            # res := 1/sqrt(leg_size)
            _tmp = np.sqrt(inc_leg_sizes_list[1])
            beamspread = np.reciprocal(_tmp, _tmp)
        else:
            leg_sizes_before_last_interface = inc_leg_sizes_list[i - 1]
            leg_sizes_after_last_interface = inc_leg_sizes_list[i]

            theta_inc = inc_angles_list[i - 1]

            alpha = refractive_indices[i - 1]

            sin_theta = np.sin(theta_inc)
            cos_theta = np.cos(theta_inc)
            beta = ((alpha * alpha - sin_theta * sin_theta)
                    / (alpha * cos_theta * cos_theta))

            # res = 1/sqrt(1 + leg_after / (beta * leg_before))
            beamspread *= np.reciprocal(np.sqrt(
                1. + leg_sizes_after_last_interface / (
                    leg_sizes_before_last_interface * beta)))

    return beamspread


def beamspread_for_path(ray_geometry):
    warnings.warn(DeprecationWarning('beamspread_for_path is deprecated and will be '
                                     'removed. Use beamspread_2d_for_path instead.'))
    return beamspread_2d_for_path(ray_geometry)


def sensitivity_conjugate_for_path(ray_weights):
    """

    Parameters
    ----------
    ray_weights : ndarray
        Shape: (numelements, numgridpoints)

    Returns
    -------
    sensitivity : ndarray
        Shape : (numgridpoints, )

    """
    ray_weights = ray_weights
    abs_ray_weights = np.abs(ray_weights)
    return np.mean(abs_ray_weights * abs_ray_weights, axis=0)


def sensitivity_conjugate_for_view(tx_sensitivity, rx_sensitivity):
    """

    Parameters
    ----------
    tx_sensitivity : ndarray
        Shape: (numgridpoints, )
    rx_sensitivity
        Shape: (numgridpoints, )

    Returns
    -------
    sensitivity_for_view : ndarray
        Shape: (numgridpoints, )

    """
    return tx_sensitivity * rx_sensitivity
