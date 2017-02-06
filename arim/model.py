"""
Functions related to the forward model.

.. seealso::
    :mod:`arim.ut`

"""
# Functions that rely on arim-specific structures (Path, Frame, Interface, etc.) should
# be put here.
# Functions that rely on simpler logic should be put in arim.ut and imported here.

import logging
from collections import namedtuple

import numpy as np

from . import core as c
from .ut import snell_angles, fluid_solid, solid_l_fluid, solid_t_fluid, \
    directivity_finite_width_2d

logger = logging.getLogger(__name__)


def directivity_finite_width_2d_for_path(ray_geometry, element_width, wavelength):
    """
    Wrapper for :func:`directivity_finite_width_2d` that uses a :class:`RayGeometry` object.

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
    return directivity_finite_width_2d(ray_geometry.out_angles_list[0], element_width,
                                       wavelength)


def transmission_at_interface(interface_kind, material_inc, material_out, mode_inc,
                              mode_out, angles_inc,
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

    Requires to have computed the angles of incidence at each interface (cf. :meth:`Rays.get_incoming_angles`).
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

    inc_angles_list = ray_geometry.inc_angles_list

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


def beamspread_per_interface_for_path(inc_angles_list, out_angles_list,
                                      inc_leg_sizes_list, p=0.5):
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

            yield beamspread_after_interface(inc_angles_last_interface,
                                             out_angles_last_interface,
                                             leg_sizes_before_last_interface,
                                             leg_sizes_after_last_interface, p=p)


def beamspread_for_path(ray_geometry, p=0.5):
    """
    Compute the beamspread for a path.

    Parameters
    ----------
    ray_geometry : RayGeometry
    p : float
        Beamspread is ``1/distance**p`` (distance power p). Use 0.5 for 2D and 1.0 for 3D. Default: 0.5

    Returns
    -------
    beamspread : ndarray

    """
    inc_angles_list = ray_geometry.inc_angles_list
    out_angles_list = ray_geometry.out_angles_list
    inc_leg_sizes_list = ray_geometry.inc_leg_sizes_list

    beamspread = None
    for relative_beamspread in beamspread_per_interface_for_path(inc_angles_list,
                                                                 out_angles_list,
                                                                 inc_leg_sizes_list,
                                                                 p=p):
        if relative_beamspread is None:
            continue
        if beamspread is None:
            beamspread = relative_beamspread
        else:
            beamspread *= relative_beamspread
    return beamspread


def beamspread_after_interface_snell(inc_angles_last_interface,
                                     leg_sizes_before_last_interface,
                                     leg_sizes_after_last_interface, refractive_index,
                                     p=0.5):
    """
    Compute the contribution to beamspread of an interface.


    Parameters
    ----------
    inc_angles_last_interface
        Angles of the incoming rays at the last interface.
    leg_sizes_before_last_interface
        For each ray, size of the leg between the second to last interface and the last interface.
    leg_sizes_after_last_interface : ndarray
        For each ray, size of the leg between the last interface and the current point.
    refractive_indices : list of float
        Refractive index for each interface. None if not relevant.
    p : float
        Beamspread is ``1/distance**p`` (distance power p). Use 0.5 for 2D and 1.0 for 3D. Default: 0.5

    Returns
    -------
    beamspread : ndarray

    """
    beta = (refractive_index ** 2 - np.sin(inc_angles_last_interface) ** 2) \
           / (refractive_index * np.cos(inc_angles_last_interface) ** 2)
    distance_virtual = leg_sizes_before_last_interface * beta
    return (distance_virtual / (distance_virtual + leg_sizes_after_last_interface)) ** p


def beamspread_per_interface_for_path_snell(inc_angles_list, inc_leg_sizes_list,
                                            refractive_indices,
                                            p=0.5):
    """
    Compute the beamspread for a path as a list of the contribution of each interface.

    Parameters
    ----------
    inc_angles_list : list of ndarray
        Each array of the list is the angle of the incoming ray to the interface. One array per interface.
        None if not relevant.
    inc_leg_sizes_list : list of ndarray
        Each array of the list is the size of the incoming leg to the interface. Legs are assumed to be straight.
        One array per interface.
        None if not relevant.
    refractive_indices : list of float
        Refractive index for each interface. None if not relevant.
    p : float
        Beamspread is ``1/distance**p`` (distance power p). Use 0.5 for 2D and 1.0 for 3D. Default: 0.5

    Returns
    -------
    beamspread_list : list of ndarray

    """
    # At the first interface, beamspread is not defined
    yield None

    assert len(inc_angles_list) == len(inc_leg_sizes_list) == len(refractive_indices)
    numinterfaces = len(inc_angles_list)

    for i in range(1, numinterfaces):
        if i == 1:
            # Leg size between the probe and the first interface
            yield inc_leg_sizes_list[1] ** (-p)
        else:
            leg_sizes_before_last_interface = inc_leg_sizes_list[i - 1]
            leg_sizes_after_last_interface = inc_leg_sizes_list[i]

            inc_angles_last_interface = inc_angles_list[i - 1]

            yield beamspread_after_interface_snell(inc_angles_last_interface,
                                                   leg_sizes_before_last_interface,
                                                   leg_sizes_after_last_interface,
                                                   refractive_indices[i - 1], p=p)


def beamspread_for_path_snell(path, ray_geometry, p=0.5):
    """
    Compute the beamspread for a path.

    Compared to the function beamspread_for_path: compute outgoing angles with
    Snell-Descartes law instead of the using the actual rays. This solves numerical issues
    for small angles.

    Parameters
    ----------
    path : Path
    ray_geometry : RayGeometry
    p : float
        Beamspread is ``1/distance**p`` (distance power p). Use 0.5 for 2D and 1.0 for 3D. Default: 0.5

    Returns
    -------
    beamspread : ndarray

    """
    inc_angles_list = ray_geometry.inc_angles_list
    inc_leg_sizes_list = ray_geometry.inc_leg_sizes_list

    refractive_indices = [None]
    for inc_velocity, out_velocity in zip(path.velocities, path.velocities[1:]):
        refractive_indices.append(inc_velocity / out_velocity)
    refractive_indices.append(None)

    beamspread = None
    for relative_beamspread in beamspread_per_interface_for_path_snell(inc_angles_list,
                                                                       inc_leg_sizes_list,
                                                                       refractive_indices,
                                                                       p=p):
        if relative_beamspread is None:
            continue
        if beamspread is None:
            beamspread = relative_beamspread
        else:
            beamspread *= relative_beamspread
    return beamspread


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
    return np.mean(np.abs(ray_weights) ** 2, axis=0)


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


class RayGeometry(namedtuple("RayGeometry", "inc_angles_list out_angles_list "
                                            "inc_leg_sizes_list")):
    """
    RayGeometry(inc_angles_list, out_angles_list, inc_leg_sizes_list)

    Storage object: holds the angles and the sizes of the legs of rays.

    Parameters
    ----------
    inc_angles_list : list of ndarray
        Each array of the list is the angle of the incoming ray to the interface.
        One array per interface. None if not relevant.
    out_angles_list : list of ndarray
        Each array of the list is the angle of the outgoing ray from the interface. One array per interface.
        None if not relevant.
    inc_leg_sizes_list : list of ndarray
        Each array of the list is the size of the incoming leg to the interface. Legs are assumed to be straight.
        One array per interface.
        None if not relevant.

    See Also
    --------
    :meth:`Rays.get_outgoing_angles`, :meth:`Rays.get_incoming_angles`

    """

    @classmethod
    def from_path(cls, path):
        if path.rays is None:
            raise ValueError("Ray-tracing must be performed first.")

        out_angles_list = []
        for alpha in path.rays.get_outgoing_angles(path.interfaces):
            out_angles_list.append(alpha)

        inc_angles_list = []
        inc_leg_sizes_list = []
        for alpha, distances in path.rays.get_incoming_angles(path.interfaces, True):
            inc_angles_list.append(alpha)
            inc_leg_sizes_list.append(distances)

        return cls(inc_angles_list, out_angles_list, inc_leg_sizes_list)
