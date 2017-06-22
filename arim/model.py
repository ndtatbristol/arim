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
from functools import reduce

import numpy as np
import numba

from . import core as c
from . import ut
from .ut import snell_angles, fluid_solid, solid_l_fluid, solid_t_fluid, \
    directivity_2d_rectangular_in_fluid
from .helpers import chunk_array
from .exceptions import ArimWarning

# for backward compatiblity:
from .path import RayGeometry

logger = logging.getLogger(__name__)


def radiation_2d_rectangular_in_fluid_for_path(ray_geometry, element_width, wavelength):
    """
    Wrapper for :func:`radiation_2d_rectangular_in_fluid` that uses
    a :class:`RayGeometry` object.


    Parameters
    ----------
    ray_geometry : RayGeometry
    element_width
    wavelength

    Returns
    -------

    """
    return ut.radiation_2d_rectangular_in_fluid(ray_geometry.conventional_out_angle(0),
                                                element_width, wavelength)


def directivity_2d_rectangular_in_fluid_for_path(ray_geometry, element_width, wavelength):
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


# alias for backward compatibility:
directivity_finite_width_2d_for_path = directivity_2d_rectangular_in_fluid_for_path


def transmission_at_interface(interface_kind, material_inc, material_out, mode_inc,
                              mode_out, angles_inc,
                              force_complex=True, unit='stress'):
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
    if unit.lower() == 'stress':
        convert_to_displacement = False
    elif unit.lower() == 'displacement':
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
        if convert_to_displacement:
            # u2/u1 = z tau2 / tau1 = -z tau2 / p1
            z = ((material_inc.density * material_inc.velocity(mode_inc)) /
                 (material_out.density * material_out.velocity(mode_out)))
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
            c_t=solid.transverse_vel)

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
            z = ((material_inc.density * material_inc.velocity(mode_inc)) /
                 (material_out.density * material_out.velocity(mode_out)))
            transmission *= z
        return transmission
    else:
        raise NotImplementedError


def reflection_at_interface(interface_kind, material_inc, material_against, mode_inc,
                            mode_out, angles_inc,
                            force_complex=True, unit='stress'):
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
    if unit.lower() == 'stress':
        convert_to_displacement = False
    elif unit.lower() == 'displacement':
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
            c_t=solid.transverse_vel)
        with np.errstate(invalid='ignore'):
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
    else:
        raise NotImplementedError


def transmission_reflection_for_path(path, ray_geometry, force_complex=True,
                                     unit='stress'):
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
    ray_geometry : RayGeometry
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

        logger.debug("compute {} coefficients at interface {}".format(
            interface.transmission_reflection.name,
            interface.points))

        if interface.transmission_reflection is c.TransmissionReflection.transmission:
            params['material_out'] = path.materials[i]
            tmp = transmission_at_interface(**params)
        elif interface.transmission_reflection is c.TransmissionReflection.reflection:
            params['material_against'] = interface.reflection_against
            tmp = reflection_at_interface(**params)
        else:
            raise RuntimeError

        if transrefl is None:
            transrefl = tmp
        else:
            transrefl *= tmp
        del tmp

    return transrefl


def reverse_transmission_reflection_for_path(path, ray_geometry, force_complex=True,
                                             unit='stress'):
    """
    Return the transmission-reflection coefficients of the reverse path.

    This function uses the same angles as :func:`transmission_reflection_for_path`.
    These angles are the incident angles in the direct path (but the reflected/transmitted
    angles in the reverse path).

    Parameters
    ----------
    path : Path
    ray_geometry : RayGeometry
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
            params['material_out'] = material_out
            params['interface_kind'] = interface.kind.reverse()

            # Compute the incident angles in the reverse path from the incident angles in the
            # direct path using Snell laws.
            if force_complex:
                trans_or_refl_angles = np.asarray(trans_or_refl_angles, complex)
            params['angles_inc'] = snell_angles(trans_or_refl_angles,
                                                material_out.velocity(mode_out),
                                                material_inc.velocity(mode_inc))
            tmp = transmission_at_interface(**params)

        elif interface.transmission_reflection is c.TransmissionReflection.reflection:
            params['material_against'] = interface.reflection_against
            params['interface_kind'] = interface.kind
            params['angles_inc'] = snell_angles(trans_or_refl_angles,
                                                material_inc.velocity(mode_out),
                                                material_inc.velocity(mode_inc))
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
    ray_geometry : arim.path.RayGeometry

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
        gamma_list.append((nu * nu - sin_theta * sin_theta)
                          / (nu * cos_theta * cos_theta))

    # Between the probe and the first interface, beamspread of an unbounded medium.
    # Use a copy because the original may be a cached value and we don't want
    # to change it by accident.
    virtual_distance = ray_geometry.inc_leg_size(1).copy()  # distance A_0 A_1

    for k in range(1, n):
        # distance A_k A_{k+1}:
        r = ray_geometry.inc_leg_size(k + 1)
        gamma = 1.
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
    ray_geometry : RayGeometry

    Returns
    -------
    rev_beamspread : ndarray
        Shape: (ray_geometry.interfaces[0].numpoints,
        ray_geometry.interfaces[-1].numpoints)

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
        gamma_list.append((nu * cos_theta * cos_theta)
                          / (1 - nu * nu * sin_theta * sin_theta))

    # Between the probe and the first interface, beamspread of an unbounded medium.
    # Use a copy because the original may be a cached value and we don't want
    # to change it by accident.
    virtual_distance = ray_geometry.inc_leg_size(n).copy()

    for k in range(1, n):
        # distance A_k A_{k+1}:
        r = ray_geometry.inc_leg_size(n - k)
        gamma = 1.
        for i in range(k):
            gamma *= gamma_list[i]
        virtual_distance += r / gamma

    return np.reciprocal(np.sqrt(virtual_distance))


def beamspread_for_path(ray_geometry):
    """
    Deprecation warning: use :func:`beamspread_2d_for_path` instead.
    """
    warnings.warn('beamspread_for_path is deprecated and will be '
                  'removed. Use beamspread_2d_for_path instead.',
                  DeprecationWarning, stacklevel=2)
    return beamspread_2d_for_path(ray_geometry)


def sensitivity_conjugate_for_path(ray_weights):
    """
    Critical bug here: works only for FMC

    Parameters
    ----------
    ray_weights : ndarray
        Shape: (numelements, numgridpoints)

    Returns
    -------
    sensitivity : ndarray
        Shape : (numgridpoints, )

    """
    warnings.warn('This function does not work propertly, to be fixed',
                  ArimWarning, stacklevel=2)
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
    warnings.warn('This function does not work propertly, to be fixed',
                  ArimWarning, stacklevel=2)
    return tx_sensitivity * rx_sensitivity


def sensitivity_image_point_source(tx_ray_weights, rx_ray_weights, tx, rx,
                                   scanline_weights=None):
    numelements, numpoints = tx_ray_weights.shape
    numscanlines = tx.shape[0]

    tx_amplitudes = np.ascontiguousarray(tx_ray_weights.T)
    rx_amplitudes = np.ascontiguousarray(rx_ray_weights.T)

    sensitivity = np.zeros(numpoints)
    block_size = 1000

    if scanline_weights is None:
        scanline_weights = ut.default_scanline_weights(tx, rx)

    for chunk in chunk_array(sensitivity.shape, block_size, axis=0):
        # Model amplitudes P_ij
        model_amplitudes = (np.take(tx_amplitudes[chunk], tx, axis=1)
                            * np.take(rx_amplitudes[chunk], rx, axis=1))

        # Compute sensitivity image (write result on sensitivity_result)
        sensitivity_image(model_amplitudes, scanline_weights, sensitivity[chunk])
    return sensitivity


# @numba.jit(nopython=True)
@numba.guvectorize([(numba.float32[:, :], numba.float32[:], numba.float32[:]),
                    (numba.float64[:, :], numba.float64[:], numba.float64[:]),
                    (numba.complex64[:, :], numba.float32[:], numba.float32[:]),
                    (numba.complex128[:, :], numba.float64[:], numba.float64[:]),
                    ], '(n, m),(m)->(n)', target='cpu')
def sensitivity_image(model_amplitudes, scanline_weights, result):
    """
    Compute sensitivity I_0. FMC or HMC agnostic.

    Parameters
    ----------
    model_amplitudes : ndarray
        (numpoints, numscanlines)
    scanline_weights : ndarray
        (numscanlines, )
    result
        (numpoints, )

    Returns
    -------
    None, write in result.

    """
    numpoints, numscanlines = model_amplitudes.shape
    for pidx in range(numpoints):
        result[pidx] = 0.
        for scan in range(numscanlines):
            x = abs(model_amplitudes[pidx, scan])
            result[pidx] += scanline_weights[scan] * x * x

