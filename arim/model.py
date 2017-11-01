"""
Core functions of the forward model.

.. seealso::
    :mod:`arim.ut`

"""
# Functions that rely on arim-specific structures (Path, Frame, Interface, etc.) should
# be put here.
# Functions that rely on simpler logic should be put in arim.ut and imported here.

import abc
import logging
from collections import namedtuple

import numpy as np
import numba

from . import core as c
from . import ut, scat
from .ut import snell_angles, fluid_solid, solid_l_fluid, solid_t_fluid, \
    directivity_2d_rectangular_in_fluid
from .helpers import chunk_array


logger = logging.getLogger(__name__)


def radiation_2d_rectangular_in_fluid_for_path(ray_geometry, element_width, wavelength):
    """
    Wrapper for :func:`radiation_2d_rectangular_in_fluid` that uses
    a :class:`RayGeometry` object.


    Parameters
    ----------
    ray_geometry : arim.ray.RayGeometry
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
    ray_geometry : arim.ray.RayGeometry
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
            c_t=solid.transverse_vel)
        with np.errstate(invalid='ignore'):
            reflection, transmission_l, transmission_t = fluid_solid(**params)

        z = material_inc.velocity(mode_inc) / material_inc.velocity(mode_out)
        if convert_to_displacement:
            return reflection * z
        else:
            return reflection
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
        gamma_list.append((nu * nu - sin_theta * sin_theta)
                          / (nu * cos_theta * cos_theta))

    # Between the probe and the first interface, beamspread of an unbounded medium.
    # Use a copy because the original may be a cached value and we don'ray want
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
    ray_geometry : arim.ray.RayGeometry

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
    # Use a copy because the original may be a cached value and we don'ray want
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


class RayWeights(namedtuple('RayWeights', [
    'tx_ray_weights_dict', 'rx_ray_weights_dict',
    'tx_ray_weights_debug_dict', 'rx_ray_weights_debug_dict', 'scattering_angles_dict'])):
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


class ModelAmplitudes(abc.ABC):
    """
    Pseudo-array of coefficients P_ij = Q_i Q'_j S_ij. Shape: (numpoints, numscanlines)

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

    Get the P_ij for the first grid point (returns an array of size (numscanlines,)):
    >>> model_amplitudes[0]

    Get the P_ij for the first ten grid points (returns an array of size
    (10, numscanlines,)):
    >>> model_amplitudes[:10]

    Get all P_ij (may run out of memory):
    >>> model_amplitudes[...]

    To get the first Get all P_ij (may run out of memory):
    >>> model_amplitudes[...]

    Indexing the second dimension will fail. For example to model amplitude of
    the fourth point and the eigth scanline, use:
    >>> model_amplitudes[3][7]  # valid usage
    >>> model_amplitudes[3, 7]  # invalid usage, raise an IndexError
    """

    @abc.abstractmethod
    def __getitem__(self, grid_slice):
        ...

    @property
    def shape(self):
        return (self.numpoints, self.numscanlines)

    def sensitivity_uniform_tfm(self, scanline_weights, **kwargs):
        # wrapper in general case, inherit and write a faster implementation if possible
        return sensitivity_uniform_tfm(self, scanline_weights, **kwargs)

    def sensitivity_model_assisted_tfm(self, scanline_weights, **kwargs):
        # wrapper in general case, inherit and write a faster implementation if possible
        return sensitivity_model_assisted_tfm(self, scanline_weights, **kwargs)


class _ModelAmplitudesWithScatFunction(ModelAmplitudes):
    def __init__(self, tx, rx, scattering_fn,
                 tx_ray_weights, rx_ray_weights,
                 tx_scattering_angles, rx_scattering_angles):
        self.tx = tx
        self.rx = rx
        self.scattering_fn = scattering_fn
        self.tx_ray_weights = tx_ray_weights
        self.rx_ray_weights = rx_ray_weights
        self.tx_scattering_angles = tx_scattering_angles
        self.rx_scattering_angles = rx_scattering_angles
        self.numpoints, self.numelements = tx_ray_weights.shape
        self.numscanlines = self.tx.shape[0]

    def __getitem__(self, grid_slice):
        # Nota bene: arrays' shape is (numpoints, numscanline), i.e. the transpose
        # of RayWeights. They are contiguous.
        if np.empty(self.numpoints)[grid_slice].ndim > 1:
            raise IndexError('Only the first dimension of the object is indexable.')

        scattering_amplitudes = self.scattering_fn(
            np.take(self.tx_scattering_angles[grid_slice], self.tx, axis=-1),
            np.take(self.rx_scattering_angles[grid_slice], self.rx, axis=-1))

        model_amplitudes = (scattering_amplitudes
                            * np.take(self.tx_ray_weights[grid_slice], self.tx, axis=-1)
                            * np.take(self.rx_ray_weights[grid_slice], self.rx, axis=-1))
        return model_amplitudes


@numba.guvectorize(
    'void(int32[:], int32[:], complex128[:,:], complex128[:], complex128[:], float64[:], float64[:], complex128[:])',
    '(n),(n),(s,s),(e),(e),(e),(e)->(n)', nopython=True, target='parallel')
def _model_amplitudes_with_scat_matrix(tx, rx, scattering_matrix, tx_ray_weights,
                                       rx_ray_weights, tx_scattering_angles,
                                       rx_scattering_angles, res):
    # This is a kernel on a grid point.
    numscanlines = tx.shape[0]
    # assert res.shape[0] == tx_ray_weights.shape[0]
    # assert tx_ray_weights.shape == rx_ray_weights.shape == tx_scattering_angles.shape == rx_scattering_angles.shape
    for scan in range(numscanlines):
        inc_theta = tx_scattering_angles[tx[scan]]
        out_theta = rx_scattering_angles[rx[scan]]

        scattering_amp = scat._interpolate_scattering_matrix_kernel(scattering_matrix,
                                                                    inc_theta, out_theta)
        res[scan] = (scattering_amp
                     * tx_ray_weights[tx[scan]]
                     * rx_ray_weights[rx[scan]])


class _ModelAmplitudesWithScatMatrix(ModelAmplitudes):
    def __init__(self, tx, rx, scattering_mat,
                 tx_ray_weights, rx_ray_weights,
                 tx_scattering_angles, rx_scattering_angles):
        self.tx = tx
        self.rx = rx
        self.scattering_mat = scattering_mat
        self.tx_ray_weights = tx_ray_weights
        self.rx_ray_weights = rx_ray_weights
        self.tx_scattering_angles = tx_scattering_angles
        self.rx_scattering_angles = rx_scattering_angles
        self.numpoints, self.numelements = tx_ray_weights.shape
        self.numscanlines = self.tx.shape[0]

    def __getitem__(self, grid_slice):
        # Nota bene: arrays' shape is (numpoints, numscanline), i.e. the transpose
        # of RayWeights. They are contiguous.
        res_dtype = np.result_type(self.scattering_mat, self.tx_ray_weights,
                                   self.rx_ray_weights)
        return _model_amplitudes_with_scat_matrix(
            self.tx, self.rx, self.scattering_mat,
            self.tx_ray_weights[grid_slice],
            self.rx_ray_weights[grid_slice],
            self.tx_scattering_angles[grid_slice],
            self.rx_scattering_angles[grid_slice])

        # grid_chunk = np.empty(self.numpoints)[grid_slice]
        # if grid_chunk.ndim == 0:
        #     # grid_slice is an int. We are working on one point, add a dimension
        #     # to please _model_amplitudes_with_scat_matrix
        #     res = np.zeros((1, self.numscanlines), res_dtype)
        # elif grid_chunk.ndim == 1:
        #     # grid_slice is a slice
        #     res = np.zeros((grid_chunk.shape[0], self.numscanlines), res_dtype)
        # else:
        #     raise IndexError('Only the first dimension of the object is indexable.')
        #
        # res = _model_amplitudes_with_scat_matrix(
        #     self.tx, self.rx, self.scattering_mat,
        #     np.atleast_2d(self.tx_ray_weights[grid_slice]),
        #     np.atleast_2d(self.rx_ray_weights[grid_slice]),
        #     np.atleast_2d(self.tx_scattering_angles[grid_slice]),
        #     np.atleast_2d(self.rx_scattering_angles[grid_slice]),
        #     res)
        # if grid_chunk.ndim == 0:
        #     # remove the extra dimension added to please _model_amplitudes_with_scat_matrix
        #     return res[0]
        # else:
        #     return res


def model_amplitudes_factory(tx, rx, view, ray_weights, scattering):
    """
    Yield P_ij = Q_i Q'_j S_ij

    Works block per block to avoid running out of memory.

    Parameters
    ----------
    tx : ndarray
    rx : ndarray
    view : View
    ray_weights : RayWeights
    scattering : dict
        Dict of functions (slow) or matrices (fast).

    Returns
    ------
    model_amplitudes : ModelAmplitudes
        Object that is indexable with a grid point index or a slice of grid points. The
         values are computed on the fly.

        can be indexe as an array but that computes the
        Function that returns the model amplitudes and takes as argument a slice.
        ndarray
        Shape: (blocksize, numscanlines)
        Yield until all grid points are processed.

    Examples
    --------
    >>> model_amplitudes = model_amplitudes_factory(tx, rx, view, ray_weights, scattering)
    >>> model_amplitudes[0]
    # returns the 'numscanlines' amplitudes at the grid point 0
    >>> model_amplitudes[:10] # returns the amplitudes for the first 10 grid points
    array([ 0.27764253,  0.78863332,  0.83998295,  0.96811351,  0.57929045, 0.00935137,  0.8905348 ,  0.46976061,  0.08101099,  0.57615469])
    >>> model_amplitudes[...] # returns the amplitudes for all points. Warning: you may
    ... # run out of memory!
    array([...])

    """
    # Pick the right scattering matrix/function.
    # scat_key is LL, LT, TL or TT
    scat_key = view.tx_path.modes[-1].key() + view.rx_path.modes[-1].key()
    scattering_obj = scattering[scat_key]

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
    assert tx_ray_weights.flags.f_contiguous
    assert rx_ray_weights.flags.f_contiguous
    assert tx_scattering_angles.flags.f_contiguous
    assert rx_scattering_angles.flags.f_contiguous

    assert (tx_ray_weights.shape == rx_ray_weights.shape ==
            tx_scattering_angles.shape == rx_scattering_angles.shape)

    # the great transposition
    tx_ray_weights = tx_ray_weights.T
    rx_ray_weights = rx_ray_weights.T
    tx_scattering_angles = tx_scattering_angles.T
    rx_scattering_angles = rx_scattering_angles.T

    if is_scattering_func:
        return _ModelAmplitudesWithScatFunction(tx, rx, scattering_obj,
                                                tx_ray_weights, rx_ray_weights,
                                                tx_scattering_angles,
                                                rx_scattering_angles)
    else:
        return _ModelAmplitudesWithScatMatrix(tx, rx, scattering_obj,
                                              tx_ray_weights, rx_ray_weights,
                                              tx_scattering_angles,
                                              rx_scattering_angles)


def sensitivity_uniform_tfm(model_amplitudes, scanline_weights, block_size=4000):
    """
    Return the sensitivity for uniform TFM.

    The sensitivity at a point is defined the predicted TFM amplitude that a sole
    scatterer centered on that point would have.

    Parameters
    ----------
    model_amplitudes : ndarray or ModelAmplitudes
        Coefficients P_ij. Shape: (numpoints, numscanlines)
    scanline_weights : ndarray
        Shape: (numscanlines, )

    Returns
    -------
    predicted_intensities
        Shape: (numpoints, )
    """
    numpoints, numscanlines = model_amplitudes.shape
    assert scanline_weights.ndim == 1
    assert model_amplitudes.shape[1] == scanline_weights.shape[0]

    sensitivity = None

    # chunk the array in case we have an array too big (ModelAmplitudes)
    for chunk in chunk_array((numpoints, numscanlines), block_size):
        tmp = (scanline_weights[np.newaxis] * model_amplitudes[chunk]).sum(axis=1)
        if sensitivity is None:
            sensitivity = np.zeros((numpoints,), dtype=tmp.dtype)
        sensitivity[chunk] = tmp
    return sensitivity


def sensitivity_model_assisted_tfm(model_amplitudes, scanline_weights, block_size=4000):
    """
    Return the sensitivity for model assisted TFM (multiply TFM scanlines by conjugate
    of scatterer contribution).

    The sensitivity at a point is defined the predicted TFM amplitude that a sole
    scatterer centered on that point would have.

    Parameters
    ----------
    model_amplitudes : ndarray or ModelAmplitudes
        Coefficients P_ij. Shape: (numpoints, numscanlines)
    scanline_weights : ndarray
        Shape: (numscanlines, )

    Returns
    -------
    predicted_intensities
        Shape: (numpoints, ).
    """
    numpoints, numscanlines = model_amplitudes.shape
    assert scanline_weights.ndim == 1
    assert model_amplitudes.shape[1] == scanline_weights.shape[0]

    sensitivity = None

    # chunk the array in case we have an array too big (ModelAmplitudes)
    for chunk in chunk_array((numpoints, numscanlines), block_size):
        absval = np.abs(model_amplitudes[chunk])
        tmp = (absval * absval * scanline_weights[np.newaxis]).sum(axis=1)
        if sensitivity is None:
            sensitivity = np.zeros((numpoints,), dtype=tmp.dtype)
        sensitivity[chunk] = tmp
    return sensitivity
