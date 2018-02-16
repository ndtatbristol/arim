"""
Forward model of the inspection of a solid block in immersion

Boilerplate::

    import arim.models.block_in_immersion as bim

    probe_p = probe.to_oriented_points()
    frontwall = \
        arim.geometry.points_1d_wall_z(xmin, xmax, z_frontwall, numpoints)
    backwall = \
        arim.geometry.points_1d_wall_z(xmin, xmax, z_backwall, numpoints)

    grid = arim.geometry.Grid(xmin, xmax, ymin, ymax, zmin, zmax, pixel_size)
    grid_p = grid.to_oriented_points()

    exam_obj = arim.BlockInImmersion(block_material, couplant_material,
                                     frontwall, backwall)

    views = bim.make_views(examination_object, probe_p,
                           grid_p, max_number_of_reflection=1,
                           tfm_unique_only=False)

Scattering precomputation
=========================

In the computation of the model, there are two options for scattering. The
first option is to pass functions, which are called for each pair of incident
and scatterer angles. The evaluation time per angle pair is in general almost
constant.

The second option is to pass scattering matrices, which are the function
outputs for a grid of incident and scatterer angle. Missing angles are obtained
by linear interpolation. The second option suffers from a loss of accuracy if
the number of angles used for evaluation is too small. The total evaluation
time is the sum of the precomputation time and the interpolation.

For a small number of angles to evaluate, passing the functions (option 1) is
often the most computationally efficient. For a large amount of angles to
evaluate, precomputing the scattering matrices (option 2) is often more
computationally efficient.

"""
import logging
from collections import namedtuple, OrderedDict
import cmath

import numpy as np
import numba

from .. import model, ray, ut, helpers
from .. import core as c, geometry as g
from ..ray import RayGeometry

logger = logging.getLogger(__name__)

_RayWeightsCommon = namedtuple('_RayWeightsCommon',
                               ['couplant', 'numgridpoints', 'wavelength_in_couplant',
                                'wavelengths_in_block'])


def _init_ray_weights(path, frequency, probe_element_width, use_directivity):
    if path.rays is None:
        raise ValueError('Ray tracing must have been performed first.')

    couplant, block = path.materials[:2]
    numgridpoints = len(path.interfaces[-1].points)

    if use_directivity and probe_element_width is None:
        raise ValueError(
            'probe_element_width must be provided to compute directivity')

    wavelength_in_couplant = couplant.longitudinal_vel / frequency
    if block.transverse_vel is None:
        wavelengths_in_block = dict([(c.Mode.L, block.longitudinal_vel / frequency),
                                     (c.Mode.T, float('nan'))])
    else:
        wavelengths_in_block = dict([(c.Mode.L, block.longitudinal_vel / frequency),
                                     (c.Mode.T, block.transverse_vel / frequency)])

    return _RayWeightsCommon(couplant, numgridpoints, wavelength_in_couplant,
                             wavelengths_in_block)


def tx_ray_weights(path, ray_geometry, frequency, probe_element_width=None,
                   use_directivity=True, use_beamspread=True, use_transrefl=True):
    """
    Coefficients Q_i(r, omega) in forward model.

    Parameters
    ----------
    path : Path
    ray_geometry : arim.ray.RayGeometry
    frequency : float
    probe_element_width : float or None
        Mandatory if use_directivity is True
    use_directivity : bool
        Default True
    use_beamspread : bool
        Default True
    use_transrefl : bool
        Default: True

    Returns
    -------
    weights : ndarray
        Shape (numelements, numgridpoints)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection
    """
    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints), order='F')

    if use_directivity:
        weights_dict['directivity'] = model.radiation_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant)
    else:
        weights_dict['directivity'] = one
    if use_transrefl:
        weights_dict['transrefl'] = model.transmission_reflection_for_path(
            path, ray_geometry, unit='displacement')
    else:
        weights_dict['transrefl'] = one
    if use_beamspread:
        weights_dict['beamspread'] = (model.beamspread_2d_for_path(ray_geometry) *
                                      np.sqrt(d.wavelength_in_couplant))
    else:
        weights_dict['beamspread'] = one

    weights = (weights_dict['directivity'] *
               weights_dict['transrefl'] *
               weights_dict['beamspread'])
    return weights, weights_dict


def rx_ray_weights(path, ray_geometry, frequency, probe_element_width=None,
                   use_directivity=True, use_beamspread=True, use_transrefl=True):
    """
    Coefficients Q'_i(r, omega) in forward model.

    Parameters
    ----------
    path : Path
    ray_geometry : arim.ray.RayGeometry
    frequency : float
    probe_element_width : float or None
        Mandatory if use_directivity is True
    use_directivity : bool
        Default True
    use_beamspread : bool
        Default True
    use_transrefl : bool
        Default: True

    Returns
    -------
    weights : ndarray
        Shape (numelements, numgridpoints)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection
    """
    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints))

    if use_directivity:
        weights_dict['directivity'] = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant)
    else:
        weights_dict['directivity'] = one
    if use_transrefl:
        weights_dict['transrefl'] = model.reverse_transmission_reflection_for_path(
            path, ray_geometry, unit='displacement')
    else:
        weights_dict['transrefl'] = one
    if use_beamspread:
        weights_dict['beamspread'] = (
                model.reverse_beamspread_2d_for_path(ray_geometry)
                * np.sqrt(d.wavelengths_in_block[path.modes[-1]]))
    else:
        weights_dict['beamspread'] = one

    weights = (weights_dict['directivity'] *
               weights_dict['transrefl'] *
               weights_dict['beamspread'])
    return weights, weights_dict


def ray_weights_for_views(views, frequency, probe_element_width=None,
                          use_directivity=True, use_beamspread=True,
                          use_transrefl=True, save_debug=False):
    """
    Compute coefficients Q_i(r, omega) and Q'_j(r, omega) from the forward model for
    all views.
    NB: do not compute the scattering.

    Internally use :func:`tx_ray_weights` and :func:`rx_way_weights`.

    Parameters
    ----------
    views
    frequency
    probe_element_width
    use_directivity
    use_beamspread
    use_transrefl
    save_debug

    Returns
    -------
    RayWeights
    """
    tx_ray_weights_dict = {}
    rx_ray_weights_dict = {}
    if save_debug:
        tx_ray_weights_debug_dict = {}
        rx_ray_weights_debug_dict = {}
    else:
        tx_ray_weights_debug_dict = None
        rx_ray_weights_debug_dict = None
    scat_angle_dict = {}

    all_tx_paths = {view.tx_path for view in views.values()}
    all_rx_paths = {view.rx_path for view in views.values()}
    all_paths = all_tx_paths | all_rx_paths

    model_options = dict(frequency=frequency,
                         probe_element_width=probe_element_width,
                         use_beamspread=use_beamspread,
                         use_directivity=use_directivity,
                         use_transrefl=use_transrefl)

    # By proceeding this way, geometrical computations can be reused for both
    # tx and rx path.
    for path in all_paths:
        ray_geometry = RayGeometry.from_path(path)
        scat_angle_dict[path] = np.asfortranarray(ray_geometry.signed_inc_angle(-1))
        scat_angle_dict[path].flags.writeable = False

        if path in all_tx_paths:
            ray_weights, ray_weights_debug = tx_ray_weights(path, ray_geometry,
                                                            **model_options)
            ray_weights = np.asfortranarray(ray_weights)
            ray_weights.flags.writeable = False
            tx_ray_weights_dict[path] = ray_weights
            if save_debug:
                tx_ray_weights_debug_dict[path] = ray_weights_debug
            del ray_weights, ray_weights_debug
        if path in all_rx_paths:
            ray_weights, ray_weights_debug = rx_ray_weights(path, ray_geometry,
                                                            **model_options)
            ray_weights = np.asfortranarray(ray_weights)
            ray_weights.flags.writeable = False
            rx_ray_weights_dict[path] = ray_weights
            if save_debug:
                rx_ray_weights_debug_dict[path] = ray_weights_debug
            del ray_weights, ray_weights_debug

    return model.RayWeights(tx_ray_weights_dict, rx_ray_weights_dict,
                            tx_ray_weights_debug_dict, rx_ray_weights_debug_dict,
                            scat_angle_dict)


def frontwall_path(couplant_material, block_material, probe_points,
                   probe_orientations, frontwall_points, frontwall_orientations):
    """
    Probe -> couplant -> frontwall -> couplant -> probe

    Parameters
    ----------
    couplant_material
    block_material
    probe_points
    probe_orientations
    frontwall_points
    frontwall_orientations

    Returns
    -------
    Path

    """
    probe_start = c.Interface(probe_points, probe_orientations,
                              are_normals_on_out_rays_side=True)
    probe_end = c.Interface(probe_points, probe_orientations,
                            are_normals_on_inc_rays_side=True)
    frontwall_ext_refl = c.Interface(frontwall_points, frontwall_orientations,
                                     'fluid_solid', 'reflection',
                                     reflection_against=block_material,
                                     are_normals_on_inc_rays_side=False,
                                     are_normals_on_out_rays_side=False)

    return c.Path(
        interfaces=(probe_start, frontwall_ext_refl, probe_end),
        materials=(couplant_material, couplant_material),
        modes=(c.Mode.L, c.Mode.L),
        name='Frontwall')


def backwall_paths(couplant_material, block_material, probe_oriented_points,
                   frontwall, backwall):
    """
    Make backwall paths

    Probe -> couplant -> frontwall -> block (L or T) -> backwall -> block (L or T) -> frontwall -> couplant -> probe

    Parameters
    ----------
    couplant_material : Material
    block_material : Material
    probe_oriented_points : OrientedPoints
    frontwall: OrientedPoints
    backwall: OrientedPoints

    Returns
    -------
    OrderedDict of Path
        Keys: LL, LT, TL, TT

    """
    probe_start = c.Interface(*probe_oriented_points,
                              are_normals_on_out_rays_side=True)

    frontwall_couplant_to_block = c.Interface(
        *frontwall,
        'fluid_solid', 'transmission',
        are_normals_on_inc_rays_side=False,
        are_normals_on_out_rays_side=True)

    backwall_refl = c.Interface(
        *backwall,
        'solid_fluid', 'reflection',
        reflection_against=couplant_material,
        are_normals_on_inc_rays_side=False,
        are_normals_on_out_rays_side=False)

    frontwall_block_to_couplant = c.Interface(
        *frontwall,
        'solid_fluid', 'transmission',
        are_normals_on_inc_rays_side=True,
        are_normals_on_out_rays_side=False)

    probe_end = c.Interface(*probe_oriented_points,
                            are_normals_on_inc_rays_side=True)

    paths = OrderedDict()

    for mode1 in (c.Mode.L, c.Mode.T):
        for mode2 in (c.Mode.L, c.Mode.T):
            key = mode1.key() + mode2.key()
            paths[key] = c.Path(
                interfaces=(probe_start, frontwall_couplant_to_block, backwall_refl,
                            frontwall_block_to_couplant, probe_end),
                materials=(couplant_material, block_material, block_material, couplant_material),
                modes=(c.Mode.L, mode1, mode2, c.Mode.L),
                name='Backwall ' + key)

    return paths


def ray_weights_for_wall(path, frequency, probe_element_width=None,
                         use_directivity=True, use_beamspread=True,
                         use_transrefl=True):
    """
    Parameters
    ----------
    path
    frequency
    probe_element_width
    use_directivity
    use_beamspread
    use_transrefl

    Returns
    -------
    weights : ndarray
        Shape (numelements, numelements)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection

    """
    # perform ray tracing if needed
    if path.rays is None:
        ray.ray_tracing_for_paths([path])

    ray_geometry = RayGeometry.from_path(path)

    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints), order='F')

    if use_directivity:
        directivity_tx = model.radiation_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant)
        directivity_rx = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant)

        weights_dict['directivity'] = directivity_tx * directivity_rx.T
    else:
        weights_dict['directivity'] = one
    if use_transrefl:
        weights_dict['transrefl'] = model.transmission_reflection_for_path(
            path, ray_geometry, unit='displacement')
    else:
        weights_dict['transrefl'] = one
    if use_beamspread:
        weights_dict['beamspread'] = (model.beamspread_2d_for_path(ray_geometry) *
                                      np.sqrt(d.wavelength_in_couplant))
    else:
        weights_dict['beamspread'] = one

    weights = (weights_dict['directivity'] *
               weights_dict['transrefl'] *
               weights_dict['beamspread'])
    return weights, weights_dict


def make_interfaces(couplant_material,
                    probe_oriented_points,
                    frontwall,
                    backwall,
                    grid_oriented_points):
    """
    Construct Interface objects for the case of a solid block in immersion
    (couplant is liquid).

    The interfaces are for rays starting from the probe and arriving in the
    grid. There is at the frontwall interface a liquid-to-solid transmission.
    There is at the backwall interface a solid-against-liquid reflection.

    Assumes all normals are pointing roughly towards the same direction (example: (0, 0, 1) or so).

    Parameters
    ----------
    couplant_material: Material
    couplant_material: Material
    probe_oriented_points : OrientedPoints
    frontwall: OrientedPoints
    backwall: OrientedPoints
    grid_oriented_points: OrientedPoints

    Returns
    -------
    interface_dict : dict[Interface]
        Keys: probe, frontwall_trans, backwall_refl, grid, frontwall_refl
    """
    interface_dict = OrderedDict()

    interface_dict['probe'] = c.Interface(*probe_oriented_points,
                                          are_normals_on_out_rays_side=True)
    interface_dict['frontwall_trans'] = c.Interface(
        *frontwall,
        'fluid_solid', 'transmission',
        are_normals_on_inc_rays_side=False,
        are_normals_on_out_rays_side=True)
    if backwall is not None:
        interface_dict['backwall_refl'] = c.Interface(
            *backwall,
            'solid_fluid', 'reflection',
            reflection_against=couplant_material,
            are_normals_on_inc_rays_side=False,
            are_normals_on_out_rays_side=False)
    interface_dict['grid'] = c.Interface(*grid_oriented_points,
                                         are_normals_on_inc_rays_side=True)
    interface_dict['frontwall_refl'] = c.Interface(*frontwall,
                                                   'solid_fluid', 'reflection',
                                                   reflection_against=couplant_material,
                                                   are_normals_on_inc_rays_side=True,
                                                   are_normals_on_out_rays_side=True)

    return interface_dict


def make_paths(block_material, couplant_material, interface_dict,
               max_number_of_reflection=1):
    """
    Creates the paths L, T, LL, LT, TL, TT (in this order).

    Paths are returned in transmit convention: for the path XY, X is the mode
    before reflection against the backwall and Y is the mode after reflection.
    The path XY in transmit convention is the path YX in receive convention.

    Parameters
    ----------
    block_material : Material
    couplant_material : Material
    interface_dict : dict[Interface]
    max_number_of_reflection : int
        Default: 1.


    Returns
    -------
    paths : OrderedDict

    """
    paths = OrderedDict()

    if max_number_of_reflection > 2:
        raise NotImplementedError
    if max_number_of_reflection < 0:
        raise ValueError

    probe = interface_dict['probe']
    frontwall = interface_dict['frontwall_trans']
    grid = interface_dict['grid']
    if max_number_of_reflection >= 1:
        backwall = interface_dict['backwall_refl']
    if max_number_of_reflection >= 2:
        frontwall_refl = interface_dict['frontwall_refl']

    paths['L'] = c.Path(
        interfaces=(probe, frontwall, grid),
        materials=(couplant_material, block_material),
        modes=(c.Mode.L, c.Mode.L),
        name='L')

    paths['T'] = c.Path(
        interfaces=(probe, frontwall, grid),
        materials=(couplant_material, block_material),
        modes=(c.Mode.L, c.Mode.T),
        name='T')

    if max_number_of_reflection >= 1:
        paths['LL'] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.L, c.Mode.L),
            name='LL')

        paths['LT'] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.L, c.Mode.T),
            name='LT')

        paths['TL'] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.T, c.Mode.L),
            name='TL')

        paths['TT'] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.T, c.Mode.T),
            name='TT')

    if max_number_of_reflection >= 2:
        keys = ['LLL', 'LLT', 'LTL', 'LTT', 'TLL', 'TLT', 'TTL', 'TTT']

        for key in keys:
            paths[key] = c.Path(
                interfaces=(probe, frontwall, backwall, frontwall_refl, grid),
                materials=(couplant_material, block_material, block_material,
                           block_material),
                modes=(c.Mode.L,
                       helpers.parse_enum_constant(key[0], c.Mode),
                       helpers.parse_enum_constant(key[1], c.Mode),
                       helpers.parse_enum_constant(key[2], c.Mode)),
                name=key)

    return paths


def make_views_from_paths(paths_dict, tfm_unique_only=False):
    """
    Returns 'View' objects for the case of a block in immersion.

    Consut all possible views that can be constructed with the paths given as argument.

    If unique only ``unique_only`` is false,

    Parameters
    ----------
    paths_dict : Dict[Path]
        Key: path names (exemple: 'L', 'LT'). Values: :class:`Path`
    tfm_unique_only : bool
        Default: False. If True, returns only the views that give *different* imaging
        results with TFM (AB-CD and DC-BA give the same imaging result).

    Returns
    -------
    views: OrderedDict[Views]

    """
    viewnames = ut.make_viewnames(paths_dict.keys(), tfm_unique_only=tfm_unique_only)
    views = OrderedDict()
    for view_name_tuple in viewnames:
        tx_name, rx_name = view_name_tuple
        view_name = '{}-{}'.format(tx_name, rx_name)

        tx_path = paths_dict[tx_name]
        # to get the receive path: return the string of the corresponding transmit path
        rx_path = paths_dict[rx_name[::-1]]

        views[view_name] = c.View(tx_path, rx_path, view_name)
    return views


def make_views(examination_object, probe_oriented_points,
               scatterers_oriented_points, max_number_of_reflection=1,
               tfm_unique_only=False):
    """
    Make views for the measurement model of a block in immersion (scatterers response
    only).

    Parameters
    ----------
    examination_object : arim.core.BlockInImmersion
    probe_oriented_points : OrientedPoints
    scatterers_oriented_points : OrientedPoints
    max_number_of_reflection : int
        Number of internal reflections. Default: 1. If this number is 1 or above, the
        backwall must be defined in ``frame.examination_object``.
    tfm_unique_only : bool
        Default False. If True, returns only the views that give *different* imaging
        results with TFM (AB-CD and DC-BA give the same imaging result).

    Returns
    -------
    views: OrderedDict[Views]

    """
    try:
        couplant = examination_object.couplant_material
        block = examination_object.block_material
        frontwall = examination_object.frontwall
        backwall = examination_object.backwall
    except AttributeError as e:
        raise ValueError("Examination object should be a BlockInImmersion") from e

    interfaces = make_interfaces(
        couplant, probe_oriented_points, frontwall,
        backwall, scatterers_oriented_points)

    paths = make_paths(block, couplant, interfaces, max_number_of_reflection)

    return make_views_from_paths(paths, tfm_unique_only)


@numba.jit(nopython=True, parallel=True)
def _make_transfer_function_singlef(times_tx, times_rx, model_coefficients, freq_array, out):
    """
    Transfer[Scanline, Frequency] = Sum_Scatterer
       exp(2j pi delay[Scanline, Scatterer] frequency[Frequency])
       * model_coefficients[Scanline, Scatterer]

    Parameters
    ----------
    times_tx :
        shape: (numscanlines, numscatterers)
    times_rx
    model_coefficients :
        shape: (numscanlines, numscatterers)
    freq_array
        shape: numfreq
    out :
        Transfer func.
        shape : (numscanlines, numfreq)


    Returns
    -------

    """
    numscanlines = times_tx.shape[0]
    numscatterers = times_tx.shape[1]

    for scan_idx in numba.prange(numscanlines):
        for freq_idx in range(freq_array.shape[0]):
            tmp = 0j
            freq = freq_array[freq_idx]
            for scat_idx in range(numscatterers):
                tmp += model_coefficients[scan_idx, scat_idx] * cmath.exp(
                    -2j * np.pi * freq * (times_tx[scan_idx, scat_idx] + times_rx[scan_idx, scat_idx]))
            out[scan_idx, freq_idx] = tmp
    return out


def singlefreq_scat_transfer_functions(views, tx, rx, frequency, freq_array, scat_obj,
                                       probe_element_width=None,
                                       use_directivity=True, use_beamspread=True,
                                       use_transrefl=True, scat_angle=0.,
                                       numangles_for_scat_precomp=0):
    """
    Transfer function for all views, returned view per view.

    Parameters
    ----------
    views : Dict[Views]
    tx : ndarray
        Shape: (numscanlines, )
    rx : ndarray
        Shape: (numscanlines, )
    frequency : float
    freq_array : ndarray
        Shape: (numfreq, )
    scat_obj : arim.scat.Scattering2d
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    scat_angle : float
    numangles_for_scat_precomp : int
        Number of angles in [-pi, pi] for scattering precomputation.
        0 to disable. See module documentation.

    Yields
    ------
    viewname
        Key of `views`
    partial_transfer_function_f : ndarray
        Shape: (numscanlines, numfreq). Complex. Contribution for one view.


    """
    from ..scat import ScatFromData

    scat_keys_to_compute = set(view.scat_key() for view in views.values())
    # model_amplitudes_factory is way faster with the scattering is given as matrices instead of functions.
    # If matrices can be computed cheaply, it's worth it.
    if isinstance(scat_obj, ScatFromData):
        with helpers.timeit('Scattering', logger):
            scattering = scat_obj.as_single_freq_matrices(frequency, scat_obj.numangles,
                                                          to_compute=scat_keys_to_compute)
    elif numangles_for_scat_precomp > 0:
        with helpers.timeit('Scattering', logger):
            scattering = scat_obj.as_single_freq_matrices(frequency, numangles_for_scat_precomp,
                                                          to_compute=scat_keys_to_compute)
    else:
        scattering = scat_obj.as_angles_funcs(frequency)

    ray_weights = ray_weights_for_views(views, frequency=frequency, probe_element_width=probe_element_width,
                                        use_beamspread=use_beamspread, use_directivity=use_directivity,
                                        use_transrefl=use_transrefl)

    for viewname, view in views.items():
        logger.info('Transfer function for scatterers in view {}'.format(viewname))

        # compute Q_i Q'_j S_ij
        # shape: (numscanlines, numscatterers)
        model_coefficients = model.model_amplitudes_factory(
            tx, rx, view, ray_weights, scattering,
            scat_angle=scat_angle)[...].T
        model_coefficients = np.conj(model_coefficients, out=model_coefficients)

        times_tx = np.take(view.tx_path.rays.times, tx, axis=0)
        times_rx = np.take(view.rx_path.rays.times, rx, axis=0)

        # shape (numscanlines, numfreq)
        partial_transfer_function_f = np.zeros((len(tx), len(freq_array)), np.complex_)

        _make_transfer_function_singlef(times_tx, times_rx, model_coefficients, freq_array, partial_transfer_function_f)

        yield viewname, partial_transfer_function_f


def singlefreq_scat_transfer_function(views, tx, rx, frequency, freq_array, scat_obj,
                                      probe_element_width=None,
                                      use_directivity=True, use_beamspread=True,
                                      use_transrefl=True, scat_angle=0.,
                                      numangles_for_scat_precomp=0):
    """
    Transfer function for all views.

    Parameters
    ----------
    views : dict[Views]
    tx : ndarray
        Shape: (numscanlines, )
    rx : ndarray
        Shape: (numscanlines, )
    frequency : float
    freq_array : ndarray
        Shape: (numfreq, )
    scat_obj : arim.scat.Scattering2d
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    scat_angle : float
    numangles_for_scat_precomp : int
        Number of angles in [-pi, pi] for scattering precomputation.
        0 to disable. See module documentation.

    Returns
    -------
    partial_transfer_function_f : ndarray
        Shape: (numscanlines, numfreq). Complex. Total for all views

    """
    return sum(singlefreq_scat_transfer_functions(
        views, tx, rx, frequency, freq_array, scat_obj, probe_element_width,
        use_directivity, use_beamspread, use_transrefl, scat_angle, numangles_for_scat_precomp)[1])


def multifreq_scat_transfer_functions(views, tx, rx, freq_array, scat_obj,
                                      probe_element_width=None,
                                      use_directivity=True, use_beamspread=True,
                                      use_transrefl=True, scat_angle=0.,
                                      numangles_for_scat_precomp=0):
    """
    Transfer function for all views, returned view per view.

    Parameters
    ----------
    views : Dict[Views]
    tx : ndarray
        Shape: (numscanlines, )
    rx : ndarray
        Shape: (numscanlines, )
    freq_array : ndarray
        Shape: (numfreq, )
    scat_obj : arim.scat.Scattering2d
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    scat_angle : float
    numangles_for_scat_precomp : int
        Number of angles in [-pi, pi] for scattering precomputation.
        0 to disable. See module documentation.

    Yields
    ------
    viewname
        Key of `views`
    partial_transfer_function_f : ndarray
        Shape: (numscanlines, numfreq). Complex. Contribution for one view.


    """
    nonzero_freq_idx = ~np.isclose(freq_array, 0)
    nonzero_freq_array = freq_array[nonzero_freq_idx]
    nonzero_to_all_freq_idx = np.arange(len(freq_array))[nonzero_freq_idx]

    # precompute all ray weights
    ray_weights_allfreq = []
    with helpers.timeit('Computation of ray weights', logger):
        for frequency in nonzero_freq_array:
            # logger.debug(f'ray weight freq={frequency}')
            ray_weights = ray_weights_for_views(views, frequency=frequency, probe_element_width=probe_element_width,
                                                use_beamspread=use_beamspread, use_directivity=use_directivity,
                                                use_transrefl=use_transrefl)
            ray_weights_allfreq.append(ray_weights)

    from ..scat import ScatFromData

    scat_keys_to_compute = set(view.scat_key() for view in views.values())
    # model_amplitudes_factory is way faster with the scattering is given as matrices instead of functions.
    # If matrices can be computed cheaply, it's worth it.
    if isinstance(scat_obj, ScatFromData):
        with helpers.timeit('Scattering', logger):
            scat_matrices = scat_obj.as_multi_freq_matrices(nonzero_freq_array, scat_obj.numangles,
                                                            to_compute=scat_keys_to_compute)
    elif numangles_for_scat_precomp > 0:
        with helpers.timeit('Scattering', logger):
            scat_matrices = scat_obj.as_multi_freq_matrices(nonzero_freq_array, numangles_for_scat_precomp,
                                                            to_compute=scat_keys_to_compute)
    else:
        scat_matrices = None

    for viewname, view in views.items():
        logger.info('Transfer function for scatterers in view {}'.format(viewname))

        partial_transfer_function_f = np.zeros((len(tx), len(freq_array)), np.complex_)

        # shape: (numscanlines, numscatterers)
        delay = (np.take(view.tx_path.rays.times, tx, axis=0) +
                 np.take(view.rx_path.rays.times, rx, axis=0))

        for freq_idx, frequency in enumerate(nonzero_freq_array):
            all_freq_idx = nonzero_to_all_freq_idx[freq_idx]
            # logger.debug(f'transfer func freq={frequency}')

            if scat_matrices:
                scattering = {key: mat[freq_idx] for key, mat in scat_matrices.items()}
            else:
                scattering = scat_obj.as_angles_funcs(frequency)

            ray_weights = ray_weights_allfreq[freq_idx]

            # compute Q_i Q'_j S_ij
            # shape: (numscanlines, numscatterers)
            model_coefficients = model.model_amplitudes_factory(
                tx, rx, view, ray_weights, scattering,
                scat_angle=scat_angle)[...].T

            # Transfer[Scanline, Frequency] = Sum_Scatterer
            #   exp(2j pi delay[Scanline, Scatterer] frequency[Frequency])
            #   * model_coefficients[Scanline, Scatterer]
            partial_transfer_function_f[:, all_freq_idx] = np.sum(
                model_coefficients * np.exp(2j * np.pi * frequency * delay),
                axis=-1
            ).conj()
        yield viewname, partial_transfer_function_f


def multifreq_scat_transfer_function(views, tx, rx, freq_array, scat_obj,
                                     probe_element_width=None,
                                     use_directivity=True, use_beamspread=True,
                                     use_transrefl=True, scat_angle=0.,
                                     numangles_for_scat_precomp=0):
    """
    Transfer function for all views.

    Parameters
    ----------
    views : dict[Views]
    tx : ndarray
        Shape: (numscanlines, )
    rx : ndarray
        Shape: (numscanlines, )
    freq_array : ndarray
        Shape: (numfreq, )
    scat_obj : arim.scat.Scattering2d
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    scat_angle : float
    numangles_for_scat_precomp : int
        Number of angles in [-pi, pi] for scattering precomputation.
        0 to disable. See module documentation.

    Returns
    -------
    partial_transfer_function_f : ndarray
        Shape: (numscanlines, numfreq). Complex. Total for all views

    """
    return sum(multifreq_scat_transfer_functions(
        views, tx, rx, freq_array, scat_obj, probe_element_width,
        use_directivity, use_beamspread, use_transrefl, scat_angle,
        numangles_for_scat_precomp)[1])
