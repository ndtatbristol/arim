"""
Forward model of the inspection of a solid block in immersion

Boilerplate::

    import arim.models.block_in_immersion as bim

    probe_points, probe_orientations = arim.geometry.points_from_probe(probe)
    frontwall_points, frontwall_orientations = \
        arim.geometry.points_1d_wall_z(xmin, xmax, z_frontwall, numpoints)
    backwall_points, backwall_orientations = \
        arim.geometry.points_1d_wall_z(xmin, xmax, z_backwall, numpoints)

    grid = arim.geometry.Grid(xmin, xmax, ymin, ymax, zmin, zmax, pixel_size)
    grid_points, grid_orientation = arim.geometry.points_from_grid(grid)

    interfaces = bim.make_interfaces(couplant, probe_points, probe_orientations,
                                     frontwall_points, frontwall_orientations,
                                     backwall_points, backwall_orientations,
                                     grid_points, grid_orientation)
    paths = bim.make_paths(block, couplant, interfaces, max_number_of_reflection=1)
    views = bim.make_views(paths, unique_only=True)

"""

import logging
from collections import namedtuple, OrderedDict

import numpy as np

from .. import model, ray, ut, helpers
from .. import core as c
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
        # use conjugate because we forget to put it in the model
        weights_dict['transrefl'] = model.transmission_reflection_for_path(
            path, ray_geometry, unit='displacement').conjugate()
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
        # use conjugate because we forget to put it in the model
        weights_dict['transrefl'] = model.reverse_transmission_reflection_for_path(
            path, ray_geometry, unit='displacement').conjugate()
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
        # use conjugate because we forget to put it in the model
        weights_dict['transrefl'] = model.transmission_reflection_for_path(
            path, ray_geometry, unit='displacement').conjugate()
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
                    probe_points, probe_orientations,
                    frontwall_points, frontwall_orientations,
                    backwall_points, backwall_orientations,
                    grid_points, grid_orientations):
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
    probe_points: Points
    probe_orientations: Points
    frontwall_points: Points
    frontwall_orientations: Points
    backwall_points: Points
    backwall_orientations: Points
    grid_points: Points
    grid_orientations: Points

    Returns
    -------
    interface_dict : dict[Interface]
        Keys: probe, frontwall_trans, backwall_refl, grid, frontwall_refl
    """
    interface_dict = OrderedDict()

    interface_dict['probe'] = c.Interface(probe_points, probe_orientations,
                                        are_normals_on_out_rays_side=True)
    interface_dict['frontwall_trans'] = c.Interface(frontwall_points,
                                                  frontwall_orientations,
                                                  'fluid_solid', 'transmission',
                                                  are_normals_on_inc_rays_side=False,
                                                  are_normals_on_out_rays_side=True)
    interface_dict['backwall_refl'] = c.Interface(backwall_points, backwall_orientations,
                                                'solid_fluid', 'reflection',
                                                reflection_against=couplant_material,
                                                are_normals_on_inc_rays_side=False,
                                                are_normals_on_out_rays_side=False)
    interface_dict['grid'] = c.Interface(grid_points, grid_orientations,
                                       are_normals_on_inc_rays_side=True)
    interface_dict['frontwall_refl'] = c.Interface(frontwall_points, frontwall_orientations,
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


def make_views(paths_dict, unique_only=True):
    """
    Returns 'View' objects for the case of a block in immersion.

    Consut all possible views that can be constructed with the paths given as argument.

    If unique only ``unique_only`` is false,

    Parameters
    ----------
    paths_dict : Dict[Path]
        Key: path names (exemple: 'L', 'LT'). Values: :class:`Path`
    unique_only : bool
        Default: True. Returns only the views that give *different* imaging results with
        TFM (AB-CD and DC-BA give the same imaging result).

    Returns
    -------
    views: OrderedDict[Views]

    """
    viewnames = ut.make_viewnames(paths_dict.keys(), unique_only=unique_only)
    views = OrderedDict()
    for view_name_tuple in viewnames:
        tx_name, rx_name = view_name_tuple
        view_name = '{}-{}'.format(tx_name, rx_name)

        tx_path = paths_dict[tx_name]
        # to get the receive path: return the string of the corresponding transmit path
        rx_path = paths_dict[rx_name[::-1]]

        views[view_name] = c.View(tx_path, rx_path, view_name)
    return views
