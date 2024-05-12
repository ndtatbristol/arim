"""
Model for solid block on which the probe is in direct contact

Imaging should work as expected.

The forward model is not finalised and is not experimentally validated. Buyer
beware.

Known issue :

- the model is not reciprocal (swapping the transmitter and the receiver gives
  different results)

Limits of the forward model:

- the scaling between the L and the T directivity of the elements is dubious
- material underneath is fluid
- do not model reflection against the backwall

See also :mod:`arim.models.model.block_in_immersion`

"""
from itertools import product
import logging
from collections import OrderedDict, namedtuple

import numpy as np

from .. import core as c
from .. import helpers, model, ray, signal
from ..ray import RayGeometry
from .helpers import make_views_from_paths

logger = logging.getLogger(__name__)


_RayWeightsCommon = namedtuple(
    "_RayWeightsCommon",
    ["numgridpoints", "wavelengths_in_block"],
)


def _init_ray_weights(path, frequency, probe_element_width, use_directivity):
    if path.rays is None:
        raise ValueError("Ray tracing must have been performed first.")

    block = path.materials[0]
    numgridpoints = len(path.interfaces[-1].points)

    if use_directivity and probe_element_width is None:
        raise ValueError("probe_element_width must be provided to compute directivity")

    if block.transverse_vel is None:
        wavelengths_in_block = dict(
            [(c.Mode.L, block.longitudinal_vel / frequency), (c.Mode.T, float("nan"))]
        )
    else:
        wavelengths_in_block = dict(
            [
                (c.Mode.L, block.longitudinal_vel / frequency),
                (c.Mode.T, block.transverse_vel / frequency),
            ]
        )

    return _RayWeightsCommon(numgridpoints, wavelengths_in_block)


def tx_ray_weights(
    path,
    ray_geometry,
    frequency,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
):
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
    use_attenuation : bool
        Default: True

    Returns
    -------
    weights : ndarray
        Shape (numelements, numgridpoints)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection, attenuation
    """
    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints), order="F")

    if use_directivity:
        if path.modes[0] is c.Mode.L:
            directivity_func = model.directivity_2d_rectangular_on_solid_l
        elif path.modes[0] is c.Mode.T:
            directivity_func = model.directivity_2d_rectangular_on_solid_t
        else:
            raise RuntimeError
        weights_dict["directivity"] = directivity_func(
            ray_geometry.conventional_out_angle(0),
            probe_element_width,
            d.wavelengths_in_block[c.Mode.L],
            d.wavelengths_in_block[c.Mode.T],
        )
    else:
        weights_dict["directivity"] = one
    if use_transrefl and path.numlegs >= 2:
        weights_dict["transrefl"] = model.transmission_reflection_for_path(
            path, ray_geometry, unit="displacement"
        )
    else:
        weights_dict["transrefl"] = one
    if use_beamspread:
        weights_dict["beamspread"] = model.beamspread_2d_for_path(ray_geometry)
    else:
        weights_dict["beamspread"] = one
    if use_attenuation:
        weights_dict["attenuation"] = model.material_attenuation_for_path(
            path, ray_geometry, frequency
        )
    else:
        weights_dict["attenuation"] = one

    weights = (
        weights_dict["directivity"]
        * weights_dict["transrefl"]
        * weights_dict["beamspread"]
        * weights_dict["attenuation"]
    )
    return weights, weights_dict


def rx_ray_weights(
    path,
    ray_geometry,
    frequency,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
):
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
    use_attenuation : bool
        Default: True

    Returns
    -------
    weights : ndarray
        Shape (numelements, numgridpoints)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection, attenuation
    """
    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints), order="F")

    if use_directivity:
        if path.modes[0] is c.Mode.L:
            directivity_func = model.directivity_2d_rectangular_on_solid_l
        elif path.modes[0] is c.Mode.T:
            directivity_func = model.directivity_2d_rectangular_on_solid_t
        else:
            raise RuntimeError
        weights_dict["directivity"] = directivity_func(
            ray_geometry.conventional_out_angle(0),
            probe_element_width,
            d.wavelengths_in_block[c.Mode.L],
            d.wavelengths_in_block[c.Mode.T],
        )
    else:
        weights_dict["directivity"] = one
    if use_transrefl and path.numlegs >= 2:
        weights_dict["transrefl"] = model.reverse_transmission_reflection_for_path(
            path, ray_geometry, unit="displacement"
        )
    else:
        weights_dict["transrefl"] = one
    if use_beamspread:
        weights_dict["beamspread"] = model.reverse_beamspread_2d_for_path(ray_geometry)
    else:
        weights_dict["beamspread"] = one
    if use_attenuation:
        weights_dict["attenuation"] = model.material_attenuation_for_path(
            path, ray_geometry, frequency
        )
    else:
        weights_dict["attenuation"] = one

    # the coefficient accounts for the normalisation convention of the scattering in Bristol's literature
    scat_normalisation = np.sqrt(d.wavelengths_in_block[path.modes[-1]])
    weights = (
        weights_dict["directivity"]
        * weights_dict["transrefl"]
        * weights_dict["beamspread"]
        * weights_dict["attenuation"]
    )
    # if path.modes[-1] is c.Mode.L:
    #     reception_coeff = d.wavelengths_in_block[c.Mode.L]**1.5
    # elif path.modes[-1] is c.Mode.T:
    #     reception_coeff = -d.wavelengths_in_block[c.Mode.T]**1.5
    # else:
    #     raise RuntimeError
    # weights *= scat_normalisation * reception_coeff
    weights *= scat_normalisation
    return weights, weights_dict


def _make_backwall_refl_interface(backwall, under_material):
    if under_material is not None:
        backwall_refl = c.Interface(
            *backwall,
            "solid_fluid",
            "reflection",
            reflection_against=under_material,
            are_normals_on_inc_rays_side=False,
            are_normals_on_out_rays_side=False,
        )
    else:
        backwall_refl = c.Interface(
            *backwall,
            are_normals_on_inc_rays_side=False,
            are_normals_on_out_rays_side=False,
        )
    return backwall_refl


def backwall_paths(
    block_material, probe_oriented_points, backwall, under_material=None
):
    """
    Make backwall paths LL, LT, TL, TT

    Probe -> block -> backwall > block -> probe

    Parameters
    ----------
    block_material : Material
    probe_oriented_points : OrientedPoints
    backwall: OrientedPoints
    under_material : Material or None

    Returns
    -------
    OrderedDict of Path
        Keys: LL, LT, TL, TT

    """
    probe_start = c.Interface(*probe_oriented_points, are_normals_on_out_rays_side=True)
    backwall_refl = _make_backwall_refl_interface(backwall, under_material)
    probe_end = c.Interface(*probe_oriented_points, are_normals_on_inc_rays_side=True)

    paths = OrderedDict()

    for mode1 in (c.Mode.L, c.Mode.T):
        for mode2 in (c.Mode.L, c.Mode.T):
            key = mode1.key() + mode2.key()
            paths[key] = c.Path(
                interfaces=(
                    probe_start,
                    backwall_refl,
                    probe_end,
                ),
                materials=(
                    block_material,
                    block_material,
                ),
                modes=(mode1, mode2),
                name="Backwall " + key,
            )
    return paths


def ray_weights_for_wall(
    path,
    frequency,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
):
    """
    Compute model coefficients for wall echoes.

    Parameters
    ----------
    path
    frequency
    probe_element_width
    use_directivity
    use_beamspread
    use_transrefl
    use_attenuation

    Returns
    -------
    weights : ndarray
        Shape (numelements, numelements)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection, attenuation

    """
    # perform ray tracing if needed
    if path.rays is None:
        ray.ray_tracing_for_paths([path])

    ray_geometry = RayGeometry.from_path(path)

    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints), order="F")

    if use_directivity:
        if path.modes[0] is c.Mode.L:
            directivity_func_tx = model.directivity_2d_rectangular_on_solid_l
        elif path.modes[0] is c.Mode.T:
            directivity_func_tx = model.directivity_2d_rectangular_on_solid_t
        if path.modes[-1] is c.Mode.L:
            directivity_func_rx = model.directivity_2d_rectangular_on_solid_l
        elif path.modes[-1] is c.Mode.T:
            directivity_func_rx = model.directivity_2d_rectangular_on_solid_t
        directivity_tx = directivity_func_tx(
            ray_geometry.conventional_out_angle(0),
            probe_element_width,
            d.wavelengths_in_block[c.Mode.L],
            d.wavelengths_in_block[c.Mode.T],
        )
        directivity_rx = directivity_func_rx(
            ray_geometry.conventional_inc_angle(-1),
            probe_element_width,
            d.wavelengths_in_block[c.Mode.L],
            d.wavelengths_in_block[c.Mode.T],
        )
        weights_dict["directivity"] = directivity_tx * directivity_rx
    else:
        weights_dict["directivity"] = one
    if use_transrefl:
        weights_dict["transrefl"] = model.transmission_reflection_for_path(
            path, ray_geometry, unit="displacement"
        )
    else:
        weights_dict["transrefl"] = one
    if use_beamspread:
        weights_dict["beamspread"] = model.beamspread_2d_for_path(ray_geometry)
    else:
        weights_dict["beamspread"] = one
    if use_attenuation:
        weights_dict["attenuation"] = model.material_attenuation_for_path(
            path, ray_geometry, frequency
        )
    else:
        weights_dict["attenuation"] = one

    weights = (
        weights_dict["directivity"]
        * weights_dict["transrefl"]
        * weights_dict["beamspread"]
        * weights_dict["attenuation"]
    )
    return weights, weights_dict


def make_interfaces(
    probe_oriented_points,
    grid_oriented_points,
    reflecting_walls=None,
    under_material=None,
):
    """    
    Construct interfaces for the case of a solid block in contact with the
    probe.

    The interfaces are for rays starting from the probe and arriving in the
    grid. Additional walls can be included to allow reflections.
    
    Assumes that walls are provided in the order that reflection is expected
    (i.e. for 2 reflections from the backwall and then the frontwall, then
    walls should contain them in this order). Assumes all that walls have
    orientation facing into the solid.

    Parameters
    ----------
    probe_oriented_points : OrientedPoints
    grid_oriented_points: OrientedPoints
    reflecting_walls: list[OrientedPoints] or None
    under_material : Material or None

    Returns
    -------
    interface_dict : dict[Interface]
        Keys: probe, grid, wall_name_1 (optional), ...
    """
    interface_dict = OrderedDict()

    interface_dict["probe"] = c.Interface(
        *probe_oriented_points, are_normals_on_out_rays_side=True
    )
    interface_dict["grid"] = c.Interface(
        *grid_oriented_points, are_normals_on_inc_rays_side=True
    )
    if reflecting_walls is not None:
        for wall in reflecting_walls:
            name = wall.points.name
            if name != "frontwall" and under_material is not None:
                kind = "solid_fluid"
                transmission_reflection = "reflection"
            else:
                kind = None
                transmission_reflection = None
            
            interface_dict[name] = c.Interface(
                *wall,
                kind,
                transmission_reflection,
                are_normals_on_inc_rays_side=True,
                are_normals_on_out_rays_side=True,
                reflection_against=under_material,
            )
    return interface_dict


def make_paths(
        block_material, interface_dict, max_number_of_reflection=0
    ):
    """
    Creates a dictionary a Paths for the block-in-contact model.

    Paths are returned in transmit convention: for the path XY, X is the mode
    before reflection against the backwall and Y is the mode after reflection.
    The path XY in transmit convention is the path YX in receive convention.

    Parameters
    ----------
    block_material : Material
    interface_dict : dict[Interface]
        Use ``make_interfaces()`` to create
    max_number_of_reflection : int
        Default: 0

    Returns
    -------
    paths : OrderedDict

    """
    if max_number_of_reflection > 2:
        raise NotImplementedError
    if max_number_of_reflection < 0:
        raise ValueError

    paths = OrderedDict()
    probe = interface_dict["probe"]
    grid = interface_dict["grid"]
    wall_dict = OrderedDict((key, value) for key, value in interface_dict.items()
                            if key not in ["probe", "grid"])
    wall_names = list(wall_dict.keys())
    
    mode_names = ("L", "T")
    modes = (c.Mode.longitudinal. c.Mode.transverse)
    for no_reflections in range(max_number_of_reflection+1):
        # For this number of reflections, make all the combinations of paths.
        path_idxs_up_to_refl = list(product(range(2), repeat=no_reflections+1))
        for path_idxs in path_idxs_up_to_refl:
            # For each path with this number of reflections.
            path_name = ""
            path_modes = []
            path_interfaces = [probe]
            path_materials = []
            
            for i, mode in enumerate(path_idxs):
                if i == 0:
                    path_name += mode_names[mode]
                else:
                    # Have to settle on a naming convention.
                    # Published work to date (~2023) joins wave modes and assumes a constant wall (typically back wall).
                    # New `Path` method `longname` splices wall names into the mode names to indicate which wall was skipped from. Preserve simple naming convention for path dict keys.
                    # If multiple paths with the same modes but different wall skips are needed, they will need to be stored in different dicts. Edit this string if this is inconvenient.
                    path_name += "{}".format(mode_names[mode])
                    path_interfaces.append(wall_dict[wall_names[i-1]])
                path_modes.append(modes[mode])
                path_materials.append(block_material)
            path_interfaces.append(grid)
            
            paths[path_name] = c.Path(
                interfaces=path_interfaces,
                materials=path_materials,
                modes=path_modes,
                name=path_name,
            )
    return paths


def make_views(
    examination_object,
    probe_oriented_points,
    grid_oriented_points,
    max_number_of_reflection=0,
    tfm_unique_only=False,
):
    """
    Make views for the measurement model of a block in contact (scatterers response
    only).

    Parameters
    ----------
    examination_object : arim.BlockInContact or arim.ExaminationObject
    probe_oriented_points : OrientedPoints
    grid_oriented_points : OrientedPoints
        Scatterers (for forward model) or grid (for imaging)
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
        block_material = examination_object.block_material
    except AttributeError:
        # Plan B
        block_material = examination_object.material
    try:
        walls = [examination_object.walls[i]
                 for i in examination_object.wall_idxs_for_imaging]
        if max_number_of_reflection > 0 and len(walls) < 1:
            raise ValueError("Not enough walls available for reflection.")
    except AttributeError:
        walls = None
    try:
        under_material = examination_object.under_material
    except AttributeError:
        under_material = None
    interfaces = make_interfaces(
        probe_oriented_points,
        grid_oriented_points,
        walls=walls,
        under_material=under_material,
    )
    paths = make_paths(block_material, interfaces, max_number_of_reflection)
    return make_views_from_paths(paths, tfm_unique_only)


# ----------------
# From now on, almost perfect copy/paste of block_in_immersion.
# To factorise!


def ray_weights_for_views(
    views,
    frequency,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
    save_debug=False,
):
    """
    Compute coefficients Q_i(r, omega) and Q'_j(r, omega) from the forward model for
    all views.

    NB: do not compute the scattering.

    Internally use :func:`tx_ray_weights` and :func:`rx_way_weights`.

    Parameters
    ----------
    views : dict[Views]
    frequency : float
    probe_element_width : float
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    use_attenuation : bool
    save_debug : bool

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

    model_options = dict(
        frequency=frequency,
        probe_element_width=probe_element_width,
        use_beamspread=use_beamspread,
        use_directivity=use_directivity,
        use_transrefl=use_transrefl,
        use_attenuation=use_attenuation,
    )

    # By proceeding this way, geometrical computations can be reused for both
    # tx and rx path.
    for path in all_paths:
        ray_geometry = RayGeometry.from_path(path)
        scat_angle_dict[path] = ray_geometry.signed_inc_angle(-1)
        scat_angle_dict[path].flags.writeable = False

        if path in all_tx_paths:
            ray_weights, ray_weights_debug = tx_ray_weights(
                path, ray_geometry, **model_options
            )
            ray_weights.flags.writeable = False
            tx_ray_weights_dict[path] = ray_weights
            if save_debug:
                tx_ray_weights_debug_dict[path] = ray_weights_debug
            del ray_weights, ray_weights_debug
        if path in all_rx_paths:
            ray_weights, ray_weights_debug = rx_ray_weights(
                path, ray_geometry, **model_options
            )
            ray_weights.flags.writeable = False
            rx_ray_weights_dict[path] = ray_weights
            if save_debug:
                rx_ray_weights_debug_dict[path] = ray_weights_debug
            del ray_weights, ray_weights_debug

    return model.RayWeights(
        tx_ray_weights_dict,
        rx_ray_weights_dict,
        tx_ray_weights_debug_dict,
        rx_ray_weights_debug_dict,
        scat_angle_dict,
    )


def scat_unshifted_transfer_functions(
    views,
    tx,
    rx,
    freq_array,
    scat_obj,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
    scat_angle=0.0,
    numangles_for_scat_precomp=0,
    first_nonzero_freq_idx=None,
):
    """
    Compute unshifted transfer functions for scatterer echoes (multi-frequency model).

    Returns ``H_ij(omega) = Q_i(omega) Q'_j(omega) S(omega, theta_i, theta_j)``

    Output spectra uses the *math* Fourier convention (not the acoustics one).

    Parameters
    ----------
    views : Dict[Views]
    tx : ndarray
        Shape: (numtimetraces, )
    rx : ndarray
        Shape: (numtimetraces, )
    freq_array : ndarray or float
        Shape: (numfreq, )
    scat_obj : arim.scat.Scattering2d
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    use_attenuation : bool
    scat_angle : float
    numangles_for_scat_precomp : int
        Number of angles in [-pi, pi] for scattering precomputation.
        0 to disable. See module documentation.
    first_nonzero_freq_idx : int or None
        Default: assumes first freq is zero, except if only one freq is given.

    Yields
    ------
    partial_transfer_function_f : ndarray
        Shape: (numscatterers, numtimetraces, numfreq). Complex. Contribution for one view.
    delays : ndarray
        Shape: (numscatterers, numtimetraces). Float. Contribution for one view.

    See Also
    --------
    :func:`arim.signal.timeshift_spectra`

    """
    freq_array = np.atleast_1d(freq_array)
    numfreq = len(freq_array)

    if first_nonzero_freq_idx is None:
        if numfreq == 1:
            # assume the freq is nonzero
            first_nonzero_freq_idx = 0
        else:
            # assume only the first freq is zero, as returned by fftfreq
            first_nonzero_freq_idx = 1
    nonzero_freq_array = freq_array[first_nonzero_freq_idx:]

    # Precompute all ray weights
    ray_weights_allfreq = []
    with helpers.timeit("Computation of ray weights", logger):
        for frequency in nonzero_freq_array:
            # logger.debug(f'ray weight freq={frequency}')
            ray_weights = ray_weights_for_views(
                views,
                frequency=frequency,
                probe_element_width=probe_element_width,
                use_beamspread=use_beamspread,
                use_directivity=use_directivity,
                use_transrefl=use_transrefl,
                use_attenuation=use_attenuation,
            )
            ray_weights_allfreq.append(ray_weights)

    # (Pre)compute scattering
    from ..scat import ScatFromData

    scat_keys_to_compute = set(view.scat_key() for view in views.values())
    # model_amplitudes_factory is way faster with the scattering is given as matrices instead of functions.
    # If matrices can be computed cheaply, it's worth it.
    if isinstance(scat_obj, ScatFromData):
        with helpers.timeit("Scattering", logger):
            scat_matrices = scat_obj.as_multi_freq_matrices(
                nonzero_freq_array, scat_obj.numangles, to_compute=scat_keys_to_compute
            )
    elif numangles_for_scat_precomp > 0:
        with helpers.timeit("Scattering", logger):
            scat_matrices = scat_obj.as_multi_freq_matrices(
                nonzero_freq_array,
                numangles_for_scat_precomp,
                to_compute=scat_keys_to_compute,
            )
    else:
        scat_matrices = None

    numtimetraces = len(tx)

    for view in views.values():
        logger.info(f"Transfer function for scatterers in view {view.name}")

        numscatterers = view.tx_path.rays.times.shape[1]
        partial_transfer_function_f = np.zeros(
            (numscatterers, numtimetraces, numfreq), np.complex_
        )

        # shape: (numscatterers, numtimetraces)
        delays = np.ascontiguousarray(
            (
                np.take(view.tx_path.rays.times, tx, axis=0)
                + np.take(view.rx_path.rays.times, rx, axis=0)
            ).T
        )

        for freq_idx, frequency in enumerate(nonzero_freq_array):
            freq_idx2 = first_nonzero_freq_idx + freq_idx

            if scat_matrices:
                scattering = {key: mat[freq_idx] for key, mat in scat_matrices.items()}
            else:
                scattering = scat_obj.as_angles_funcs(frequency)

            ray_weights = ray_weights_allfreq[freq_idx]

            # compute Q_i Q'_j S_ij
            # shape: (numscatterers, numtimetraces, )
            model_coefficients = model.model_amplitudes_factory(
                tx, rx, view, ray_weights, scattering, scat_angle=scat_angle
            )[...]
            np.conj(model_coefficients, out=model_coefficients)

            partial_transfer_function_f[..., freq_idx2] = model_coefficients

        yield partial_transfer_function_f, delays


def wall_unshifted_transfer_functions(
    wall_paths,
    tx,
    rx,
    freq_array,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
    first_nonzero_freq_idx=None,
):
    """Compute unshifted transfer functions for walls echoes.

    Output spectra uses the *math* Fourier convention (not the acoustics one).

    Parameters
    ----------
    wall_paths : Dict[arim.Path]
    tx : ndarray
        Shape: (numtimetraces, )
    rx : ndarray
        Shape: (numtimetraces, )
    freq_array : ndarray or float
        Shape: (numfreq, )
    probe_element_width : [type], optional
        [description] (the default is None, which [default_description])
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    use_attenuation : bool
    first_nonzero_freq_idx : int or None
        Default: assumes first freq is zero, except if only one freq is given.

    Yields
    ------
    partial_transfer_function_f : ndarray
        Shape: (numtimetraces, numfreq). Complex. Contribution for one wall path.
    delays : ndarray
        Shape: (numtimetraces). Float. Contribution for wall path.
    """
    freq_array = np.atleast_1d(freq_array)
    numfreq = len(freq_array)

    if first_nonzero_freq_idx is None:
        if numfreq == 1:
            # assume the freq is nonzero
            first_nonzero_freq_idx = 0
        else:
            # assume only the first freq is zero, as returned by fftfreq
            first_nonzero_freq_idx = 1
    nonzero_freq_array = freq_array[first_nonzero_freq_idx:]

    for pathname, path in wall_paths.items():
        logger.info(f"Transfer function for wall {pathname}")

        partial_transfer_function_f = np.zeros((len(tx), len(freq_array)), np.complex_)

        for freq_idx, frequency in enumerate(nonzero_freq_array):
            freq_idx2 = first_nonzero_freq_idx + freq_idx

            # shape: (numelements, numelements)
            ray_weights, _ = ray_weights_for_wall(
                path,
                frequency=frequency,
                probe_element_width=probe_element_width,
                use_beamspread=use_beamspread,
                use_directivity=use_directivity,
                use_transrefl=use_transrefl,
                use_attenuation=use_attenuation,
            )

            # Fancy indexing:
            partial_transfer_function_f[:, freq_idx2] = ray_weights[tx, rx].conj()
            delays = path.rays.times[tx, rx]

        yield partial_transfer_function_f, delays


def singlefreq_scat_transfer_functions(
    views,
    tx,
    rx,
    frequency,
    freq_array,
    scat_obj,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
    scat_angle=0.0,
    numangles_for_scat_precomp=0,
):
    """
    Compute transfer functions for scatterer echoes (single-frequency model).

    Output spectra uses the *math* Fourier convention (not the acoustics one).

    Parameters
    ----------
    views : Dict[Views]
    tx : ndarray
        Shape: (numtimetraces, )
    rx : ndarray
        Shape: (numtimetraces, )
    frequency : float
    freq_array : ndarray
        Shape: (numfreq, )
    scat_obj : arim.scat.Scattering2d
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    use_attenuation : bool
    scat_angle : float
    numangles_for_scat_precomp : int
        Number of angles in [-pi, pi] for scattering precomputation.
        0 to disable. See module documentation.

    Yields
    ------
    viewname : str
        Key of `views`
    partial_transfer_function_f : ndarray
        Shape: (numtimetraces, numfreq). Complex. Contribution for one view.

    Notes
    -----
    Legacy function, superseeded by :func:`scat_unshifted_transfer_functions`
    and :func:`arim.signal.timeshift_spectra`.
    """
    unshifted_tfs = scat_unshifted_transfer_functions(
        views,
        tx,
        rx,
        frequency,
        scat_obj,
        probe_element_width=probe_element_width,
        use_directivity=use_directivity,
        use_beamspread=use_beamspread,
        use_transrefl=use_transrefl,
        use_attenuation=use_attenuation,
        scat_angle=scat_angle,
        numangles_for_scat_precomp=numangles_for_scat_precomp,
    )

    for viewname, (unshifted_tf, delays) in zip(views.keys(), unshifted_tfs):
        # shape (numscatterers, numtimetraces, numfreq)
        tf = signal.timeshift_spectra(unshifted_tf, delays, freq_array)

        # lazy tf.sum(axis=0):
        if tf.shape[0] == 1:
            tf = tf[0]
        else:
            tf = tf.sum(axis=0)
        yield viewname, tf


def singlefreq_wall_transfer_functions(
    wall_paths,
    tx,
    rx,
    frequency,
    freq_array,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
):
    """
    Compute transfer functions for wall echoes (single-frequency model).

    Parameters
    ----------
    wall_paths : Dict[Path]
    tx : ndarray
        Shape: (numtimetraces, )
    rx : ndarray
        Shape: (numtimetraces, )
    frequency : float
        Frequency at which the model runs.
    freq_array : ndarray
        Shape: (numfreq, ). First freq is assumed to be zero.
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    use_attenuation : bool

    Yields
    ------
    pathname : str
        Key of `wall_paths`
    partial_transfer_function_f : ndarray
        Shape: (numtimetraces, numfreq). Complex. Contribution for one path.

    Notes
    -----
    Legacy function, superseeded by :func:`wall_unshifted_transfer_functions`
    and :func:`arim.signal.timeshift_spectra`.

    """
    unshifted_tfs = wall_unshifted_transfer_functions(
        wall_paths,
        tx,
        rx,
        frequency,
        probe_element_width=probe_element_width,
        use_directivity=use_directivity,
        use_beamspread=use_beamspread,
        use_transrefl=use_transrefl,
        use_attenuation=use_attenuation,
    )

    for pathname, (unshifted_tf, delays) in zip(wall_paths.keys(), unshifted_tfs):
        tf = signal.timeshift_spectra(unshifted_tf, delays, freq_array)
        yield pathname, tf


def multifreq_scat_transfer_functions(
    views,
    tx,
    rx,
    freq_array,
    scat_obj,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
    scat_angle=0.0,
    numangles_for_scat_precomp=0,
):
    """
    Compute transfer functions for scatterer echoes (multi-frequency model).

    Output spectra uses the *math* Fourier convention (not the acoustics one).

    Parameters
    ----------
    views : Dict[Views]
    tx : ndarray
        Shape: (numtimetraces, )
    rx : ndarray
        Shape: (numtimetraces, )
    freq_array : ndarray
        Shape: (numfreq, )
    scat_obj : arim.scat.Scattering2d
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    use_attenuation : bool
    scat_angle : float
    numangles_for_scat_precomp : int
        Number of angles in [-pi, pi] for scattering precomputation.
        0 to disable. See module documentation.

    Yields
    ------
    viewname : str
        Key of `views`
    partial_transfer_function_f : ndarray
        Shape: (numtimetraces, numfreq). Complex. Contribution for one view.

    Notes
    -----
    Legacy function, superseeded by :func:`scat_unshifted_transfer_functions`
    and :func:`arim.signal.timeshift_spectra`.
    """
    unshifted_tfs = scat_unshifted_transfer_functions(
        views,
        tx,
        rx,
        freq_array,
        scat_obj,
        probe_element_width=probe_element_width,
        use_directivity=use_directivity,
        use_beamspread=use_beamspread,
        use_transrefl=use_transrefl,
        use_attenuation=use_attenuation,
        scat_angle=scat_angle,
        numangles_for_scat_precomp=numangles_for_scat_precomp,
    )

    for viewname, (unshifted_tf, delays) in zip(views.keys(), unshifted_tfs):
        # shape (numscatterers, numtimetraces, numfreq)
        tf = signal.timeshift_spectra(unshifted_tf, delays, freq_array)

        # lazy tf.sum(axis=0):
        if tf.shape[0] == 1:
            tf = tf[0]
        else:
            tf = tf.sum(axis=0)
        yield viewname, tf


def multifreq_wall_transfer_functions(
    wall_paths,
    tx,
    rx,
    freq_array,
    probe_element_width=None,
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
):
    """
    Compute transfer functions for scatterer echoes (multi-frequency model).

    Parameters
    ----------
    wall_paths : Dict[Path]
    tx : ndarray
        Shape: (numtimetraces, )
    rx : ndarray
        Shape: (numtimetraces, )
    freq_array : ndarray
        Shape: (numfreq, ). First freq is assumed to be zero.
    probe_element_width : float or None
    use_directivity : bool
    use_beamspread : bool
    use_transrefl : bool
    use_attenuation : bool

    Yields
    ------
    pathname : str
        Key of `wall_paths`
    partial_transfer_function_f : ndarray
        Shape: (numtimetraces, numfreq). Complex. Contribution for one path.

    Notes
    -----
    Legacy function, superseeded by :func:`wall_unshifted_transfer_functions`
    and :func:`arim.signal.timeshift_spectra`.
    """
    unshifted_tfs = wall_unshifted_transfer_functions(
        wall_paths,
        tx,
        rx,
        freq_array,
        probe_element_width=probe_element_width,
        use_directivity=use_directivity,
        use_beamspread=use_beamspread,
        use_transrefl=use_transrefl,
        use_attenuation=use_attenuation,
    )

    for pathname, (unshifted_tf, delays) in zip(wall_paths.keys(), unshifted_tfs):
        tf = signal.timeshift_spectra(unshifted_tf, delays, freq_array)
        yield pathname, tf
