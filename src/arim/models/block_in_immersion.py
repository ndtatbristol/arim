"""
Forward model of the inspection of a solid block in immersion

Calculating the frequency-domain **scatterer** responses:

- :func:`scat_unshifted_transfer_functions`: base function, no time-shift
- :func:`singlefreq_scat_transfer_functions`
- :func:`multifreq_scat_transfer_functions`

Calculating the frequency-domain **wall** responses:

- :func:`wall_unshifted_transfer_functions`: base function, no time-shift
- :func:`singlefreq_wall_transfer_functions`
- :func:`multifreq_wall_transfer_functions`

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
from itertools import product
import logging
import warnings
from collections import OrderedDict, namedtuple

import numpy as np

from .. import core as c
from .. import helpers, model, ray, signal
from ..ray import RayGeometry
from .helpers import make_views_from_paths

logger = logging.getLogger(__name__)

_RayWeightsCommon = namedtuple(
    "_RayWeightsCommon",
    ["couplant", "numgridpoints", "wavelength_in_couplant", "wavelengths_in_block"],
)


def _init_ray_weights(path, frequency, probe_element_width, use_directivity):
    if path.rays is None:
        raise ValueError("Ray tracing must have been performed first.")

    couplant, block = path.materials[:2]
    numgridpoints = len(path.interfaces[-1].points)

    if use_directivity and probe_element_width is None:
        raise ValueError("probe_element_width must be provided to compute directivity")

    wavelength_in_couplant = couplant.longitudinal_vel / frequency
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

    return _RayWeightsCommon(
        couplant, numgridpoints, wavelength_in_couplant, wavelengths_in_block
    )


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
        weights_dict[
            "directivity"
        ] = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant
        )
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
        weights_dict[
            "directivity"
        ] = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant
        )
    else:
        weights_dict["directivity"] = one
    if use_transrefl:
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
    weights *= scat_normalisation
    return weights, weights_dict


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


def frontwall_path(
    couplant_material,
    block_material,
    probe_points,
    probe_orientations,
    frontwall_points,
    frontwall_orientations,
):
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
    probe_start = c.Interface(
        probe_points, probe_orientations, are_normals_on_out_rays_side=True
    )
    probe_end = c.Interface(
        probe_points, probe_orientations, are_normals_on_inc_rays_side=True
    )
    frontwall_ext_refl = c.Interface(
        frontwall_points,
        frontwall_orientations,
        "fluid_solid",
        "reflection",
        reflection_against=block_material,
        are_normals_on_inc_rays_side=False,
        are_normals_on_out_rays_side=False,
    )

    return c.Path(
        interfaces=(probe_start, frontwall_ext_refl, probe_end),
        materials=(couplant_material, couplant_material),
        modes=(c.Mode.L, c.Mode.L),
        name="Frontwall",
    )


def backwall_paths(
    couplant_material,
    block_material,
    probe_oriented_points,
    frontwall,
    backwall,
    max_number_of_reflection=1,
):
    """
    Make backwall paths

    when max_backwall_refl == 1

    Probe -> couplant -> frontwall -> block (L or T) -> backwall -> block (L or T) -> frontwall -> couplant -> probe

    (additional) when max_backwall_refl == 2

    Probe -> couplant -> frontwall -> block (L or T) -> backwall -> block (L or T) -> frontwall
            -> block (L or T) -> backwall -> block (L or T) -> frontwall -> couplant -> probe

    (additional) when max_backwall_refl == 3

    Probe -> couplant -> frontwall -> block (L or T) -> backwall -> block (L or T) -> frontwall
            -> block (L or T) -> backwall -> block (L or T) -> frontwall
            -> block (L or T) -> backwall -> block (L or T) -> frontwall -> couplant -> probe


    Parameters
    ----------
    couplant_material : Material
    block_material : Material
    probe_oriented_points : OrientedPoints
    frontwall: OrientedPoints
    backwall: OrientedPoints
    max_number_of_reflection : int
        Number of internal reflections. Default: 1.


    Returns
    -------
    OrderedDict of Path
        Keys: LL, LT, TL, TT
        (additional) LLLL, LLLT, LLTL, LLTT, LTLT, LTTL LTTT, TLLT, TLTT, TTTT
        (additional, ...refl=3) left to user to work out...

    """

    if max_number_of_reflection > 3:
        msg = "The maximum number of backwall reflections exceeds coding limit (3)"
        raise ValueError(msg)

    probe_start = c.Interface(*probe_oriented_points, are_normals_on_out_rays_side=True)

    frontwall_couplant_to_block = c.Interface(
        *frontwall,
        "fluid_solid",
        "transmission",
        are_normals_on_inc_rays_side=False,
        are_normals_on_out_rays_side=True,
    )

    backwall_refl = c.Interface(
        *backwall,
        "solid_fluid",
        "reflection",
        reflection_against=couplant_material,
        are_normals_on_inc_rays_side=False,
        are_normals_on_out_rays_side=False,
    )

    frontwall_block_to_couplant = c.Interface(
        *frontwall,
        "solid_fluid",
        "transmission",
        are_normals_on_inc_rays_side=True,
        are_normals_on_out_rays_side=False,
    )

    frontwall_refl = c.Interface(
        *frontwall,
        "solid_fluid",
        "reflection",
        reflection_against=couplant_material,
        are_normals_on_inc_rays_side=True,
        are_normals_on_out_rays_side=True,
    )

    probe_end = c.Interface(*probe_oriented_points, are_normals_on_inc_rays_side=True)

    paths = OrderedDict()

    for mode1 in (c.Mode.L, c.Mode.T):
        for mode2 in (c.Mode.L, c.Mode.T):
            key = mode1.key() + mode2.key()
            paths[key] = c.Path(
                interfaces=(
                    probe_start,
                    frontwall_couplant_to_block,
                    backwall_refl,
                    frontwall_block_to_couplant,
                    probe_end,
                ),
                materials=(
                    couplant_material,
                    block_material,
                    block_material,
                    couplant_material,
                ),
                modes=(c.Mode.L, mode1, mode2, c.Mode.L),
                name="Backwall " + key,
            )

    if max_number_of_reflection <= 1:
        return paths

    for mode1 in (c.Mode.L, c.Mode.T):
        for mode2 in (c.Mode.L, c.Mode.T):
            for mode3 in (c.Mode.L, c.Mode.T):
                for mode4 in (c.Mode.L, c.Mode.T):
                    key = mode1.key() + mode2.key() + mode3.key() + mode4.key()
                    paths[key] = c.Path(
                        interfaces=(
                            probe_start,
                            frontwall_couplant_to_block,
                            backwall_refl,
                            frontwall_refl,
                            backwall_refl,
                            frontwall_block_to_couplant,
                            probe_end,
                        ),
                        materials=(
                            couplant_material,
                            block_material,
                            block_material,
                            block_material,
                            block_material,
                            couplant_material,
                        ),
                        modes=(c.Mode.L, mode1, mode2, mode3, mode4, c.Mode.L),
                        name="Backwall " + key,
                    )

    if max_number_of_reflection == 2:
        return paths

    for mode1 in (c.Mode.L, c.Mode.T):
        for mode2 in (c.Mode.L, c.Mode.T):
            for mode3 in (c.Mode.L, c.Mode.T):
                for mode4 in (c.Mode.L, c.Mode.T):
                    for mode5 in (c.Mode.L, c.Mode.T):
                        for mode6 in (c.Mode.L, c.Mode.T):
                            key = (
                                mode1.key()
                                + mode2.key()
                                + mode3.key()
                                + mode4.key()
                                + mode5.key()
                                + mode6.key()
                            )
                            paths[key] = c.Path(
                                interfaces=(
                                    probe_start,
                                    frontwall_couplant_to_block,
                                    backwall_refl,
                                    frontwall_refl,
                                    backwall_refl,
                                    frontwall_refl,
                                    backwall_refl,
                                    frontwall_block_to_couplant,
                                    probe_end,
                                ),
                                materials=(
                                    couplant_material,
                                    block_material,
                                    block_material,
                                    block_material,
                                    block_material,
                                    block_material,
                                    block_material,
                                    couplant_material,
                                ),
                                modes=(
                                    c.Mode.L,
                                    mode1,
                                    mode2,
                                    mode3,
                                    mode4,
                                    mode5,
                                    mode6,
                                    c.Mode.L,
                                ),
                                name="Backwall " + key,
                            )

    return paths


def backwall_paths2(
    couplant_material,
    block_material,
    probe_oriented_points,
    frontwall,
    backwall,
    max_backwall_refl=1,
):
    warnings.warn("Deprecated, use backwall_paths() instead", DeprecationWarning)
    return backwall_paths(
        couplant_material,
        block_material,
        probe_oriented_points,
        frontwall,
        backwall,
        max_number_of_reflection=max_backwall_refl,
    )


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
        directivity = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant
        )

        weights_dict["directivity"] = directivity * directivity.T
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
    couplant_material, probe_oriented_points, walls, grid_oriented_points
):
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
    walls: list[OrientedPoints]
    grid_oriented_points: OrientedPoints

    Returns
    -------
    interface_dict : dict[Interface]
        Keys: probe, frontwall_trans, grid, wall_name_1 (optional), ...
    """
    interface_dict = OrderedDict()

    interface_dict["probe"] = c.Interface(
        *probe_oriented_points, are_normals_on_out_rays_side=True
    )
    interface_dict["grid"] = c.Interface(
        *grid_oriented_points, are_normals_on_inc_rays_side=True
    )
    for wall in walls:
        name = wall.points.name
        if name == "Frontwall":
            # Need both transmission and reflection for frontwall
            name = wall.points.name + "_trans"
            interface_dict[name+"_trans"] = c.Interface(
                *wall,
                kind="fluid_solid",
                transmission_reflection="transmission",
                reflection_against=None,
                are_normals_on_inc_rays_side=False,
                are_normals_on_out_rays_side=True,
            )
        else:
            # Ideally want to allow front wall to be reflected as well for later reflections.
            # Too much of a headache for now, when ready remove this else block.
            interface_dict[name] = c.Interface(
                *wall,
                kind="solid_fluid",
                transmission_reflection="reflection",
                reflection_against=couplant_material,
                are_normals_on_inc_rays_side=True,
                are_normals_on_out_rays_side=True,
            )

    return interface_dict


def make_paths(
    block_material, couplant_material, interface_dict, max_number_of_reflection=1
):
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

    probe = interface_dict["probe"]
    frontwall = interface_dict["frontwall_trans"]
    grid = interface_dict["grid"]
    if max_number_of_reflection >= 1:
        backwall = interface_dict["backwall_refl"]
    if max_number_of_reflection >= 2:
        frontwall_refl = interface_dict["frontwall_refl"]

    paths["L"] = c.Path(
        interfaces=(probe, frontwall, grid),
        materials=(couplant_material, block_material),
        modes=(c.Mode.L, c.Mode.L),
        name="L",
    )

    paths["T"] = c.Path(
        interfaces=(probe, frontwall, grid),
        materials=(couplant_material, block_material),
        modes=(c.Mode.L, c.Mode.T),
        name="T",
    )

    if max_number_of_reflection >= 1:
        paths["LL"] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.L, c.Mode.L),
            name="LL",
        )

        paths["LT"] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.L, c.Mode.T),
            name="LT",
        )

        paths["TL"] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.T, c.Mode.L),
            name="TL",
        )

        paths["TT"] = c.Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(c.Mode.L, c.Mode.T, c.Mode.T),
            name="TT",
        )

    if max_number_of_reflection >= 2:
        keys = ["LLL", "LLT", "LTL", "LTT", "TLL", "TLT", "TTL", "TTT"]

        for key in keys:
            paths[key] = c.Path(
                interfaces=(probe, frontwall, backwall, frontwall_refl, grid),
                materials=(
                    couplant_material,
                    block_material,
                    block_material,
                    block_material,
                ),
                modes=(
                    c.Mode.L,
                    helpers.parse_enum_constant(key[0], c.Mode),
                    helpers.parse_enum_constant(key[1], c.Mode),
                    helpers.parse_enum_constant(key[2], c.Mode),
                ),
                name=key,
            )

    return paths


def make_views(
    examination_object,
    probe_oriented_points,
    scatterers_oriented_points,
    max_number_of_reflection=1,
    tfm_unique_only=False,
):
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
        couplant, probe_oriented_points, frontwall, backwall, scatterers_oriented_points
    )

    paths = make_paths(block, couplant, interfaces, max_number_of_reflection)

    return make_views_from_paths(paths, tfm_unique_only)


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
