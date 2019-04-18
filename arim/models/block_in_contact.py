"""
Model for solid block on which the probe is in direct contact


"""
from collections import OrderedDict

from .helpers import make_views_from_paths
from .. import core as c


def make_interfaces(
    probe_oriented_points, grid_oriented_points, frontwall=None, backwall=None
):
    """
    Construct interfaces

    This function is a placeholder for the to-be-implemented forward model.

    Parameters
    ----------
    probe_oriented_points : OrientedPoints
    grid_oriented_points: OrientedPoints
    frontwall: OrientedPoints or None
    backwall: OrientedPoints or None

    Returns
    -------
    interface_dict : dict[Interface]
        Keys: probe, grid, backwall_refl (optional), frontwall_refl (optional)
    """
    interface_dict = OrderedDict()

    interface_dict["probe"] = c.Interface(
        *probe_oriented_points, are_normals_on_out_rays_side=True
    )
    interface_dict["grid"] = c.Interface(
        *grid_oriented_points, are_normals_on_inc_rays_side=True
    )
    if backwall is not None:
        interface_dict["backwall_refl"] = c.Interface(
            *backwall,
            are_normals_on_inc_rays_side=False,
            are_normals_on_out_rays_side=False,
        )
    if frontwall is not None:
        interface_dict["frontwall_refl"] = c.Interface(
            *frontwall,
            are_normals_on_inc_rays_side=True,
            are_normals_on_out_rays_side=True,
        )
    return interface_dict


def make_paths(block_material, interface_dict, max_number_of_reflection=0):
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
    if max_number_of_reflection >= 1:
        try:
            backwall = interface_dict["backwall_refl"]
        except KeyError:
            raise ValueError("Backwall must be defined to use skip paths")
    if max_number_of_reflection >= 2:
        try:
            frontwall_refl = interface_dict["frontwall_refl"]
        except KeyError:
            raise ValueError("Frontwall must be defined to use double-skip paths")

    paths["L"] = c.Path(
        interfaces=(probe, grid),
        materials=(block_material,),
        modes=(c.Mode.L,),
        name="L",
    )
    paths["T"] = c.Path(
        interfaces=(probe, grid),
        materials=(block_material,),
        modes=(c.Mode.T,),
        name="T",
    )

    mode_dict = {"L": c.Mode.longitudinal, "T": c.Mode.transverse}
    if max_number_of_reflection >= 1:
        keys = ["LL", "LT", "TL", "TT"]
        for key in keys:
            paths[key] = c.Path(
                interfaces=(probe, backwall, grid),
                materials=(block_material, block_material),
                modes=(mode_dict[key[0]], mode_dict[key[1]]),
                name=key,
            )
    if max_number_of_reflection >= 2:
        keys = ["LLL", "LLT", "LTL", "LTT", "TLL", "TLT", "TTL", "TTT"]
        for key in keys:
            paths[key] = c.Path(
                interfaces=(probe, backwall, frontwall_refl, grid),
                materials=(block_material, block_material, block_material),
                modes=(mode_dict[key[0]], mode_dict[key[1]], mode_dict[key[2]]),
                name=key,
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
        frontwall = examination_object.frontwall
    except AttributeError:
        frontwall = None
    try:
        backwall = examination_object.backwall
    except AttributeError:
        backwall = None
    interfaces = make_interfaces(
        probe_oriented_points,
        grid_oriented_points,
        frontwall=frontwall,
        backwall=backwall,
    )
    paths = make_paths(block_material, interfaces, max_number_of_reflection)
    return make_views_from_paths(paths, tfm_unique_only)
