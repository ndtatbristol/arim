"""
Objects and helpers related to paths and interfaces.

Remark: Interface and Path objects are defined in arim.core
"""
from collections import OrderedDict

import numpy as np

from . import geometry as g
from . import settings as s
from .core import Path, View, Interface, Mode
from .helpers import parse_enum_constant

__all__ = ['interfaces_for_block_in_immersion', 'default_orientations',
           'points_1d_wall_z', 'points_from_grid',
           'points_from_probe', 'paths_for_block_in_immersion',
           'views_for_block_in_immersion', 'IMAGING_MODES']

# Order by length then by lexicographic order
# Remark: independent views for one array (i.e. consider that view AB-CD is the
# same as view DC-BA).t
IMAGING_MODES = ["L-L", "L-T", "T-T",
                 "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L", "TT-T",
                 "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                 "LT-LT", "LT-TL", "LT-TT",
                 "TL-LT", "TL-TT",
                 "TT-TT"]

DIRECT_PATHS = ['L', 'T']
SKIP_PATHS = ['LL', 'LT', 'TL', 'TT']
DOUBLE_SKIP_PATHS = ['LLL', 'LLT', 'LTL', 'LTT', 'TLL', 'TLT', 'TTL', 'TTT']

L = Mode.L
T = Mode.T


def viewname_order(tx_rx_tuple):
    """
    The views are sorted in ascending order with the following criteria (in this order):

    1) the total number of legs,
    2) the maximum number of legs for transmit and receive paths,
    3) the number of legs for receive path,
    4) the number of legs for transmit path,
    5) lexicographic order for transmit path,
    6) lexicographic order for receive path.

    Parameters
    ----------
    tx_rx_tuple

    Returns
    -------
    order_tuple

    """
    tx, rx = tx_rx_tuple
    return (len(tx) + len(rx), max(len(tx), len(rx)), len(rx), len(tx), tx, rx)


def filter_unique_views(viewnames):
    """
    Remove views that would give the same result because of time reciprocity
    (under linear assumption). Order is unchanged.

    Parameters
    ----------
    viewnames : list[tuple[str]]

    Returns
    -------
    list[tuple[str]]

    Examples
    --------

    >>> filter_unique_views([('AB', 'CD'), ('DC', 'BA'), ('X', 'YZ'), ('ZY', 'X')])
    ... [('AB', 'CD'), ('X', 'YZ')]

    """
    unique_views = []
    seen_so_far = set()
    for view in viewnames:
        tx, rx = view
        rev_view = (rx[::-1], tx[::-1])
        if rev_view in seen_so_far:
            continue
        else:
            seen_so_far.add(view)
            unique_views.append(view)
    return unique_views


def make_viewnames(pathnames, unique_only=True, order_func=viewname_order):
    """
    Parameters
    ----------
    pathnames : list[str]
    unique_only : bool
        If True, consider Default True.
    order_func : func

    Returns
    -------
    list[tuple[str]

    """
    viewnames = []
    for tx in pathnames:
        for rx in pathnames:
            viewnames.append((tx, rx))

    if order_func is not None:
        viewnames = list(sorted(viewnames, key=viewname_order))

    if unique_only:
        viewnames = filter_unique_views(viewnames)

    return viewnames


def points_1d_wall_z(xmin, xmax, z, numpoints, y=0., name=None, dtype=None):
    """
    Return a wall between 1

    Returns a set of regularly spaced points between (xmin, y, z) and (xmax, y, z).

    Orientation of the point: (0., 0., 1.)

    """
    if dtype is None:
        dtype = s.FLOAT

    points = g.Points.from_xyz(
        x=np.linspace(xmin, xmax, numpoints, dtype=dtype),
        y=np.full((numpoints,), y, dtype=dtype),
        z=np.full((numpoints,), z, dtype=dtype),
        name=name,
    )

    orientations = default_orientations(points)

    return points, orientations


def default_orientations(points):
    """
    Assign to each point the following orientation::

        x = (1, 0, 0)
        y = (0, 0, 0)
        z = (0, 0, 1)

    Parameters
    ----------
    points: arim.geometry.Points

    Returns
    -------
    orientations : arim.geometry.Points

    """
    # No need to create a full array because all values are the same: we cheat
    # using a broadcast array. This saves memory space and reduces access time.
    orientations_arr = np.broadcast_to(np.identity(3, dtype=points.dtype),
                                       (*points.shape, 3, 3))
    orientations = g.Points(orientations_arr)
    return orientations


def points_from_probe(probe, name='Probe'):
    """
    Probe object to Points (centres and orientations).
    """
    points = probe.locations
    if name is not None:
        points.name = name

    orientations_arr = np.zeros((3, 3), dtype=points.dtype)
    orientations_arr[0] = probe.pcs.i_hat
    orientations_arr[1] = probe.pcs.j_hat
    orientations_arr[2] = probe.pcs.k_hat
    orientations = g.Points(np.broadcast_to(orientations_arr, (*points.shape, 3, 3)))

    return points, orientations


def points_from_grid(grid):
    """
    Grid object to Points (centres and orientations).
    """
    points = grid.as_points
    orientations = default_orientations(points)
    return points, orientations


def interfaces_for_block_in_immersion(couplant_material,
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

    interface_dict['probe'] = Interface(probe_points, probe_orientations,
                                        are_normals_on_out_rays_side=True)
    interface_dict['frontwall_trans'] = Interface(frontwall_points,
                                                  frontwall_orientations,
                                                  'fluid_solid', 'transmission',
                                                  are_normals_on_inc_rays_side=False,
                                                  are_normals_on_out_rays_side=True)
    interface_dict['backwall_refl'] = Interface(backwall_points, backwall_orientations,
                                                'solid_fluid', 'reflection',
                                                reflection_against=couplant_material,
                                                are_normals_on_inc_rays_side=False,
                                                are_normals_on_out_rays_side=False)
    interface_dict['grid'] = Interface(grid_points, grid_orientations,
                                       are_normals_on_inc_rays_side=True)
    interface_dict['frontwall_refl'] = Interface(frontwall_points, frontwall_orientations,
                                                 'solid_fluid', 'reflection',
                                                 reflection_against=couplant_material,
                                                 are_normals_on_inc_rays_side=True,
                                                 are_normals_on_out_rays_side=True)

    return interface_dict


def paths_for_block_in_immersion(block_material, couplant_material, interface_dict,
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

    paths['L'] = Path(
        interfaces=(probe, frontwall, grid),
        materials=(couplant_material, block_material),
        modes=(L, L),
        name='L')

    paths['T'] = Path(
        interfaces=(probe, frontwall, grid),
        materials=(couplant_material, block_material),
        modes=(L, T),
        name='T')

    if max_number_of_reflection >= 1:
        paths['LL'] = Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(L, L, L),
            name='LL')

        paths['LT'] = Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(L, L, T),
            name='LT')

        paths['TL'] = Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(L, T, L),
            name='TL')

        paths['TT'] = Path(
            interfaces=(probe, frontwall, backwall, grid),
            materials=(couplant_material, block_material, block_material),
            modes=(L, T, T),
            name='TT')

    if max_number_of_reflection >= 2:
        keys = ['LLL', 'LLT', 'LTL', 'LTT', 'TLL', 'TLT', 'TTL', 'TTT']

        for key in keys:
            paths[key] = Path(
                interfaces=(probe, frontwall, backwall, frontwall_refl, grid),
                materials=(couplant_material, block_material, block_material,
                           block_material),
                modes=(L,
                       parse_enum_constant(key[0], Mode),
                       parse_enum_constant(key[1], Mode),
                       parse_enum_constant(key[2], Mode)),
                name=key)

    return paths


def views_for_block_in_immersion(paths_dict, unique_only=True):
    """
    Returns a list of views for the case of a block in immersion.

    Returns the 21 first independent modes.

    A common way to create the paths it to use ``paths_for_block_in_immersion``.

    Remark on the nomenclature: XX'-YY' means the XX' is the transmit path and YY' is the receive path.


    Parameters
    ----------
    paths_dict : Dict[Path]
        Must have the keys: L, T, LL, LT, TL, TT (in transmit convention).

    Returns
    -------
    views: OrderedDict[Views]

    """
    viewnames = make_viewnames(paths_dict.keys(), unique_only=unique_only)
    views = OrderedDict()
    for view_name_tuple in viewnames:
        tx_name, rx_name = view_name_tuple
        view_name = '{}-{}'.format(tx_name, rx_name)

        tx_path = paths_dict[tx_name]
        # to get the receive path: return the string of the corresponding transmit path
        rx_path = paths_dict[rx_name[::-1]]

        views[view_name] = View(tx_path, rx_path, view_name)
    return views


