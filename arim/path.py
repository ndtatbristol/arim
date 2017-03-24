"""
Objects and helpers related to paths and interfaces.

Remark: Interface and Path objects are defined in arim.core
"""
import contextlib
import warnings

import numpy as np
import numba

from collections import OrderedDict
from .core import Path, View, Interface
from .exceptions import ArimWarning
from . import geometry as g
from . import settings as s
from .helpers import Cache, NoCache

__all__ = ['interfaces_for_block_in_immersion', 'default_orientations',
           'points_1d_wall_z', 'points_from_grid',
           'points_from_probe', 'paths_for_block_in_immersion',
           'views_for_block_in_immersion', 'IMAGING_MODES']

# Order by length then by lexicographic order
# Remark: independent views for one array (i.e. consider that view AB-CD is the
# same as view DC-BA).
IMAGING_MODES = ["L-L", "L-T", "T-T",
                 "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L", "TT-T",
                 "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                 "LT-LT", "LT-TL", "LT-TT",
                 "TL-LT", "TL-TT",
                 "TT-TT"]


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
    probe_interface: Interface
    frontwall_interface: Interface
    backwall_interface: Interface
    grid_interface: Interface
    """
    probe_interface = Interface(probe_points, probe_orientations,
                                are_normals_on_out_rays_side=True)
    frontwall_interface = Interface(frontwall_points, frontwall_orientations,
                                    'fluid_solid', 'transmission',
                                    are_normals_on_inc_rays_side=False,
                                    are_normals_on_out_rays_side=True)
    backwall_interface = Interface(backwall_points, backwall_orientations,
                                   'solid_fluid', 'reflection',
                                   reflection_against=couplant_material,
                                   are_normals_on_inc_rays_side=True,
                                   are_normals_on_out_rays_side=False)
    grid_interface = Interface(grid_points, grid_orientations,
                               are_normals_on_inc_rays_side=True)

    return probe_interface, frontwall_interface, backwall_interface, grid_interface


def paths_for_block_in_immersion(block_material, couplant_material, probe_interface,
                                 frontwall_interface, backwall_interface, grid_interface):
    """
    Creates the paths L, T, LL, LT, TL, TT (in this order).

    Paths are returned in transmit convention: for the path XY, X is the mode
    before reflection against the backwall and Y is the mode after reflection.
    The path XY in transmit convention is the path YX in receive convention.

    Parameters
    ----------
    block_material : Material
    couplant_material : Material
    probe_interface : Interface
    frontwall_interface : Interface
    backwall_interface : Interface
    grid_interface : Interface

    Returns
    -------
    paths : OrderedDict

    """
    paths = OrderedDict()

    if backwall_interface.reflection_against != couplant_material:
        warnings.warn("Different couplant materials are used.", ArimWarning)

    paths['L'] = Path(
        interfaces=(probe_interface, frontwall_interface, grid_interface),
        materials=(couplant_material, block_material),
        modes=('L', 'L'),
        name='L')

    paths['T'] = Path(
        interfaces=(probe_interface, frontwall_interface, grid_interface),
        materials=(couplant_material, block_material),
        modes=('L', 'T'),
        name='T')

    paths['LL'] = Path(
        interfaces=(
            probe_interface, frontwall_interface, backwall_interface, grid_interface),
        materials=(couplant_material, block_material, block_material),
        modes=('L', 'L', 'L'),
        name='LL')

    paths['LT'] = Path(
        interfaces=(
            probe_interface, frontwall_interface, backwall_interface, grid_interface),
        materials=(couplant_material, block_material, block_material),
        modes=('L', 'L', 'T'),
        name='LT')

    paths['TL'] = Path(
        interfaces=(
            probe_interface, frontwall_interface, backwall_interface, grid_interface),
        materials=(couplant_material, block_material, block_material),
        modes=('L', 'T', 'L'),
        name='TL')

    paths['TT'] = Path(
        interfaces=(
            probe_interface, frontwall_interface, backwall_interface, grid_interface),
        materials=(couplant_material, block_material, block_material),
        modes=('L', 'T', 'T'),
        name='TT')

    return paths


def views_for_block_in_immersion(paths_dict):
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
    views = OrderedDict()
    for view_name in IMAGING_MODES:
        tx_name, rx_name = view_name.split('-')

        tx_path = paths_dict[tx_name]
        # to get the receive path: return the string of the corresponding transmit path
        rx_path = paths_dict[rx_name[::-1]]

        views[view_name] = View(tx_path, rx_path, view_name)
    return views


class LegsCoordinates:
    """
    Stor
    """

    def __init__(self, legs_points, are_normals_on_legs_side=None):
        self.legs_points = g.aspoints(legs_points)
        self.clear_cache()
        self.are_normals_on_legs_side = are_normals_on_legs_side

    def clear_cache(self):
        self._polar = None
        self._polar2 = None
        self._azimuth = None
        self._radius = None

    @property
    def x(self):
        return self.legs_points.x

    @property
    def y(self):
        return self.legs_points.y

    @property
    def z(self):
        return self.legs_points.z

    @property
    def radius(self):
        if self._radius is None:
            self._radius = g.spherical_coordinates_r(self.x, self.y, self.z)
        return self._radius

    @property
    def polar(self):
        if self._polar is None:
            self._polar = g.spherical_coordinates_theta(self.z, self.radius)
        return self._polar

    @property
    def azimuth(self):
        if self._azimuth is None:
            self._azimuth = g.spherical_coordinates_phi(self.x, self.y)
        return self._azimuth

    @property
    def polar2(self):
        if self._polar2 is None:
            if self.are_normals_on_legs_side is None:
                raise AttributeError(
                    "attribute are_normals_on_legs_side must be defined to use polar2")
            elif self.are_normals_on_legs_side:
                self._polar2 = self.polar
            else:
                self._polar2 = np.pi / 2 - self.polar2
        return self._polar2


@numba.vectorize(['float64(float64, float64)', 'float32(float32, float32)'],
                 nopython=True, target='parallel')
def _signed_leg_angle(polar, azimuth):
    pi2 = np.pi / 2
    if -pi2 < azimuth <= polar:
        return polar
    else:
        return -polar


def _cache_ray_geometry(user_func):
    """
    Cache decorator companion for RayGeometry.

    Parameters
    ----------
    user_func

    Returns
    -------

    """

    def wrapper(self, interface_idx, is_final=True):
        """

        Parameters
        ----------
        self : RayGeometry
        interface_idx : int
        is_final : bool

        Returns
        -------

        """
        actual_interface_idx = self._interface_indices[interface_idx]
        key = '{}:{}'.format(user_func.__name__, actual_interface_idx)
        try:
            # Cache hit
            res = self._cache[key]
        except KeyError:
            pass
        else:
            if is_final:
                # Promote to final result if necessary
                self._final_keys.add(key)
            return res

        # Cache miss: compute and save result
        res = user_func(self, interface_idx=interface_idx)

        # Save in cache:
        self._cache[key] = res
        if is_final:
            self._final_keys.add(key)

        return res

    return wrapper


class RayGeometry:
    """
    RayGeometry computes the leg sizes and various angles in rays.

    RayGeometry uses an internal cache during the computation which reduces the
    computational time. The cache has two levels: intermediate results and final results.

    """

    def __init__(self, interfaces, rays, use_cache=True):
        """

        Parameters
        ----------
        interfaces : Sized of Interface
        rays : Rays
        use_cache : bool
            Default: True
        """
        numinterfaces = len(interfaces)
        self.interfaces = interfaces
        self.rays = rays

        assert rays.fermat_path.points == tuple(i.points for i in interfaces), \
            'Inconsistent rays and interfaces'

        # self.legs = [] * path.numlegs
        # self.incoming_legs = [None] + [] * path.numlegs
        # self.outgoing_legs = [] * (path.numlegs - 1) + [None]

        # Used for caching via _cache_ray_geometry wrapper
        self._use_cache = use_cache
        self._cache = Cache() if use_cache else NoCache()
        self._final_keys = set()

        self._interface_indices = tuple(range(numinterfaces))

    @classmethod
    def from_path(cls, path, use_cache=True):
        """

        Parameters
        ----------
        path : Path
        use_cache : bool
            Optional

        Returns
        -------

        """
        if path.rays is None:
            raise ValueError('Rays must be computed first.')
        return cls(path.interfaces, path.rays, use_cache=use_cache)

    @property
    def numinterfaces(self):
        return len(self.interfaces)

    @property
    def numlegs(self):
        return len(self.interfaces) - 1

    @contextlib.contextmanager
    def precompute(self):
        """
        Context manager for cleaning intermediate results after execution.

        Example
        -------

        >>> ray_geometry = RayGeometry.from_path(path)
        >>> with ray_geometry.precompute():
        ...     ray_geometry.inc_angle(1)
        ...     ray_geometry.signed_inc_angle(1)
        ... # At this stage, the two results are stored in cache and the intermadiate
        ... # results are discarded.
        >>> ray_geometry.inc_angle(1)  # fetch from cache
        """
        if not self._use_cache:
            warnings.warn("Caching is not enabled therefore precompute() will not work.",
                          ArimWarning)
        yield
        self.clear_intermediate_results()

    @_cache_ray_geometry
    def leg_points(self, interface_idx):
        """
        Returns the coordinates (x, y, z) in the GCS of the points crossed by the rays at
        a given interface.

        ``points[i, j]`` contains the (x, y, z) coordinates of the points through which
        the ray (i, j) goes at the interface of index ``interface_idx``.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        points : Points

        """
        interfaces = self.interfaces
        points = interfaces[interface_idx].points
        legs_points = points.coords.take(self.rays.indices[interface_idx], axis=0)
        return g.aspoints(legs_points)

    @_cache_ray_geometry
    def orientations_of_legs_points(self, interface_idx):
        """
        ``orientations[i, j]`` is the basis (3x3 orthogonal matrix) attached to the point
        through which the ray (i, j) goes at the interface of index ``interface_idx``.


        Parameters
        ----------
        interface_idx

        Returns
        -------
        orientations : Points

        """
        orientations_all_points = self.interfaces[interface_idx].orientations
        rays_indices = self.rays.indices[interface_idx]
        orientations = orientations_all_points.coords.take(rays_indices, axis=0)
        return g.aspoints(orientations)

    def clear_intermediate_results(self):
        """
        Clear cache containing intermediate (non-final) results.
        """
        keys_to_flush = [key for key in self._cache if key not in self._final_keys]
        for key in keys_to_flush:
            self._cache.pop(key, None)

    def clear_all_results(self):
        """
        Clear the whole cache (intermediate and final results).
        """
        self._cache = self._cache.__class__()
        self._final_keys = set()

    @property
    def inc_angles_list(self):
        """
        Legacy interface
        """
        warnings.warn('inc_angles_list is deprecated, use conventional_inc_angle() '
                      'instead', DeprecationWarning)
        return [self.conventional_inc_angle(i) for i in range(self.numinterfaces)]

    @property
    def out_angles_list(self):
        """
        Legacy interface
        """
        warnings.warn('out_angles_list is deprecated, use conventional_out_angle() '
                      'instead', DeprecationWarning)
        return [self.conventional_out_angle(i) for i in range(self.numinterfaces)]

    @_cache_ray_geometry
    def inc_leg_size(self, interface_idx):
        """
        Compute the size of the incoming leg.

        Gives the same result as
        :meth:`RayGeometry.inc_leg_radius` but does not rely on the spherical
        coordinates, therefore this method is faster when spherical coordinates are not
        required elsewhere.

        Parameters
        ----------
        interface_idx

        Returns
        -------

        """
        if interface_idx == 0:
            return None

        legs_starts = self.leg_points(interface_idx - 1, is_final=False).coords
        legs_ends = self.leg_points(interface_idx, is_final=False).coords

        legs = g.Points(legs_starts - legs_ends)
        return legs.norm2()

    @_cache_ray_geometry
    def inc_leg_cartesian(self, interface_idx):
        """
        Returns the coordinates of the source points of the legs that arrive at the
        interface of index ``interface_idx``.

        The coordinates of each leg are given in the Cartesian coordinate system attached
        to the its end point (i.e. point on interface interface_idx).

        No incoming legs towards the first interface.

        Parameters
        ----------
        interface_idx : int


        Returns
        -------
        points : Points or None
            None for the first interface.

        """
        if interface_idx == 0:
            return None

        legs_starts = self.leg_points(interface_idx - 1, is_final=False).coords
        legs_ends = self.leg_points(interface_idx, is_final=False).coords

        orientations = self.orientations_of_legs_points(interface_idx,
                                                        is_final=False).coords

        legs_local = g.from_gcs(legs_starts, orientations, legs_ends)
        return g.aspoints(legs_local)

    @_cache_ray_geometry
    def inc_leg_radius(self, interface_idx):
        """
        Returns the radius (length) of the leg incoming to interface interface_idx.

        Use spherical coordinate system attached to the end point of the leg.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        ndarray

        """
        cartesian = self.inc_leg_cartesian(interface_idx, is_final=False)
        if cartesian is None:
            return None
        return g.spherical_coordinates_r(cartesian.x, cartesian.y, cartesian.z)

    @_cache_ray_geometry
    def inc_leg_polar(self, interface_idx):
        """
        Returns the polar angle in [0, pi] of the leg incoming to interface interface_idx.

        Use spherical coordinate system attached to the end point of the leg.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        ndarray

        """
        cartesian = self.inc_leg_cartesian(interface_idx, is_final=False)
        if cartesian is None:
            return None
        radius = self.inc_leg_radius(interface_idx, is_final=False)
        return g.spherical_coordinates_theta(cartesian.z, radius)

    @_cache_ray_geometry
    def inc_leg_azimuth(self, interface_idx):
        """
        Returns the corrected polar angle in [0, pi] of the leg incoming to interface
        interface_idx. Corresponds to the polar angle if the normals of the interface
        are on the same side as the incoming legs, pi/2 minus polar angle otherwise.

        Use spherical coordinate system attached to the end point of the leg.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        ndarray

        """
        cartesian = self.inc_leg_cartesian(interface_idx, is_final=False)
        if cartesian is None:
            return None
        return g.spherical_coordinates_phi(cartesian.x, cartesian.y)

    @_cache_ray_geometry
    def inc_angle(self, interface_idx):
        return self.inc_leg_polar(interface_idx, is_final=False)

    @_cache_ray_geometry
    def signed_inc_angle(self, interface_idx):
        azimuth = self.inc_leg_azimuth(interface_idx, is_final=False)
        if azimuth is None:
            return None
        polar = self.inc_leg_polar(interface_idx, is_final=False)
        return _signed_leg_angle(polar, azimuth)

    @_cache_ray_geometry
    def conventional_inc_angle(self, interface_idx):
        if interface_idx == 0:
            return None
        are_normals_on_inc_rays_side = self.interfaces[
            interface_idx].are_normals_on_inc_rays_side
        if are_normals_on_inc_rays_side is None:
            raise ValueError('Attribute are_normals_on_inc_rays_side must be set for the'
                             'interface {}.'.format(interface_idx))
        elif are_normals_on_inc_rays_side:
            return self.inc_leg_polar(interface_idx, is_final=False)
        else:
            # out = pi - theta
            out = self.inc_leg_polar(interface_idx, is_final=False).copy()
            out[...] *= -1
            out[...] += np.pi
            return out

    @_cache_ray_geometry
    def out_leg_cartesian(self, interface_idx):
        """
        Returns the coordinates of the destination points of the legs that start from the
        interface of index ``interface_idx``.

        The coordinates of each leg are given in the Cartesian coordinate system attached
        to the its start point (i.e. point on interface interface_idx).

        No outgoing legs from the last interface.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        points : Points or None
            None for the last interface.

        """
        actual_interface_idx = self._interface_indices[interface_idx]
        if actual_interface_idx == (self.numinterfaces - 1):
            # No outgoing legs from the last interface
            return None

        legs_starts = self.leg_points(interface_idx, is_final=False).coords
        legs_ends = self.leg_points(interface_idx + 1, is_final=False).coords

        orientations = self.orientations_of_legs_points(interface_idx,
                                                        is_final=False).coords
        # Convert legs in the local coordinate systems.
        legs_local = g.from_gcs(legs_ends, orientations, legs_starts)
        return g.aspoints(legs_local)

    @_cache_ray_geometry
    def out_leg_radius(self, interface_idx):
        """
        Returns the radius (length) of the leg outgoing from interface interface_idx.

        Use spherical coordinate system attached to the end point of the leg.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        ndarray

        """
        cartesian = self.out_leg_cartesian(interface_idx, is_final=False)
        if cartesian is None:
            return None
        return g.spherical_coordinates_r(cartesian.x, cartesian.y, cartesian.z)

    @_cache_ray_geometry
    def out_leg_polar(self, interface_idx):
        """
        Returns the polar angle in [0, pi] of the leg outgoing from interface interface_idx.

        Use spherical coordinate system attached to the end point of the leg.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        ndarray

        """
        cartesian = self.out_leg_cartesian(interface_idx, is_final=False)
        if cartesian is None:
            return None
        radius = self.out_leg_radius(interface_idx, is_final=False)
        return g.spherical_coordinates_theta(cartesian.z, radius)

    @_cache_ray_geometry
    def out_leg_azimuth(self, interface_idx):
        """
        Returns the corrected polar angle in [0, pi] of the leg outgoing from interface
        interface_idx. Corresponds to the polar angle if the normals of the interface
        are on the same side as the outgoing legs, pi/2 minus polar angle otherwise.

        Use spherical coordinate system attached to the end point of the leg.

        Parameters
        ----------
        interface_idx : int

        Returns
        -------
        ndarray

        """
        cartesian = self.out_leg_cartesian(interface_idx, is_final=False)
        if cartesian is None:
            return None
        return g.spherical_coordinates_phi(cartesian.x, cartesian.y)

    @_cache_ray_geometry
    def out_angle(self, interface_idx):
        return self.out_leg_polar(interface_idx, is_final=False)

    @_cache_ray_geometry
    def signed_out_angle(self, interface_idx):
        azimuth = self.out_leg_azimuth(interface_idx, is_final=False)
        if azimuth is None:
            return None
        polar = self.out_leg_polar(interface_idx, is_final=False)
        return _signed_leg_angle(polar, azimuth)

    @_cache_ray_geometry
    def conventional_out_angle(self, interface_idx):
        actual_interface_idx = self._interface_indices[interface_idx]
        if actual_interface_idx == (self.numinterfaces - 1):
            return None
        are_normals_on_out_rays_side = self.interfaces[
            interface_idx].are_normals_on_out_rays_side
        if are_normals_on_out_rays_side is None:
            raise ValueError('Attribute are_normals_on_out_rays_side must be set for the'
                             'interface {}.'.format(interface_idx))
        elif are_normals_on_out_rays_side:
            return self.out_leg_polar(interface_idx, is_final=False)
        else:
            # out = pi - theta
            out = self.out_leg_polar(interface_idx, is_final=False).copy()
            out[...] *= -1
            out[...] += np.pi
            return out
