"""
Ray tracing module

"""
import contextlib
import gc
import logging
import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

from . import geometry as g
from . import settings as s
from .exceptions import ArimWarning, InvalidDimension
from .helpers import Cache, NoCache, chunk_array

use_parallel = os.environ.get("ARIM_USE_PARALLEL", not numba.core.config.IS_32BITS)


def find_minimum_times(
    time_1, time_2, dtype=None, dtype_indices=None, block_size=None, numthreads=None
):
    """
    For i=1:n and j=1:p,

        out_min_times(i, j)   := min_{k=1:m}    time_1[i, k] + time_2[k, j]
        out_min_indices(i, j) := argmin_{k=1:m} time_1[i, k] + time_2[k, j]


    Parameters
    ----------
    time_1
        Shape: (n, m)
    time_2
        Shape: (m, p)
    dtype
    dtype_indices

    Returns
    -------
    out_min_times
        Shape: (n, p)
    out_min_indices
        Shape: (n, p)

    Notes
    -----
    Memory usage:
    - duplicate 'time_1' if it not in C-order.
    - duplicate 'time_2' if it not in Fortran-order.

    """
    assert time_1.ndim == 2
    assert time_2.ndim == 2
    try:
        n, m = time_1.shape
        m_, p = time_2.shape
    except ValueError:
        raise InvalidDimension("time_1 and time_2 must be 2d.")

    if m != m_:
        raise ValueError("Array shapes must be (n, m) and (m, p).")

    if dtype is None:
        dtype = np.result_type(time_1, time_2)
    if dtype_indices is None:
        dtype_indices = s.INT

    if block_size is None:
        block_size = s.BLOCK_SIZE_FIND_MIN_TIMES
    if numthreads is None:
        numthreads = s.NUMTHREADS

    out_min_times = np.full((n, p), np.inf, dtype=dtype)
    out_best_indices = np.full((n, p), -1, dtype=dtype_indices)

    # time_1 will be scanned row per row, time_2 column per column.
    # Force to use the most efficient order (~20 times speed-up between the best and worst case).
    time_1 = np.ascontiguousarray(time_1)
    time_2 = np.asfortranarray(time_2)

    # Chunk time_1 and time_2 such as each chunk contains roughly 'block_size'
    # floats. Chunks for 'time_1' are lines (only complete lines), chunks
    # for 'time_2' are columns (only complete columns).
    block_size_adj = math.ceil(block_size / m)

    futures = []
    with ThreadPoolExecutor(max_workers=numthreads) as executor:
        for chunk1 in chunk_array((n, m), block_size_adj, axis=0):
            for chunk2 in chunk_array((m, p), block_size_adj, axis=1):
                chunk_res = (chunk1[0], chunk2[1])

                futures.append(
                    executor.submit(
                        _find_minimum_times,
                        time_1[chunk1],
                        time_2[chunk2],
                        out_min_times[chunk_res],
                        out_best_indices[chunk_res],
                    )
                )
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()

    return out_min_times, out_best_indices


@numba.jit(nopython=True, nogil=True, cache=True)
def _find_minimum_times(time_1, time_2, out_min_times, out_best_indices):
    """
    Parameters
    ----------
    time_1
    time_2
    out_min_time
    out_best_indices

    Returns
    -------

    """
    n, m = time_1.shape
    m, p = time_2.shape
    for i in range(n):
        for j in range(p):
            for k in range(m):
                new_time = time_1[i, k] + time_2[k, j]
                if new_time < out_min_times[i, j]:
                    out_min_times[i, j] = new_time
                    out_best_indices[i, j] = k


logger = logging.getLogger(__name__)


def ray_tracing_for_paths(
        paths_list, walls=None, turn_off_invalid_rays=False, convert_to_fortran_order=False
    ):
    """
    Perform the ray tracing for different paths. Save the result in ``Path.rays``.

    Parameters
    ----------
    paths_list : List[Path]
    convert_to_fortran_order

    Returns
    -------
    None

    """
    paths_list = tuple(paths_list)
    fermat_paths_tuple = tuple(FermatPath.from_path(path) for path in paths_list)

    fermat_solver = FermatSolver(fermat_paths_tuple)
    rays_dict = fermat_solver.solve()

    for path, fermat_path in zip(paths_list, fermat_paths_tuple):
        rays = rays_dict[fermat_path]
        suspicious_rays = rays.gone_through_extreme_points()
        num_suspicious_rays = suspicious_rays.sum()
        if num_suspicious_rays > 0:
            logger.warning(
                f"{num_suspicious_rays} rays of path {path.name} go through "
                "the interface limits. Extend limits."
            )
        if turn_off_invalid_rays:
            rays_dict[fermat_path]._invalid_rays = rays.gone_through_interface(path.interfaces, walls)

    if convert_to_fortran_order:
        old_rays_dict = rays_dict
        rays_dict = {k: v.to_fortran_order() for k, v in old_rays_dict.items()}

    # Save results in attribute path.rays:
    for path, fermat_path in zip(paths_list, fermat_paths_tuple):
        path.rays = rays_dict[fermat_path]


def ray_tracing(
        views_list, walls=None, turn_off_invalid_rays=False, convert_to_fortran_order=False
    ):
    """
    Perform the ray tracing for different views. Save the result in ``Path.rays``.

    Parameters
    ----------
    views : List[View]

    Returns
    -------
    None

    """
    # Ray tracing:
    paths_set = set(v.tx_path for v in views_list) | set(v.rx_path for v in views_list)
    return ray_tracing_for_paths(
        list(paths_set),
        walls=walls,
        turn_off_invalid_rays=turn_off_invalid_rays,
        convert_to_fortran_order=convert_to_fortran_order,
    )


@numba.jit(nopython=True, nogil=True, parallel=use_parallel)
def _expand_rays(interior_indices, indices_new_interface, expanded_indices):
    """
    Expand the rays by one interface knowing the beginning of the rays and the
    points the rays must go through at the last interface.

    A0, A1, ..., A(d+1) are (d+2) interfaces.

    n: number of points of interface A0
    m: number of points of interface Ad
    p: number of points of interface A(d+1)

    Arrays layout must be contiguous.

    Output: out_ray

    Parameters
    ----------
    interior_indices: *interior* indices of rays going from A(0) to A(d).
        Shape: (d, n, m)
    indices_new_interface: indices of the points of interface A(d) that the rays
    starting from A(0) cross to go to A(d+1).
        Shape: (n, p)
    expanded_indices: OUTPUT
        Shape (d+1, n, p)

    """
    d, n, m = interior_indices.shape
    _, p = indices_new_interface.shape

    for i in numba.prange(n):
        for j in range(p):
            # get the point on interface A(d) to which the ray goes
            idx = indices_new_interface[i, j]

            # copy the head of ray
            for k in range(d):
                expanded_indices[k, i, j] = interior_indices[k, i, idx]

            # add the last point
            expanded_indices[d, i, j] = idx


class Rays:
    """
    Rays(times, interior_indices, path)

    Store the rays between the first and last sets of points along
    a specific path.

    - n: number of points of the first set of points.
    - m: number of points of the last set of points.
    - d: number of interfaces along the path.

    We name A(1), A(2), ..., A(d) the d interfaces along the path.
    A ray passes  A(1), A(2), ..., A(d) in this order.

    The ray (i, j) is defined as the ray starting in `A(1)[i]`` and
    arriving in ``A(d)[j]``.


    Parameters
    ----------
    times : ndarray of floats [n x m]
        Shortest time between first and last set of points.
        ``times[i, j]`` is the total travel time for the ray (i, j).
    indices_interior : ndarray of floats [(d-2) x n x m]
        Indices of points through which each ray goes, excluding the first and last interfaces.
        ``indices[k-1, i, j]`` is the indice point of the ``k`` *interior* interface through which
        the ray (i,j) goes.

    fermat_path : FermatPath
        Sets of points crossed by the rays.

    order : None, 'C' or 'F'
        Force the order if the indices

    Attributes
    ----------
    times
    indices_interior
    fermat_path
    indices : ndarray of floats [d x n x m]
        Indices of points through which each ray goes.
        For k=0:p, a ray starting from ``A(1)[i]`` and ending in ``A(d)[i]``
        goes through the k-th interface at the point indexed by ``indices[k, i, j]``.
        By definition, ``indices[0, i, j] := i`` and ``indices[d-1, i, j] := j``
        for all i and j.
    """

    # __slots__ = []

    def __init__(self, times, interior_indices, fermat_path, invalid_rays=None, order=None):
        assert times.ndim == 2
        assert interior_indices.ndim == 3
        assert (
            times.shape
            == interior_indices.shape[1:]
            == (len(fermat_path.points[0]), len(fermat_path.points[-1]))
        )
        assert fermat_path.num_points_sets == interior_indices.shape[0] + 2

        assert interior_indices.dtype.kind == "i"
        assert times.dtype.kind == "f"

        indices = self.make_indices(interior_indices, order=order)

        self._times = times
        self._indices = indices
        self._fermat_path = fermat_path
        self._invalid_rays = invalid_rays

    @classmethod
    def make_rays_two_interfaces(cls, times, path, dtype_indices):
        """
        Alternative constructor for Rays objects when there is only two interfaces,
        i.e. no interior interface.
        """
        if path.num_points_sets != 2:
            raise ValueError(
                "This constructor works only for path with two interfaces. Use __init__ instead."
            )
        n = len(path.points[0])
        m = len(path.points[1])

        interior_indices = np.zeros((0, n, m), dtype=dtype_indices)
        return cls(times, interior_indices, path)

    @property
    def fermat_path(self):
        return self._fermat_path

    @property
    def times(self):
        return self._times

    @property
    def indices(self):
        return self._indices

    @property
    def interior_indices(self):
        return self.indices[1:-1, ...]

    @staticmethod
    def make_indices(interior_indices, order=None):
        """
        Parameters
        ----------
        interior_indices : ndarray
            Shape (d, n, m)

        Returns
        -------
        indices : ndarray
            Shape (n, m, d+2) such as:
            - indices[0, i, j] := i for all i, j
            - indices[-1, i, j] := j for all i, j
            - indices[k, i, j] := interior_indices[i, j, k+1] for all i, j and for k=1:(d-1)

        """
        dm2, n, m = interior_indices.shape

        if order is None:
            if interior_indices.flags.c_contiguous:
                order = "C"
            elif interior_indices.flags.fortran:
                order = "F"
            else:
                order = "C"

        indices = np.zeros((dm2 + 2, n, m), dtype=interior_indices.dtype, order=order)

        indices[0, ...] = np.repeat(np.arange(n), m).reshape((n, m))
        indices[-1, ...] = np.tile(np.arange(m), n).reshape((n, m))
        indices[1:-1, ...] = interior_indices
        return indices

    def get_coordinates(self, n_interface=None):
        """
        Yields the coordinates of the rays of the n-th interface, as a tuple
        of three 2d ndarrays.

        Use numpy fancy indexing.

        Example
        -------
        ::

            for (d, (x, y, z)) in enumerate(rays.get_coordinates()):
                # Coordinates at the d-th interface of the ray between ray A(1)[i] and
                # ray_A(d)[j].
                x[i, j]
                y[i, j]
                z[i, j]


        """
        if n_interface is None:
            for n_interface in range(self.fermat_path.num_points_sets):
                points = self.fermat_path.points[n_interface]
                indices = self.indices[n_interface, ...]
                x = points.x[indices]
                y = points.y[indices]
                z = points.z[indices]
                yield (x, y, z)
        else:
            points = self.fermat_path.points[n_interface]
            indices = self.indices[n_interface, ...]
            x = points.x[indices]
            y = points.y[indices]
            z = points.z[indices]
            yield (x, y, z)

    def get_coordinates_one(self, start_index, end_index):
        """
        Return the coordinates of one ray as ``Point``.

        This function is slow: use ``get_coordinates`` or a variant for treating
        a larger number of rays.
        """
        indices = self.indices[:, start_index, end_index]
        num_points_sets = self.fermat_path.num_points_sets
        x = np.zeros(num_points_sets, s.FLOAT)
        y = np.zeros(num_points_sets, s.FLOAT)
        z = np.zeros(num_points_sets, s.FLOAT)
        for i, (points, j) in enumerate(zip(self.fermat_path.points, indices)):
            x[i] = points.x[j]
            y[i] = points.y[j]
            z[i] = points.z[j]
        return g.Points.from_xyz(x, y, z, "Ray")

    def gone_through_extreme_points(self):
        """
        Returns the rays which are going through at least one extreme point in the interfaces.
        These rays can be non physical, it is then safer to be conservative and remove them all.

        Extreme points are the first/last points (in indices) in the interfaces, except the first and
        last interfaces (respectively the points1 and the grid).

        Returns
        -------
        out : ndarray of bool
            ``rays[i, j]`` is True if the rays starting from the i-th point of the first interface
            and going to the j-th point of the last interface is going through at least one extreme point
            through the middle interfaces.
            Order: same as attribute ``indices``.

        """
        order = "F" if self.indices.flags.f_contiguous else "C"

        shape = self.indices.shape[1:]
        out = np.zeros(shape, order=order, dtype=bool)

        interior_indices = self.interior_indices
        middle_points = tuple(self.fermat_path.points)[1:-1]
        for d, points in enumerate(middle_points):
            np.logical_or(out, interior_indices[d, ...] == 0, out=out)
            np.logical_or(out, interior_indices[d, ...] == (len(points) - 1), out=out)
        return out
    
    def gone_through_interface(self, path_interfaces, walls):
        """
        Returns the rays which are going through at least one interface (i.e. in the line).
        Considering a convex geometry, these are rays which are physically blocked by the interface.

        Parameters
        ----------
        path_interfaces : list[Interface]
            .
        walls : list[OrientedPoints]
            All of the walls in the geometry, passed in from ``examination_object.walls``.

        Returns
        -------
        out : ndarray[bool]
            ``rays[i, j]`` is `True` if the ray starting from the i-th point of the first interface
            and going through the j-th point on the last interface is going through at least one
            other interface in the middle.

        """
        order= "F" if self.indices.flags.f_contiguous else "C"
        
        shape = self.indices.shape[1:]
        out = np.zeros(shape, order=order, dtype=bool)
        
        if walls is not None:
            walls = {
                wall.name : wall for wall, _ in walls
            }
            interface_names = [
                interface.points.name for interface in path_interfaces
            ]
            
            loc_wrt_line = lambda a1, a2, y: (
                (a2[0] - a1[0]) * (y[2] - a1[2]) - (y[0] - a1[0]) * (a2[2] - a1[2])
            )
            # Initialise but defined at end of first loop.
            last = None
            for d, coords in enumerate(self.get_coordinates()):
                coords = np.stack(coords)
                if d != 0:
                    for name, wall in walls.items():
                        if (name in interface_names) or (name.lower() == "frontwall" and "Probe" in interface_names):
                            continue
                        # Check if bounding boxes intersect.
                        is_overlap = (
                            (np.min((last[0, :, :], coords[0, :, :]), axis=0) <= np.max(wall[:, 0]))
                            & (np.max((last[0, :, :], coords[0, :, :]), axis=0) >= np.min(wall[:, 0]))
                            & (np.min((last[1, :, :], coords[1, :, :]), axis=0) <= np.max(wall[:, 1]))
                            & (np.max((last[1, :, :], coords[1, :, :]), axis=0) >= np.min(wall[:, 1]))
                            & (np.min((last[2, :, :], coords[2, :, :]), axis=0) <= np.max(wall[:, 2]))
                            & (np.max((last[2, :, :], coords[2, :, :]), axis=0) >= np.min(wall[:, 2]))
                        )
                        # If any bboxes are overlapping, check the rest.
                        if is_overlap.any():
                            # Prep to check if segments overlap
                            for segment_start, segment_end in zip(wall[:-1], wall[1:]):
                                whereis_last_wrt_wall   = loc_wrt_line(segment_start, segment_end, last)
                                whereis_coords_wrt_wall = loc_wrt_line(segment_start, segment_end, coords)
                                whereis_wallmin_wrt_ray = loc_wrt_line(last, coords, segment_start)
                                whereis_wallmax_wrt_ray = loc_wrt_line(last, coords, segment_end)
                                
                                ray_touches_or_crosses_wall = ((
                                    (whereis_last_wrt_wall < 0) ^ (whereis_coords_wrt_wall < 0)
                                ) | (
                                    (np.abs(whereis_last_wrt_wall) < np.finfo(float).eps) | (np.abs(whereis_coords_wrt_wall) < np.finfo(float).eps)
                                ))
                                wall_touches_or_crosses_ray = ((
                                    (whereis_wallmin_wrt_ray < 0) ^ (whereis_wallmax_wrt_ray < 0)
                                ) | (
                                    (np.abs(whereis_wallmin_wrt_ray) < np.finfo(float).eps) | (np.abs(whereis_wallmax_wrt_ray) < np.finfo(float).eps)
                                ))
                                
                                is_intersection = is_overlap & ray_touches_or_crosses_wall & wall_touches_or_crosses_ray
                                np.logical_or(out, is_intersection, out=out)
                        # If no overlaps at all, return early as the above is quite slow.
                        else:
                            np.logical_or(out, is_overlap, out=out)
                            
                        
                last = coords
        return out

    def to_fortran_order(self):
        """
        Returns a Ray object with the .

        TFM objects except to have lookup times indexed by (grid_idx, probe_idx) whereas
        the arrays in this object are indexed by (probe_idx, grid_idx). By converting
        them to Fortran array, their transpose are C-contiguous and indexed as expected
        by TFM objects.

        Returns
        -------
        Rays

        """
        return self.__class__(
            np.asfortranarray(self.times),
            np.asfortranarray(self.interior_indices),
            self.fermat_path,
            self._invalid_rays,
            "F",
        )

    @staticmethod
    def expand_rays(interior_indices, indices_new_interface):
        """
        Expand the rays by one interface knowing the beginning of the rays and the
        points the rays must go through at the last interface.

        A0, A1, ..., A(d+1) are (d+2) interfaces.

        n: number of points of interface A0
        m: number of points of interface Ad
        p: number of points of interface A(d+1)

        For more information on ``interior_indices``, see the documentation of ``Rays``.

        Parameters
        ----------
        interior_indices: *interior* indices of rays going from A(0) to A(d).
            Shape: (d, n, m)
        indices_new_interface: indices of the points of interface A(d) that the rays
        starting from A(0) cross to go to A(d+1).
            Shape: (n, p)

        Returns
        -------
        expanded_indices
            Shape (d+1, n, p)
        """
        d, n, m = interior_indices.shape
        n_, p = indices_new_interface.shape
        if n != n_:
            raise ValueError("Inconsistent shapes")
        if d == 0:
            new_shape = (1, *indices_new_interface.shape)
            return indices_new_interface.reshape(new_shape)
        else:
            expanded_indices = np.empty((d + 1, n, p), dtype=interior_indices.dtype)
            _expand_rays(interior_indices, indices_new_interface, expanded_indices)
            return expanded_indices

    def reverse(self, order="f"):
        """
        Returns a new Rays object which corresponds to the reversed path.

        Parameters
        ----------
        order : str
            Order of the arrays 'times' and 'indices'. Default: 'f'

        Returns
        -------
        reversed_rays : Rays

        """
        reversed_times = np.asarray(self.times.T, order=order)

        # Input x of shape (d, n, m)
        # Output y of shape(d, m, n) such as ``x[k, i, j] == y[d - k, j, i]``
        reversed_indices = np.swapaxes(self.interior_indices, 1, 2)
        reversed_indices = reversed_indices[::-1, ...]
        reversed_indices = np.asarray(reversed_indices, order=order)

        reversed_path = self.fermat_path.reverse()
        return self.__class__(reversed_times, reversed_indices, reversed_path)


class FermatPath(tuple):
    """
    FermatPath(points_and_speeds)

    This object contain the interface points through which the pass during the propagation and the speeds
    between the consecutive interfaces.

    This object should be used only for the internal plumbing of FermatSolver. This object can be obtained from a
    (smarter) :class:`Path` object via the class method :meth:`FermatPath.from_path`.

    A FermatPath must starts and ends with Points objects. Speeds (stored as float) and Points must alternate.

    Ex: FermatPath((points_1, speed_1_2, points_2, speed_2_3, points_3))

    """

    def __new__(cls, sequence):
        if len(sequence) % 2 == 0 or len(sequence) < 3:
            raise ValueError(
                f"{cls.__name__} expects a sequence of length odd and >= 5)"
            )
        assert all(np.isfinite(sequence[1::2])), "nonfinite velocity"
        return super().__new__(cls, sequence)

    @classmethod
    def from_path(cls, path):
        """
        Create a FermatPath object from a (smarter) Path object.
        """
        path_pieces = []
        for interface, material, mode in zip(
            path.interfaces, path.materials, path.modes
        ):
            velocity = material.velocity(mode)
            path_pieces.append(interface.points)
            path_pieces.append(velocity)
        path_pieces.append(path.interfaces[-1].points)
        return cls(path_pieces)

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, ", ".join([str(x) for x in self])
        )

    def __add__(self, tail):
        if self[-1] != tail[0]:
            raise ValueError("Cannot join two subpaths with different extremities.")
        return self.__class__((*self, *tail[1:]))

    def reverse(self):
        return self.__class__(tuple(reversed(self)))

    def split_head(self):
        """
        Split a Path in two at the first interface:
        ``(points_1, speed_1_2, points_2)`` and ``(points_2, speed_2_3, ..., points_n)``.
        """
        if len(self) < 5:
            raise ValueError("Not enough elements to split (min: 5)")
        head = self.__class__(self[:3])
        tail = self.__class__(self[2:])
        return head, tail

    def split_queue(self):
        """
        Split a Path in two at the last interface:
        ``(points_1, speed_1_2, ... points_n1)`` and ``(points_n1, speed, points_n)``.
        """

        if len(self) < 5:
            raise ValueError("Not enough elements to split (min: 5)")
        head = self.__class__(self[:-2])
        tail = self.__class__(self[-3:])
        return head, tail

    @property
    def points(self):
        """
        Returns all the Points objects in Path as a tuple.
        """
        return tuple(self[0::2])

    @property
    def velocities(self):
        return tuple(self[1::2])

    @property
    def num_points_sets(self):
        return len(self) // 2 + 1

    @property
    def len_largest_interface(self):
        """
        Excluse first and last dataset
        """
        all_points = tuple(self.points)
        interfaces = all_points[1:-1]
        if not interfaces:
            return 0
        else:
            return max([len(x) for x in interfaces])


class FermatSolver:
    """
    Solver: take as input the interfaces, give as output the ray paths.

    General usage: instantiate object, then call method ``solve`` (or ``solve_no_clean``
    to keep intermediary results). Results are stored in attributes ``res``.

    Parameters
    ----------
    paths : set of FermatPath
        Paths which will be solved. Solving several paths at a time allows an efficient caching.
    dtype : numpy.dtype
        Datatype for times and distances. Optional, default: settings.FLOAT
    dtype_indices : numpy.dtype
        Datatype for indices. Optional, default: use the smallest unsigned
        integers that fits.

    Attributes
    ----------
    res : dictionary
        Rays stored as ``Rays`` objects, indexed by the ``paths``.
    paths
        Cf. above.
    dtype
        Cf. above.
    dtype_indices
        Cf. above.
    cached_distance : dict
        Keys: tuple of Points (points1, points2). Values: euclidean
        distance between all points of 'points1' and all points of 'points2'.
    cached_result : dict
        Keys: Path. Values: _FermatSolverResult


    """

    def __init__(self, fermat_paths_set, dtype=None, dtype_indices=None):
        if dtype is None:
            dtype = s.FLOAT

        if dtype_indices is None:
            dtype_indices = s.INT

        for path in fermat_paths_set:
            try:
                hash(path)
            except TypeError as e:
                raise TypeError("Path must be hashable.") from e

        self.dtype = dtype
        self.dtype_indices = dtype_indices
        self.clear_cache()
        self.res = {}
        self.paths = fermat_paths_set

        self.num_minimization = 0
        self.num_euc_distance = 0

    @classmethod
    def from_views(cls, views_list, dtype=None, dtype_indices=None):
        """
        Create a FermatSolver from a list of views (alternative constructor).

        Parameters
        ----------
        views : list of Views
        dtype : numpy.dtype or None
        dtype_indices : numpy.dtype or None

        Returns
        -------

        """
        paths = set(
            path
            for v in views_list
            for path in (v.tx_path.to_fermat_path(), v.rx_path.to_fermat_path())
        )
        return cls(paths, dtype=dtype, dtype_indices=dtype_indices)

    def solve(self):
        """
        Compute the rays for all paths and store them in ``self.res``.
        """

        self.solve_no_clean()
        self.clear_cache()
        return self.res

    def solve_no_clean(self):
        """
        Compute the rays for all paths and store them in ``self.res``.
        """
        tic = time.perf_counter()
        for path in self.paths:
            self.res[path] = self._solve(path)
        toc = time.perf_counter()
        logger.info(f"Ray tracing: solved all in {toc - tic:.3g}s")
        return self.res

    def _solve(self, path):
        """
        Returns the rays starting from the first interface and last interface of ``path``.

        This function is recursive. Intermediate results are stored
        in self.cached_result and self.cached_distance.

        Warning: it is not safe to call this with a Path not passed to __init__
        because of possible overflows.



        Returns
        -------
        res : Rays

        """
        if path in self.cached_result:
            # Cache hits, hourray
            return self.cached_result[path]

        # Special case if we have only two (consecutive) boundaries:
        if len(path) == 3:
            return self.consecutive_times(path)

        # General case: compute by calling _solve() recursively:
        head, tail = path.split_queue()

        res_head = self._solve(head)
        res_tail = self._solve(tail)
        assert isinstance(res_head, Rays)
        assert isinstance(res_tail, Rays)

        self.num_minimization += 1
        logger.debug(f"Ray tracing: solve for subpaths {str(head)} and {str(tail)}")
        times, indices_at_interface = find_minimum_times(
            res_head.times,
            res_tail.times,
            dtype=self.dtype,
            dtype_indices=self.dtype_indices,
        )

        assert res_tail.fermat_path.num_points_sets == 2
        indices = Rays.expand_rays(res_head.interior_indices, indices_at_interface)

        del indices_at_interface  # no more useful

        res = Rays(times, indices, path)
        self.cached_result[path] = res
        return res

    def clear_cache(self):
        self.cached_distance = Cache()
        self.cached_result = Cache()
        gc.collect()  # force the garbage collector to delete unreferenced objects

    def consecutive_times(self, path):
        """Computes the rays between two consecutive sets of points.
        This is straight forward: each ray is a straight line; ray lengths are
        obtained by taking the Euclidean distances between points.

        Cache the distance array in the two directions: points1 to points2,
        points 2 to points1.

        Returns a ``Rays`` object.
        """
        points1, speed, points2 = path

        key = (points1, points2)

        try:
            distance = self.cached_distance[key]
        except KeyError:
            self.num_euc_distance += 1
            distance = g.distance_pairwise(points1, points2, dtype=self.dtype)
            rkey = (points1, points2)
            self.cached_distance[key] = distance
            if key != rkey:  # if points1 and points2 are the same!
                self.cached_distance[rkey] = distance.T
        return Rays.make_rays_two_interfaces(distance / speed, path, self.dtype_indices)


@numba.vectorize(
    ["float64(float64, float64)", "float32(float32, float32)"],
    nopython=True,
    target="parallel",
)
def _signed_leg_angle(polar, azimuth):
    pi2 = np.pi / 2
    if -pi2 < azimuth <= polar:
        return polar
    else:
        return -polar


def _to_readonly(x):
    if isinstance(x, np.ndarray):
        x.flags.writeable = False
        return
    elif isinstance(x, g.Points):
        x.coords.flags.writeable = False
        return
    elif x is None:
        return
    else:
        try:
            xlist = list(x)
        except TypeError:
            raise TypeError(f"unhandled type: {type(x)}")
        else:
            for x in xlist:
                _to_readonly(x)


def _cache_ray_geometry(user_func):
    """
    Cache decorator companion for RayGeometry.

    Numpy arrays are set to read-only to avoid mistakes.

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
        key = f"{user_func.__name__}:{actual_interface_idx}"
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
        _to_readonly(res)
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

        assert rays.fermat_path.points == tuple(
            i.points for i in interfaces
        ), "Inconsistent rays and interfaces"

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
            raise ValueError("Rays must be computed first.")
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
            warnings.warn(
                "Caching is not enabled therefore precompute() will not work.",
                ArimWarning,
                stacklevel=2,
            )
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

        orientations = self.orientations_of_legs_points(
            interface_idx, is_final=False
        ).coords

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
            interface_idx
        ].are_normals_on_inc_rays_side
        if are_normals_on_inc_rays_side is None:
            raise ValueError(
                "Attribute are_normals_on_inc_rays_side must be set for the"
                f"interface {interface_idx}."
            )
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

        orientations = self.orientations_of_legs_points(
            interface_idx, is_final=False
        ).coords
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
            interface_idx
        ].are_normals_on_out_rays_side
        if are_normals_on_out_rays_side is None:
            raise ValueError(
                "Attribute are_normals_on_out_rays_side must be set for the"
                f"interface {interface_idx}."
            )
        elif are_normals_on_out_rays_side:
            return self.out_leg_polar(interface_idx, is_final=False)
        else:
            # out = pi - theta
            out = self.out_leg_polar(interface_idx, is_final=False).copy()
            out[...] *= -1
            out[...] += np.pi
            return out
