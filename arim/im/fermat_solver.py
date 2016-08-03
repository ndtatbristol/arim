"""
Module for computation of shortest ray paths accross several interfaces.
Notably used for multi-view TFM.

To improve:
    - pitch-catch (multiple arrays) imaging?


"""

import gc
import logging
import math
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import geometry as g
from ..core.cache import Cache
from .base import find_minimum_times
from .. import settings as s
from ..utils import chunk_array, smallest_uint_that_fits
from ._fermat_solver import _expand_rays

__all__ = ['FermatSolver', 'View', 'Path', 'Rays']

logger = logging.getLogger(__name__)


class Rays:
    """
    Rays(times, interior_indices, path)

    Store the rays between the first and last datasets of points along
    a specific path.

    - n: number of points of the first set of points.
    - m: number of points of the last set of points.
    - d: number of interfaces along the path.

    We name A(1), A(2), ..., A(d) the d interfaces along the path.
    A ray passes  A(1), A(2), ..., A(d) in this order.


    Parameters
    ----------
    times : ndarray of floats [n x m]
        Shortest time between first and last set of points.
        ``times[i, j]`` is the minimal time between the i-th point of the first
        interface and the j-th point of the last interface. In other words,
        ``times[i, j]`` is the minimal time between ``A(1)[i]`` and
        ``A(d)[j]``.
    indices_interior : ndarray of floats [(d-2) x n x m]
        Indices of points for each ray, excluding the first and last interfaces.

        For k=1:(p-1), a ray starting from ``A(1)[i]`` and ending in
        ``A(d)[i]`` passes through the k-th interface at the point indexed
        by ``indices[k-1, i, j]``.
    path : Path
        Sets of points crossed by the rays.

    Attributes
    ----------
    indices : ndarray of floats [d x n x m]
        Indices of points for each ray.
        For k=0:p, a ray starting from ``A(1)[i]`` and ending in ``A(d)[i]``
        passes through the k-th interface at the point indexed by ``indices[k, i, j]``.
        By definition, ``indices[0, i, j] := i`` and ``indices[d-1, i, j] := j``
        for all i and j.

    """

    # __slots__ = []

    def __init__(self, times, interior_indices, path):
        assert times.ndim == 2
        assert interior_indices.ndim == 3
        assert times.shape == interior_indices.shape[1:] == (len(path.points[0]), len(path.points[-1]))
        assert path.num_points_sets == interior_indices.shape[0] + 2

        if interior_indices.dtype.kind != 'u':
            raise TypeError("Indices must be unsigned integers.")
        assert times.dtype.kind == 'f'

        indices = self.make_indices(interior_indices)

        self._times = times
        self._indices = indices
        self._path = path

    @classmethod
    def make_rays_two_interfaces(cls, times, path, dtype_indices):
        """
        Alternative constructor for Rays objects when there is only two interfaces,
        i.e. no interior interface.
        """
        if path.num_points_sets != 2:
            raise ValueError(
                "This constructor works only for path with two interfaces. Use __init__ instead.")
        n = len(path.points[0])
        m = len(path.points[1])

        interior_indices = np.zeros((0, n, m), dtype=dtype_indices)
        return cls(times, interior_indices, path)

    @property
    def path(self):
        return self._path

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
    def make_indices(interior_indices):
        """
        Parameters
        ----------
        interior_indices : shape (d, n, m)

        Returns
        -------
        indices : shape (n, m, d+2) such as:
            - indices[0, i, j] := i for all i, j
            - indices[-1, i, j] := j for all i, j
            - indices[k, i, j] := interior_indices[i, j, k+1] for all i, j and for k=1:(d-1)

        """
        dm2, n, m = interior_indices.shape

        indices = np.zeros((dm2 + 2, n, m), dtype=interior_indices.dtype)

        indices[0, ...] = np.repeat(np.arange(n), m).reshape((n, m))
        indices[-1, ...] = np.tile(np.arange(m), n).reshape((n, m))
        indices[1:-1, ...] = interior_indices
        return indices

    def get_coordinates(self, n_interface):
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
        points = self.path.points[n_interface]
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
        num_points_sets = self.path.num_points_sets
        x = np.zeros(num_points_sets, s.FLOAT)
        y = np.zeros(num_points_sets, s.FLOAT)
        z = np.zeros(num_points_sets, s.FLOAT)
        for (i, (points, j)) in enumerate(zip(self.path.points, indices)):
            x[i] = points.x[j]
            y[i] = points.y[j]
            z[i] = points.z[j]
        return g.Points(x, y, z, 'Ray')

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
        order = 'F' if self.indices.flags.f_contiguous else 'C'

        shape = self.indices.shape[1:]
        out = np.full(shape, False, order=order, dtype=np.bool)

        interior_indices = self.interior_indices
        middle_points = tuple(self.path.points)[1:-1]
        for (d, points) in enumerate(middle_points):
            np.logical_or(out, interior_indices[d, ...] == 0, out=out)
            np.logical_or(out, interior_indices[d, ...] == (len(points) - 1), out=out)
        return out

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
            _expand_rays(interior_indices, indices_new_interface, expanded_indices, n, m, p, d)
            return expanded_indices


class Path(tuple):
    """
    Path(points_and_speeds)

    Path objects contain the interfaces (sets of points) which rays pass through during propagation, and the speeds
    between contiguous interfaces.

    A Path must starts and ends with Points objects. Speeds (stored
    as float) and Points must alternate.

    Ex: Path((points_1, speed_1_2, points_2, speed_2_3, points_3))

    """

    def __new__(cls, sequence):
        if len(sequence) % 2 == 0 or len(sequence) < 3:
            raise ValueError('{} expects a sequence of length odd and >= 5)'.format(
                cls.__name__))
        return super().__new__(cls, sequence)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join([str(x) for x in self]))

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
        return tuple(obj for (i, obj) in enumerate(self) if i % 2 == 0)

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
    paths : set of Path
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

    def __init__(self, paths, dtype=None, dtype_indices=None):
        if dtype is None:
            dtype = s.FLOAT

        if dtype_indices is None:
            max_length = max((p.len_largest_interface for p in paths))
            #dtype_indices = smallest_uint_that_fits(max_length)
            dtype_indices = s.UINT

        for path in paths:
            try:
                hash(path)
            except TypeError as e:
                raise TypeError("Path must be hashable.") from e

        self.dtype = dtype
        self.dtype_indices = dtype_indices
        self.clear_cache()
        self.res = {}
        self.paths = paths

        self.num_minimization = 0
        self.num_euc_distance = 0

    @classmethod
    def from_views(cls, views, dtype=None, dtype_indices=None):
        paths = set((path for v in views for path in (v.tx_path, v.rx_path)))
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
        tic = time.clock()
        for path in self.paths:
            self.res[path] = self._solve(path)
        toc = time.clock()
        logger.info("Ray tracing: solved all in {:.3g}s".format(toc - tic))
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
        logger.debug("Ray tracing: solve for subpaths {} and {}".format(str(head), str(tail)))
        times, indices_at_interface = find_minimum_times(
            res_head.times,
            res_tail.times,
            dtype=self.dtype,
            dtype_indices=self.dtype_indices)

        assert res_tail.path.num_points_sets == 2
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


class View(namedtuple('View', ['tx_path', 'rx_path', 'name'])):
    """
    View(tx_path, rx_path, name)
    """
    __slots__ = []

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)
