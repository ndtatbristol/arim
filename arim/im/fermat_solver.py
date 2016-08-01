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

import numba
import numpy as np

from .. import geometry as g
from ..core.cache import Cache
from .base import find_minimum_times
from .. import settings as s
from ..utils import chunk_array, smallest_uint_that_fits
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

import numba
import numpy as np

from .. import geometry as g
from ..core.cache import Cache
from .base import find_minimum_times
from .. import settings as s
from ..utils import chunk_array, smallest_uint_that_fits
from .rays import Rays


__all__ = ['FermatSolver', 'View', 'Path']

logger = logging.getLogger(__name__)




def assemble_rays(indices_head, indices_tail, indices_at_interface,
                  dtype=None, block_size=None, numthreads=None):
    """
    Assemble two rays.

    Parameters
    ----------
    indices_head: A to C in path ABC
        Shape (n, m, d1)
    indices_tail: C to F in path CDEF
        Shape (m, p, d2)
    indices_at_interface: indices at interface C of minimal rays between A to F in path ABCDEF
        Shape (n, p)

    Returns
    -------
    out_ray
        Shape (n, p, d1+d2-1)

    """
    n, m, d1 = indices_head.shape
    m_, p, d2 = indices_tail.shape
    assert m == m_
    assert indices_at_interface.shape == (n, p)

    d = d1 + d2 - 1

    if dtype is None:
        dtype = np.result_type(indices_head, indices_tail, indices_at_interface)

    if block_size is None:
        block_size = s.BLOCK_SIZE_FIND_MIN_TIMES
    if numthreads is None:
        numthreads = s.NUMTHREADS

    # Allocate out:
    out_indices = np.full((n, p, d), 0, dtype=dtype)

    indices_head = np.ascontiguousarray(indices_head)
    indices_tail = np.asfortranarray(indices_tail)

    block_size_adj = math.ceil(block_size / m)

    futures = []
    with ThreadPoolExecutor(max_workers=numthreads) as executor:
        for chunk1 in chunk_array((n, m, d1), block_size_adj, axis=0):
            for chunk2 in chunk_array((m, p, d2), block_size_adj, axis=1):
                chunk_res = (chunk1[0], chunk2[1], ...)

                futures.append(executor.submit(
                    _assemble_rays,
                    indices_head[chunk1], indices_tail[chunk2],
                    indices_at_interface[chunk_res],
                    out_indices[chunk_res]))
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()
    return out_indices


@numba.jit(nopython=True, nogil=True)
def _assemble_rays(indices_head, indices_tail, indices_at_interface, out_indices):
    """
    The layout given are for optimal speeds.

    Parameters
    ----------
    indices_head: A to C in path ABC
        Layout: C
        Shape (n, m, d1)
    indices_tail: C to F in path CDEF
        Layout: F
        Shape (m, p, d2)
    indices_at_interface: indices at interface C of minimal rays between A to F in path ABCDEF
        Layout: C
        Shape (n, m)
    out_indices: A to F in path ABCDEF
        Layout: C
        Shape (n, p, d1+d2-1)

    Returns
    -------
    None: write output in ``out_indices``

    """
    n, m, d1 = indices_head.shape
    m_, p, d2 = indices_tail.shape
    for i in range(n):
        for j in range(p):
            best_index = indices_at_interface[i, j]

            # Populate out_min_indices:
            for d in range(d1 - 1):
                out_indices[i, j, d] = indices_head[i, best_index, d]
            out_indices[i, j, d1 - 1] = best_index
            for d in range(d2 - 1):
                out_indices[i, j, d1 + d] = indices_tail[best_index, j, d + 1]


def get_ray_coordinates_np(ray_indices, interfaces, dtype=None):
    for (d, interface) in enumerate(interfaces):
        yield (interface.x[ray_indices[..., d]],
               interface.y[ray_indices[..., d]],
               interface.z[ray_indices[..., d]])


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
            raise ValueError('{} expects a sequence of length odd and >= 5)'.format(cls.__name__))
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


def make_empty_ray_indices(num1, num2, dtype):
    """
    Arguments Rays.indices in the case of two consecutive interfaces:

        indices[i, j, 0] := i
        indices[i, j, 1] := j

        indices.shape := (num1, num2, 2)

    """
    indices = np.zeros((num1, num2, 2), dtype=dtype)
    indices[..., 0] = np.repeat(np.arange(num1), num2).reshape((num1, num2))
    indices[..., 1] = np.tile(np.arange(num2), num1).reshape((num1, num2))
    return indices


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
            dtype_indices = smallest_uint_that_fits(max_length)

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
        self.convert_result_to_fortran()
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
        indices = assemble_rays(res_head.indices, res_tail.indices,
                                indices_at_interface, dtype=self.dtype_indices)

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
        ray_indices = make_empty_ray_indices(len(points1), len(points2), self.dtype_indices)
        return Rays(distance / speed, ray_indices, path)

    def convert_result_to_contiguous(self):
        """Convert all results to C order."""
        new_res = {path: ray.to_contiguous() for (path, ray) in self.res.items()}
        self.res = new_res
        return self.res

    def convert_result_to_fortran(self):
        """Convert all results to Fortran order."""
        new_res = {path: ray.to_fortran() for (path, ray) in self.res.items()}
        self.res = new_res
        return self.res


class View(namedtuple('View', ['tx_path', 'rx_path', 'name'])):
    """
    View(tx_path, rx_path, name)
    """
    __slots__ = []

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)
