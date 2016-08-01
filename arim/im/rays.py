from collections import namedtuple

import numpy as np

from .. import geometry as g
from .. import settings as s
from ..utils import chunk_array, smallest_uint_that_fits

__all__ = ['Rays']


class Rays:
    """
    Rays(times, indices, path)

    Store the rays between the first and last datasets of points along
    a specific path.

    - n: number of points of the first set of points.
    - m: number of points of the last set of points.
    - p: number of datasets of points in the path.

    We name points_1, points_2, ..., points_p the datasets of points along the path.
    A ray passes through points_1, points_2, ..., points_p in this order.


    Parameters
    ----------
    times : ndarray of floats [n x m]
        Shortest time between first and last set of points.
        ``times[i, j]`` is the minimal time between the i-th point of the first
        interface and the j-th point of the last interface. In other words,
        ``times[i, j]`` is the minimal time between ``points_1[i]`` and
        ``points_p[j]``.
    indices : ndarray of floats [n x m x p]
        Indices of points for each ray.
        A ray starting from ``points_1[i]`` and ending in ``points_p[i]`` passes through
        the k-th interface at the point indexed by ``indices[i, j, k]``.
        By definition, ``indices[i, j, 0] := i`` and ``indices[i, j, p-1] := j``
    path : Path
        Sets of points crossed by the rays.

    """

    # __slots__ = []

    def __init__(self, times, indices, path):
        assert times.shape == indices.shape[:2]
        assert path.num_points_sets == indices.shape[2]
        self._times = times
        self._indices = indices
        self._path = path

    @property
    def path(self):
        return self._path

    @property
    def times(self):
        return self._times

    @property
    def indices(self):
        return self._indices

    def get_coordinates(self, n_interface):
        """
        Yields the coordinates of the rays of the n-th interface, as a tuple
        of three 2d ndarrays.

        Use numpy fancy indexing.

        Example
        -------
        ::

            for (d, (x, y, z)) in enumerate(rays.get_coordinates()):
                # Coordinates at the d-th interface of the ray between ray points_1[i] and
                # ray_points_p[j].
                x[i, j]
                y[i, j]
                z[i, j]


        """
        points = self.path.points[n_interface]
        indices = self.indices[..., n_interface]
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
        indices = self.indices[start_index, end_index, :]
        num_points_sets = self.path.num_points_sets
        x = np.zeros(num_points_sets, s.FLOAT)
        y = np.zeros(num_points_sets, s.FLOAT)
        z = np.zeros(num_points_sets, s.FLOAT)
        for (i, (points, j)) in enumerate(zip(self.path.points, indices)):
            x[i] = points.x[j]
            y[i] = points.y[j]
            z[i] = points.z[j]
        return g.Points(x, y, z, 'Ray')

    def to_contiguous(self):
        """Returns a named tuple whose all arrays are stored in C-order."""
        times = np.ascontiguousarray(self.times)
        indices = np.ascontiguousarray(self.indices)
        return self.__class__(times, indices, self.path)

    def to_fortran(self):
        """Returns a named tuple whose all arrays are stored in Fortran order."""
        times = np.asfortranarray(self.times)
        indices = np.asfortranarray(self.indices)
        return self.__class__(times, indices, self.path)

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

        shape = self.indices.shape[0:2]
        out = np.full(shape, False, order=order, dtype=np.bool)

        middle_points = tuple(self.path.points)[1:-1]
        for (d, points) in enumerate(middle_points, start=1):
            indices = self.indices[..., d]

            out = np.logical_or(out, indices == 0, out=out)
            out = np.logical_or(out, indices == (len(points) - 1), out=out)
        return out
