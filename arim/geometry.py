"""
Utilities for geometric operations: translation, rotation, change of basis, etc.

Basis
=====

A basis (i_hat, j_hat, k_hat) is stored as a matrix::

            (i1 i2 i3)
   basis =  (j1 j2 j3)
            (k1 k2 k3)

where (i1, i2, i3) is the coordinate of the basis vector i_hat in the global coordinate system.

Remark: this storage is consistent with the Points layout: ``basis[0, :] = (i1, i2, i3)``. A basis can be seen as three
points (i_hat, j_hat, k_hat).

"""

# Remark: declaration of constant GCS at the end of this file
import math
from collections import namedtuple
import collections.abc
from concurrent.futures import ThreadPoolExecutor
from warnings import warn

import numpy as np
import numpy.linalg as linalg

from .utils import get_shape_safely, chunk_array
from . import settings as s
from .exceptions import ArimWarning, InvalidDimension, InvalidShape
from .core.cache import Cache, NoCache

__all__ = ['rotation_matrix_x', 'rotation_matrix_y', 'rotation_matrix_z',
           'rotation_matrix_ypr', 'are_points_aligned', 'norm2', 'norm2_2d', 'direct_isometry_2d',
           'direct_isometry_3d', 'Grid', 'Points', 'GCS', 'are_points_close', 'distance_pairwise',
           'GeometryHelper']
import numba

SphericalCoordinates = namedtuple('SphericalCoordinates', 'r theta phi')


class Points:
    '''
    Set of points in a 3D space.

    The coordinates (x, y, z) are stored contiguously.

    This object can contain a grid of any dimension of points:
        - one point (``points.shape == ()``)
        - a vector of points (``points.shape == (n, )``)
        - a matrix of points (``points.shape == (n, m)``)
        - etc.

    The lenght of Points (``len(points)``) is the number of points in the first dimension of the grid. If there is only
    one point, a TypeError is raised.

    Unless otherwise stated, in this class ``idx`` is a multidimensional index of a point. ``len(idx)`` equals the
    dimension of the Points object.

    Points objects support indexing: ``points[idx]`` is one point (ndarray of 3 numbers).

    Points objects are iterable: one point at a time is returned. The points are iterated over such as the right-hand
    side of the multidimensional varies the quickest. For example, for a Points object of shape (2, 2, 2), the points
    are returned in the following order: (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0),
    (1, 0, 1), (1, 1, 0), (1, 1, 1). The method ``enumerate`` returns the indices and the points in the same order.

    For convenience, wraps several functions of ``arim.geometry``.

    Parameters
    ----------
    coords: ndarray
        Dimension: (\*shape, 3)
    name: str or None
        Name of the set of points.

    Attributes
    ----------
    shape : tuple
        () if there is one point, (n, ) if vector of n points, (n, m) if matrix of points, etc.
    size_npoints : int
        Total number of points.

    '''
    __slots__ = ('coords', 'name')

    def __init__(self, coords, name=None):
        coords = np.asarray(coords)
        assert coords.flags.contiguous
        assert coords.shape[-1] == 3

        self.coords = coords
        self.name = name

    @classmethod
    def from_xyz(cls, x, y, z, name=None):
        assert x.ndim == y.ndim == z.ndim
        assert x.shape == y.shape == z.shape
        assert x.dtype == y.dtype == z.dtype

        coords = np.stack([x, y, z], axis=-1)
        return cls(coords, name)

    @property
    def shape(self):
        return self.coords.shape[:-1]

    @property
    def ndim(self):
        return self.coords.ndim - 1

    @property
    def size(self):
        size = 1
        for i in self.shape:
            size *= i
        return size

    def __str__(self):
        return 'P:{}'.format(self.name)

    def __repr__(self):
        classname = self.__class__.__qualname__
        if self.name is None:
            return '<{}{} at {}>'.format(classname, self.shape, hex(id(self)))
        else:
            return '<{}{}: {} at {}>'.format(classname, self.shape, self.name, hex(id(self)))

    @property
    def x(self):
        return self.coords[..., 0]

    @property
    def y(self):
        return self.coords[..., 1]

    @property
    def z(self):
        return self.coords[..., 2]

    @property
    def dtype(self):
        return self.coords.dtype

    def closest_point(self, x, y, z):
        dx = self.x - x
        dy = self.y - y
        dz = self.z - z
        return np.argmin(dx * dx + dy * dy + dz * dz)

    def allclose(self, other, atol=1e-8, rtol=0.):
        return are_points_close(self, other, atol=atol, rtol=rtol)

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError("Points is unsized")

    def __getitem__(self, key):
        """
        Returns one point (array of three numbers).
        """
        return self.coords[key]

    def norm2(self, out=None):
        """
        Returns a array of shape `shape`
        """
        return norm2(self.x, self.y, self.z, out=out)

    def translate(self, direction, inplace=False):
        """
        Translate the points along a given direction or several directions.

        If one direction is given (array of shape (3, )), the same direction is applied to all points::

            NewCoords[idx] = OldCoords[idx] + Direction, for all idx

        If as many direction as points are given, each point will be translated from the corresponding direction given.

            NewCoords[idx] = OldCoords[idx] + Direction[idx], for all idx

        Returns a new Points object with the same shape as the current one.

        Parameters
        ----------
        direction : ndarray
            Shape: (\*shape, 3)
            Example: if the current
            Shape: (3, )
        inplace : bool
            Default: False

        Returns
        -------
        translated_points : Points
        """
        new_coords = self.coords + direction
        translated_points = self.__class__(new_coords, self.name)
        return translated_points

    def translate_inplace(self, direction):
        np.add(self.coords, direction, out=self.coords)
        return self

    def rotate(self, rotation_matrix, centre=None):
        """Rotates the points. Returns a new Points object.

        Cf. :func:`rotate`
        """
        return Points(rotate(self.coords, rotation_matrix, centre), self.name)

    def to_gcs(self, bases, origins):
        """Returns the coordinates of the points expressed in the global coordinate system. Returns a new Points object.

        Cf. :func:`to_gcs`
        """
        return Points(to_gcs(self.coords, bases, origins), self.name)

    def from_gcs(self, bases, origins):
        """Returns the coordinates of the points expressed in the basis/bases given as parameter. Returns a new Points object.

        Cf. :func:`from_gcs`
        """
        return Points(from_gcs(self.coords, bases, origins), self.name)

    def spherical_coordinates(self):
        """
        (r, θ, φ)

        Quoted from [Spherical coordinate system](https://en.wikipedia.org/wiki/Spherical_coordinate_system):

            Spherical coordinates (r, θ, φ) as commonly used in physics: radial distance r, polar angle θ (theta),
            and azimuthal angle φ (phi).

        Returns
        -------
        r
            Radial distance.
        theta
            Polar angle.
        phi
            Azimuthal angle.
        """
        return spherical_coordinates(self.x, self.y, self.z)

    def __iter__(self):
        for idx in np.ndindex(self.shape):
            yield self.coords[idx]

    def enumerate(self):
        """
        Yield the index and the coordinates of each point.

        Yields
        ------
        idx: tuple
            Multidimensional index of the point
        (x, y, z) : ndarray of 3 numbers

        """
        for idx in np.ndindex(self.shape):
            yield idx, self.coords[idx]


class CoordinateSystem(namedtuple('CoordinateSystem', 'origin i_hat j_hat')):
    """
    A point and a direct 3D affine basis.

    A more accurate word to describe this object should be "affine frame", however we keep
    coordinate system to be consistent with MFMC terminology (global and probe coordinate systems)
    and to avoid confusion with the word "frame" as a set of scanline.


    Attributes
    ----------
    origin
    i_hat
    j_hat
    k_hat
    basis_matrix : ndarray
        i_hat, j_hat and k_hat stored in columns ('matrice de passage' de la base locale vers GCS).
        TODO: different convention as stated in header of the file
    """
    __slots__ = ()

    def __new__(cls, origin, i_hat, j_hat):
        i_hat = np.asarray(i_hat)
        j_hat = np.asarray(j_hat)
        origin = np.asarray(origin)
        get_shape_safely(i_hat, 'i_hat', (3,))
        get_shape_safely(j_hat, 'j_hat', (3,))
        get_shape_safely(origin, 'origin', (3,))
        if not np.isclose(norm2(*i_hat), 1.0):
            raise ValueError("Vector must be normalised.")
        if not np.isclose(norm2(*j_hat), 1.0):
            raise ValueError("Vector must be normalised.")
        if not np.isclose(i_hat @ j_hat, 0.):
            raise ValueError("The vectors must be orthogonal.")

        return super().__new__(cls, origin, i_hat, j_hat)

    @property
    def k_hat(self):
        return np.cross(self.i_hat, self.j_hat)

    def convert_from_gcs(self, points_gcs):
        """
        Parameters
        ----------
        points_gcs : Points
            Points whose coordinates are expressed in the GCS.

        Returns
        -------
        points_cs : Points
            Points whose coordinates are expressed in this coordinate system.

        See Also
        --------
        convert_to_gcs
        """
        points = points_gcs.translate(-self.origin)
        # TODO: improve convert_from_gcs
        return Points(points.coords @ self.basis_matrix)

    def convert_from_gcs_pairwise(self, points_gcs, origins):
        """
        Returns the coordinates of the 'points_gcs' in the following CS:

        - (origins[0], i_hat, j_hat, k_hat)
        - (origins[1], i_hat, j_hat, k_hat)
        - ...
        - (origins[num2-1], i_hat, j_hat, k_hat)

        Coordinates are returned as three 2d ndarrays (x, y, z) such as x[i, j] is the x-coordinate of the
        i-th point of ``points_gcs`` in the j-th coordinate system.

        Parameters
        ----------
        points_gcs : Points
            Points to convert (coordinates in GCS).
        origins : Points
            Origins of the coordinate systems where to express the points. Must be in the current coordinate system.

        Returns
        -------
        x, y, z : ndarray
            2d ndarray.

        Notes
        -----
        This function is used to express a set of points relatively to a set of probe elements.
        """
        # The rotation of the points is likely to be the longest operation. Do it only once.
        points_cs = self.convert_from_gcs(points_gcs)

        # C_ij = A_i + B_j
        x = points_cs.x[..., np.newaxis] - origins.x[np.newaxis, ...]
        y = points_cs.y[..., np.newaxis] - origins.y[np.newaxis, ...]
        z = points_cs.z[..., np.newaxis] - origins.z[np.newaxis, ...]

        return x, y, z

    def convert_to_gcs(self, points_cs):
        """
        Convert coordinates in the current coordinate system in the global one.

            OM' = Origin + OM_x * i_hat + OM_y * j_hat + OM_z * k_hat

        Parameters
        ----------
        points_cs : Points
            Points whose coordinates are expressed in this coordinate system.

        Returns
        -------
        points_gcs : Points
            Points whose coordinates are expressed in the global coordinate system.

        See Also
        --------
        convert_from_gcs
        """
        points_cs = points_cs.coords

        # Vectorise the following operation:
        # OM' = Origin + OM_x * i_hat + OM_y * j_hat + OM_z * k_hat
        return Points((points_cs @ self.basis_matrix.T) + self.origin)

    def translate(self, vector):
        """
        Translate the coordinate system. Returns a new instance.

        Parameters
        ----------
        vector

        Returns
        -------
        cs
            New coordinate system.
        """
        old_origin = Points(self.origin)
        new_origin = old_origin.translate(vector)[()]
        return self.__class__(new_origin, self.i_hat, self.j_hat)

    def rotate(self, rotation_matrix, centre=None):
        """
        Rotate the coordinate system. Returns a new instance.

        Parameters
        ----------
        rotation_matrix
        centre

        Returns
        -------
        cs
            New coordinate system.
        """
        old_basis = np.stack((self.origin, self.origin + self.i_hat, self.origin + self.j_hat), axis=0)
        new_basis = Points(old_basis).rotate(rotation_matrix, centre).coords

        origin = new_basis[0, :]
        i_hat = new_basis[1, :] - origin
        j_hat = new_basis[2, :] - origin
        return self.__class__(origin, i_hat, j_hat)

    def isclose(self, other, atol=1e-8, rtol=0.):
        """
        Compare two coordinate system.
        """
        return (
            np.allclose(self.origin, other.origin, rtol=rtol, atol=atol) and
            np.allclose(self.i_hat, other.i_hat, rtol=rtol, atol=atol) and
            np.allclose(self.j_hat, other.j_hat, rtol=rtol, atol=atol))

    @property
    def basis_matrix(self):
        # i_hat, j_hat and k_hat stored in columns
        return np.stack((self.i_hat, self.j_hat, self.k_hat), axis=1)


class Grid:
    """
    Regularly spaced 3d grid

    Attributes
    ----------
    x: 1d array
        Unique points along first axis
    x: 1d array
        Unique points along second axis
    z: 1d array
        Unique points along third axis
    xx: 3d array
        First coordinate of all points
    yy: 3d array
        Second coordinate of all points
    zz: 3d array
        Third coordinate of all points
    dx, dy, dz: float
        Exact distance between points
    numx, numy, numz, numpoints

    Parameters
    ----------
    pixel_size: float
        *Approximative* distance between points to use. Either one or three floats.
    """

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, pixel_size):
        try:
            dx, dy, dz = pixel_size
        except TypeError:
            dx = pixel_size
            dy = pixel_size
            dz = pixel_size

        if xmin == xmax:
            x = np.array([xmin])
            dx = None
        else:
            if xmin > xmax:
                warn("xmin > xmax in grid", ArimWarning)
            x = np.linspace(xmin, xmax, np.abs(np.ceil((xmax - xmin) / dx)) + 1, dtype=s.FLOAT)
            dx = np.mean(np.diff(x))

        if ymin == ymax:
            y = np.array([ymin], dtype=s.FLOAT)
            dy = None
        else:
            if ymin > ymax:
                warn("ymin > ymax in grid", ArimWarning)
            y = np.linspace(ymin, ymax, np.abs(np.ceil((ymax - ymin) / dy)) + 1, dtype=s.FLOAT)
            dy = np.mean(np.diff(y))

        if zmin == zmax:
            z = np.array([zmin], dtype=s.FLOAT)
            dz = None
        else:
            if zmin > xmax:
                warn("zmin > xmax in grid", ArimWarning)
            z = np.linspace(zmin, zmax, np.abs(np.ceil((zmax - zmin) / dz)) + 1, dtype=s.FLOAT)
            dz = np.mean(np.diff(z))

        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, indexing='ij')

        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz

    @property
    def xmin(self):
        return self.x[0]

    @property
    def xmax(self):
        return self.x[-1]

    @property
    def ymin(self):
        return self.y[0]

    @property
    def ymax(self):
        return self.y[-1]

    @property
    def zmin(self):
        return self.z[0]

    @property
    def zmax(self):
        return self.z[-1]

    @property
    def numx(self):
        return len(self.x)

    @property
    def numy(self):
        return len(self.y)

    @property
    def numz(self):
        return len(self.z)

    @property
    def numpoints(self):
        return self.numx * self.numy * self.numz

    @property
    def as_points(self):
        """
        Returns the grid points as Points object of dimension 1 (flatten the grid points). Returns always the same object.
        """
        if getattr(self, '_points', None) is None:
            self._points = Points.from_xyz(self.xx.ravel(), self.yy.ravel(), self.zz.ravel(),
                                           self.__class__.__qualname__)
        return self._points


class GeometryHelper:
    """
    Helper class for computing distances and angles between two sets of points.

    Canonical usage: compute distances between probe elements and all grid points.

    Warning: if the points coordinates evolve over time, do not use the cache, or clear the cache at every change.
    """

    def __init__(self, points1_gcs, points2_gcs, pcs, use_cache=True):
        """

        Parameters
        ----------
        points1_gcs : Points
        points2_gcs : Points
        pcs : CoordinateSystem
        use_cache : bool

        Returns
        -------

        """
        self._points1_gcs = points1_gcs
        self._points2_gcs = points2_gcs

        assert isinstance(pcs, CoordinateSystem)
        self._pcs = pcs

        if use_cache:
            self._cache = Cache()
        else:
            # this will drop silently any attempt to data on this
            self._cache = NoCache()

        self.distance_pairwise_kwargs = {}

    @property
    def points1_gcs(self):
        return self._points1_gcs

    @property
    def points2_gcs(self):
        return self._points2_gcs

    @property
    def points1_pcs(self):
        return self._pcs.convert_from_gcs(self._points1_gcs)

    @property
    def pcs(self):
        return self._pcs

    def is_valid(self, probe, points):
        return (probe is self._points1_gcs) and (points is self._points2_gcs)

    def distance_pairwise(self):
        out = self._cache.get('distance_pairwise', None)
        if out is None:
            out = distance_pairwise(self._points2_gcs, self._points1_gcs, **self.distance_pairwise_kwargs)
            self._cache['distance_pairwise'] = out
        return out

    def points2_to_pcs_pairwise(self):
        out = self._cache.get('points2_to_pcs_pairwise', None)
        if out is None:
            out = self._pcs.convert_from_gcs_pairwise(self._points2_gcs, self.points1_pcs)
            self._cache['points2_to_pcs_pairwise'] = out
        return out

    def points2_to_pcs_pairwise_spherical(self):
        out = self._cache.get('points2_to_pcs_pairwise_spherical', None)
        if out is None:
            points_pcs = self.points2_to_pcs_pairwise()
            dist = self.distance_pairwise()
            out = spherical_coordinates(points_pcs[0], points_pcs[1], points_pcs[2], dist)
            self._cache['points2_to_pcs_pairwise_spherical'] = out
        return out

    def clear(self):
        self._cache.clear()

    def __str__(self):
        return '{} between {} and {}'.format(self.__class__.__name__, str(self._points1_gcs), str(self._points2_gcs))


def _spherical_coordinates_r(x, y, z, out=None):
    """radial distance
    """
    return norm2(x, y, z, out=out)


def _spherical_coordinates_theta(z, r, out=None):
    """inclination angle"""
    return np.arccos(z / r, out=out)


def _spherical_coordinates_phi(x, y, out=None):
    """azimuthal angle"""
    return np.arctan2(y, x, out=out)


def spherical_coordinates(x, y, z, r=None):
    """
    Compute the spherical coordinates (r, θ, φ) of points.

    Coordinates are assumed to be in the GCS (O, x_hat, y_hat, z_hat).

    Quoted from [Spherical coordinate system](https://en.wikipedia.org/wiki/Spherical_coordinate_system):

        Spherical coordinates (r, θ, φ) as commonly used in physics: radial distance r, polar angle θ (theta),
        and azimuthal angle φ (phi).

    Parameters
    ----------
    x, y, z : ndarray
    r : ndarray or None
        Computed on the fly is not provided.

    Returns
    -------
    r, theta, phi : ndarray
        Arrays with same shape as input.

    See Also
    --------
    Points.spherical_coordinates : corresponding function with the ``Points`` interface.

    """
    if r is None:
        r = _spherical_coordinates_r(x, y, z)
    theta = _spherical_coordinates_theta(z, r)
    phi = _spherical_coordinates_phi(x, y)
    return SphericalCoordinates(r, theta, phi)


def rotation_matrix_x(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array(((1, 0, 0),
                     (0, c, -s),
                     (0, s, c)), dtype=np.float)


def rotation_matrix_y(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array(((c, 0, s),
                     (0, 1, 0),
                     (-s, 0, c)), dtype=np.float)


def rotation_matrix_z(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array(((c, -s, 0),
                     (s, c, 0),
                     (0, 0, 1)), dtype=np.float)


def rotate(coords, rotation_matrix, centre=None):
    """
    Rotate these points given a rotation matrix and the centre.

    The rotation of a point OM (in column) is given by OM' such as:

        OM' := RotationMatrix x OM + Centre

    This function accepts multiple rotations: for example to have one rotation matrix per point of index ``idx``,
    ``rotation_matrix[idx]`` must be a 3x3 matrix.

    Parameters
    ----------
    coords : ndarray
        Coordinates to rotate. Shape: (*shape_points, 3)
    rotation_matrix
        Shape: (3, 3) or (*shape_points, 3, 3). Rotation matrices to apply: either one for all points or one per point.
    centre : ndarray, optional
        Centre of the rotation. This point is invariant by rotation.
        Shape: Shape: (3, 3) or (*shape_points, 3).
        Default: centre = (0., 0., 0.)

    Returns
    -------
    rotated_points : ndarray
        Shape: (*shape_points, 3
    """
    assert rotation_matrix.shape[-2:] == (3, 3)

    if centre is None:
        # Out[..., j] = Sum_i Rotation[...,j, i].In[...,i]
        rotated = np.einsum('...ji,...i->...j', rotation_matrix, coords)
    else:
        centre = np.asarray(centre)
        assert centre.shape[-1] == 3
        rotated = np.einsum('...ji,...i->...j', rotation_matrix, coords - centre) + centre
    return rotated


def to_gcs(coords_cs, bases, origins):
    """
    Convert the coordinates of points expressed in the basis/bases given as parameter to coordinates expressed in the
    global coordinate system.

    Warning: the bases must be **orthogonal**. No check is performed.

    Parameters
    ----------
    coords_cs : ndarray
        Shape: (*shape_points, 3)
        Coordinates of the points in the basis.
    bases : ndarray
        Shape: (*shape_points, 3, 3) or (3, 3)
        One or several orthogonal bases. For each basis, the coordinates of the basis vectors in the global
        coordinate system must be given row per row: i_hat, j_hat, k_hat.
    origins
        Shape: (*shape_points, 3) or (3, 3)

    Returns
    -------
    coords_gcs : ndarray
        Shape: (*shape_points, 3)
    """
    # OM' = Origin + OM_x * i_hat + OM_y * j_hat + OM_z * k_hat
    return np.einsum('...ij,...i->...j', bases, coords_cs) + origins


def from_gcs(points_gcs, bases, origins):
    """
    Convert the coordinates of points expressed in the global coordinate system to coordinates expressed
    in the basis/bases given as parameter.

    Warning: the bases must be **orthogonal**. No check is performed.

    Parameters
    ----------
    coords_gcs : ndarray
        Shape: (*shape_points, 3)
        Coordinates of the points in the GCS.
    bases : ndarray
        Shape: (*shape_points, 3, 3) or (3, 3)
        One or several orthogonal bases. For each basis, the coordinates of the basis vectors in the global
        coordinate system must be given row per row: i_hat, j_hat, k_hat.
    origins
        Shape: (*shape_points, 3) or (3, 3)

    Returns
    -------
    coords_gcs : ndarray
        Shape: (*shape_points, 3)
    """
    return np.einsum('...ji,...i->...j', bases, points_gcs - origins)


def rotation_matrix_ypr(yaw, pitch, roll):
    '''Returns the rotation matrix (as a ndarray) from the yaw, pitch and roll
    in radians.

    This matrix corresponds to a intrinsic rotation around axes z, y, x respectively.

    https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    '''
    return rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)


def are_points_close(points1, points2, atol=1e-8, rtol=0.):
    """
    Return True if and only if the two sets of points have the same shape and coordinates close
    to the given precision.

    Attribute name is ignored.

    Parameters
    ----------
    points1 : Points
    points2 : Points
    atol : float
    rtol : float

    Returns
    -------
    bool
    """
    return (
        len(points1.shape) == len(points2.shape) and
        np.allclose(points1.coords, points2.coords, rtol=rtol, atol=atol))


def are_points_aligned(points, rtol=0., atol=1e-08):
    """
    Are the points aligned? Returns a boolean.

    Compute the cross products AB ^ AC, AB ^ AD, AB ^ AE, ...

    Warning: the two first points must be distinct (TODO: fix this).

    Parameters
    ----------
    points : Points
        Points
    rtol, atol : float
        Parameters for numpy.allclose.

    """
    numpoints = len(points)

    # TODO: are_points_aligned -> use coordinate system?
    points = points.coords
    if numpoints <= 2:
        return True

    # We call A, B, C, ... the points.

    # vectors AB, AC, AD...
    AM = points[1:, :] - points[0, :]

    AB = AM[0, :]
    assert not np.allclose(AB, np.zeros(3)), 'this function does not work if the two first points are the same'

    # Cross product AC ^ AB, AD ^ AB, AE ^ AB...
    cross = np.cross(AM[1:, :], AB)

    return np.allclose(cross, np.zeros_like(cross), rtol=rtol, atol=atol)


def norm2(x, y, z, out=None):
    """
    Euclidean norm of a ndarray

    Parameters
    ----------
    x, y, z : ndarray
    out : ndarray or None
        For inplace operations.

    Returns
    -------

    """
    if out is None:
        out = np.zeros_like(x)
    out += x * x
    out += y * y
    out += z * z
    return np.sqrt(out, out=out)


def norm2_2d(x, y, out=None):
    """
    Euclidean norm of a ndarray

    Parameters
    ----------
    x, y, z : ndarray
    out : ndarray or None
        For inplace operations.

    Returns
    -------

    """
    if out is None:
        out = np.zeros_like(x)
    out += x * x
    out += y * y
    return np.sqrt(out, out=out)


def direct_isometry_2d(A, B, Ap, Bp):
    """
    Returns a direct isometry that transform A to A' and B to B'.

    Parameters
    ----------
    originalA
    originalB
    transformedA
    transformedB

    Returns
    -------
    M : ndarray
    P : ndarray
        Such as: X' = M @ X + P

    """

    # Shorter notations:
    A = np.asarray(A)
    B = np.asarray(B)
    Ap = np.asarray(Ap)
    Bp = np.asarray(Bp)
    assert A.shape == (2,)
    assert B.shape == (2,)
    assert Ap.shape == (2,)
    assert Bp.shape == (2,)

    assert np.isclose(norm2_2d(*(B - A)), norm2_2d(*(Bp - Ap)))

    # Angle (Ox, AB)
    AB = B - A
    phi = np.arctan2(AB[1], AB[0])

    # Angle (Ox, ApBp)
    ApBp = Bp - Ap
    psi = np.arctan2(ApBp[1], ApBp[0])

    theta = psi - phi
    C = np.cos(theta)
    S = np.sin(theta)

    M = np.array([(C, -S), (S, C)])
    P = Bp - M @ B

    return M, P


def direct_isometry_3d(A, i_hat, j_hat, B, u_hat, v_hat):
    """
    Returns the isometry that send the direct orthogonal base (A, i_hat, j_hat, i_hat^j_hat)
    to (B, u_hat, v_hat, u_hat^v_hat)

    Returns M, P such as:

        X' = M @ X + P


    Parameters
    ----------
    A
    i_hat
    j_hat
    B
    u_hat
    v_hat

    Returns
    -------
    M
    P

    """
    A = np.asarray(A)
    B = np.asarray(B)
    u_hat = np.asarray(u_hat)
    v_hat = np.asarray(v_hat)
    i_hat = np.asarray(i_hat)
    j_hat = np.asarray(j_hat)

    assert A.shape == (3,)
    assert B.shape == (3,)
    assert u_hat.shape == (3,)
    assert v_hat.shape == (3,)
    assert i_hat.shape == (3,)
    assert j_hat.shape == (3,)

    assert np.isclose(norm2(*u_hat), 1.)
    assert np.isclose(norm2(*v_hat), 1.)
    assert np.isclose(norm2(*i_hat), 1.)
    assert np.isclose(norm2(*j_hat), 1.)

    assert np.allclose(i_hat @ j_hat, 0.)
    assert np.allclose(u_hat @ v_hat, 0.)

    k_hat = np.cross(i_hat, j_hat)
    w_hat = np.cross(u_hat, v_hat)

    baseDep = np.stack((i_hat, j_hat, k_hat), axis=1)
    baseArr = np.stack((u_hat, v_hat, w_hat), axis=1)

    # baseArr = M @ baseDep
    # <=> baseDep.T @ M.T = baseArr.T
    M = linalg.solve(baseDep.T, baseArr.T).T

    # assert np.allclose(M @ baseDep, baseArr)

    # Y = M @ (X - A) + B
    AB = B - A
    P = B - M @ A

    return M, P


def distance_pairwise(points1, points2, out=None, dtype=None, block_size=None, numthreads=None):
    '''
    Compute the Euclidean distances of flight between two sets of points.

    The time of flight between the two points ``A(x1[i], y1[i], z1[i])`` and ``B(x2[i], y2[i], z2[i])`` is:

       distance[i, j] := distance_pairwise(A, B)
                      := sqrt( delta_x**2 + delta_y**2 + delta_z**2 )

    The computation is parallelized with multithreading. Both sets of points are chunked.

    Parameters
    ----------
    points1 : Points
        Coordinates of the first set of points (num1 points).
    points2 : Points
        Coordinates of the first set of points (num2 points).
    out : ndarray, optional
        Preallocated array where to write the result.
        Default: allocate on the fly.
    dtype : numpy.dtype, optional
        Data type for `out`, if not given. Default: infer from points1, points2.
    block_size : int, optional
        Number of points to treat in a row.
        Default: arim.settings.BLOCK_SIZE_EUC_DISTANCE
    numthreads int, optional
        Number of threads to start.
        Default: number of CPU cores plus one.

    Returns
    -------
    distance : ndarray [num1 x num2]
        Euclidean distances between the points of the two input sets.
    '''
    # Check dimensions and shapes
    try:
        (num1,) = points1.x.shape
        (num2,) = points2.x.shape
    except ValueError:
        raise InvalidDimension("The dimension of the coordinates of points must be one.")
    if not (points1.x.shape == points1.y.shape == points1.z.shape):
        raise InvalidShape("Must have: points1.x.shape == points1.y.shape == points1.z.shape")
    if not (points2.x.shape == points2.y.shape == points2.z.shape):
        raise InvalidShape("Must have: points2.x.shape == points2.y.shape == points2.z.shape")

    if out is None:
        if dtype is None:
            # Infer dtype:
            dtype = np.result_type(points1.dtype, points2.dtype)
        distance = np.full((num1, num2), 0, dtype=dtype)
    else:
        distance = out
        if distance.shape != (num1, num2):
            raise InvalidShape("'distance'")

    if block_size is None:
        block_size = s.BLOCK_SIZE_EUC_DISTANCE
    if numthreads is None:
        numthreads = s.NUMTHREADS
    chunk_size = math.ceil(block_size / 6)

    futures = []
    with ThreadPoolExecutor(max_workers=numthreads) as executor:
        for chunk1 in chunk_array((num1,), chunk_size):
            for chunk2 in chunk_array((num2,), chunk_size):
                chunk_tof = (chunk1[0], chunk2[0])

                futures.append(executor.submit(
                    _distance_pairwise,
                    points1.x[chunk1], points1.y[chunk1], points1.z[chunk1],
                    points2.x[chunk2], points2.y[chunk2], points2.z[chunk2],
                    distance[chunk_tof]))
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()
    return distance


def is_orthonormal(basis):
    assert basis.shape == (3, 3)
    return (basis.dtype.kind == 'f' # must be real
            and np.allclose(basis.T, np.linalg.inv(basis)))

def is_orthonormal_direct(basis):
    """
    Returns True if the basis is orthonormal and direct:
    Parameters
    ----------
    basis

    Returns
    -------

    """
    return is_orthonormal(basis) and np.allclose(np.cross(basis[0, :], basis[1, :]), basis[2, :])


@numba.jit(nopython=True, nogil=True)
def _distance_pairwise(x1, y1, z1, x2, y2, z2, distance):
    """
    Cf. distance_pairwise.

    ``distance`` is the result. The array must be preallocated before.

    """
    # For each grid point and each element:
    num1, num2 = distance.shape

    for i in range(num1):
        for j in range(num2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dz = z1[i] - z2[j]
            distance[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return


GCS = CoordinateSystem(origin=np.array((0., 0., 0.)),
                       i_hat=np.array((1., 0., 0.)),
                       j_hat=np.array((0., 1., 0.)))
