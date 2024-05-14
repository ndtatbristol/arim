"""
.. currentmodule: arim.geometry

Utilities for geometric operations: translation, rotation, change of basis, etc.

Points
======

A :class:`Points` object contains the Cartesian coordinates of one or more
points. The points can be stored as a ndarray.
However ray tracing and many parts of arim expects as input an 1d array of
points.

Function :func:`points_1d_wall_z` provides an easy way to create a flat line.
For more complicated lines and surfaces, create the points manually.

Oriented points
===============

An oriented point is defined as a point and three orthonormal vectors.
It is actually a full coordinate system.

For probe and surfaces (front, back wall), the two first vectors must be
tangential to the surface and the third vector must be normal to it.

In the block in immersion model, the probe normals must be towards the
examination object. The front and back walls' normals must be towards the
probe.

For scatterers and grid points, there is no tangent or normal vectors.
Only the third vector of the basis is used. It defines the reference
orientation of the scatterer. To use a rotated scatterer, one can therefore
change the orientation of this third vector; however, the recommend technique
is to argument ``scat_angle`` in :func:`arim.model.model_amplitudes_factory`.

Basis
=====

A basis (i_hat, j_hat, k_hat) is stored as a matrix::

            (i1 i2 i3)
   basis =  (j1 j2 j3)
            (k1 k2 k3)

where (i1, i2, i3) is the coordinate of the basis vector i_hat in the global coordinate system.

Remark: this storage is consistent with the :class:`Points` layout: ``basis[0, :] = (i1, i2, i3)``. A basis can be seen as three
points (i_hat, j_hat, k_hat).

Warning: basis in :class:`CoordinateSystem` objects are stored in a different convention:
they are transposed i.e. ``basis[:, 0] = (i1, i2, i3)``.

Spherical coordinate system
===========================

Physics and ISO convention (r, theta, phi):

- r is the radial distance,
- theta is the polar angle (inclination) in the rangle in the range [0, pi],
- phi is the azimuthal angle in the range [-pi, pi].

Cf. `Wikipedia article on Spherical coordinate system <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_


.. data:: GCS

   Global coordinate system (:class:`CoordinateSystem`).
   ``i = (1, 0, 0)``, ``j = (0, 1, 0)``,
   ``k = (0, 0, 1)``, ``O = (0, 0, 0)``.

"""

# Remark: declaration of constant GCS at the end of this file
import concurrent.futures
import math
from collections import namedtuple
from warnings import warn

import numba
import numpy as np

from . import settings as s
from .exceptions import ArimWarning, InvalidDimension, InvalidShape
from .helpers import chunk_array


class SphericalCoordinates(namedtuple("SphericalCoordinates", "r theta phi")):
    """
    Spherical coordinates as usually defined in physics

    Cf. https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """

    @property
    def radius(self):
        return self.r

    @property
    def polar(self):
        return self.theta

    @property
    def azimuth(self):
        return self.phi


def aspoints(array_like):
    """
    Returns a Point object: the input itself if it is a point, or wrap the object
    as a Point otherwise.

    Parameters
    ----------
    array_like : Iterable or Points

    Returns
    -------
    Points

    """
    if isinstance(array_like, Points):
        return array_like
    else:
        return Points(array_like)


class Points:
    r"""
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

    """

    __slots__ = ("coords", "name")

    def __init__(self, coords, name=None):
        coords = np.asarray(coords)
        if coords.shape[-1] != 3:
            msg = "The dimension of the coords array should be (..., 3)"
            msg += " where 3 stands for x, y and z."
            raise ValueError(msg)

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
        return self.coords.size // 3

    @property
    def numpoints(self):
        return self.size

    def __str__(self):
        return f"P:{self.name}"

    def __repr__(self):
        classname = self.__class__.__qualname__
        if self.name is None:
            return f"<{classname}{self.shape} at {hex(id(self))}>"
        else:
            return f"<{classname}{self.shape}: {self.name} at {hex(id(self))}>"

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

    def allclose(self, other, atol=1e-8, rtol=0.0):
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

    def translate(self, direction):
        """
        Translate the points along a given direction or several directions.

        If one direction is given (array of shape (3, )), the same direction is
        applied to all points::

            NewCoords[idx] = OldCoords[idx] + Direction, for all idx

        If as many direction as points are given, each point will be translated
        from the corresponding direction given::

            NewCoords[idx] = OldCoords[idx] + Direction[idx], for all idx

        Returns a new Points object with the same shape as the current one.

        Parameters
        ----------
        direction : ndarray
            Shape: (3, )

        Returns
        -------
        translated_points : Points
        """
        new_coords = self.coords + direction
        translated_points = self.__class__(new_coords, self.name)
        return translated_points

    def rotate(self, rotation_matrix, centre=None):
        """Rotates the points. Returns a new Points object.

        Cf. :func:`rotate`
        """
        if centre is not None:
            centre = np.asarray(centre)
        return Points(rotate(self.coords, np.asarray(rotation_matrix), centre), self.name)

    def to_gcs(self, bases, origins):
        """Returns the coordinates of the points expressed in the global coordinate system.
        Returns a new Points object.

        Cf. :func:`to_gcs`
        """
        return Points(to_gcs(self.coords, bases, origins), self.name)

    def from_gcs(self, bases, origins):
        """Returns the coordinates of the points expressed in the basis/bases given as
        parameter. Returns a new Points object.

        Cf. :func:`from_gcs`
        """
        return Points(from_gcs(self.coords, bases, origins), self.name)

    def spherical_coordinates(self):
        """
        (r, theta, phi)

        Quoted from [Spherical coordinate system](https://en.wikipedia.org/wiki/Spherical_coordinate_system):

            Spherical coordinates (r, θ, φ) as commonly used in physics: radial distance r,
            polar angle θ (theta), and azimuthal angle φ (phi).

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

    def points_in_rectbox(
        self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None
    ):
        """
        Returns points in the rectangular box.

        See Also
        --------
        points_in_rectbox

        """
        return points_in_rectbox(
            self.x, self.y, self.z, xmin, xmax, ymin, ymax, zmin, zmax
        )

    def to_1d_points(self):
        """
        Returns a new 1d Points object (shape: (numpoints, ))

        Returns
        -------
        Points

        """
        return self.reshape(self.size)

    def reshape(self, new_shape):
        """
        Returns the reshaped coordinates.

        Parameters
        ----------
        shape : tuple

        Returns
        -------
        Points

        """
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        return Points(self.coords.reshape((*new_shape, 3)), self.name)


OrientedPoints = namedtuple("OrientedPoints", "points orientations")


class CoordinateSystem:
    """
    A point and a direct 3D affine basis.

    A more accurate word to describe this object should be "affine frame", however we keep
    coordinate system to be consistent with MFMC terminology (global and probe coordinate systems)
    and to avoid confusion with the word "frame" as a set of timetrace.


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

    __slots__ = ["_origin", "_i_hat", "_j_hat"]

    def __init__(self, origin, i_hat, j_hat):
        # init values
        self._i_hat = None
        self._j_hat = None
        self._origin = None

        # use the setters (they check the data for us):
        self.origin = origin
        self.i_hat = i_hat
        self.j_hat = j_hat

    @property
    def i_hat(self):
        return self._i_hat

    @i_hat.setter
    def i_hat(self, i_hat):
        i_hat = np.asarray(i_hat)
        if i_hat.shape != (3,):
            raise ValueError

        if not np.isclose(norm2(*i_hat), 1.0):
            raise ValueError("Vector must be normalised.")
        self._i_hat = i_hat

    @property
    def j_hat(self):
        return self._j_hat

    @j_hat.setter
    def j_hat(self, j_hat):
        j_hat = np.asarray(j_hat)
        if j_hat.shape != (3,):
            raise ValueError

        if not np.isclose(norm2(*j_hat), 1.0):
            raise ValueError("Vector must be normalised.")
        self._j_hat = j_hat

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        origin = np.asarray(origin)
        if origin.shape != (3,):
            raise ValueError
        self._origin = origin

    @property
    def k_hat(self):
        return np.cross(self.i_hat, self.j_hat)

    def convert_from_gcs(self, points_gcs):
        """
        Convert from global to local coordinate system

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
        old_basis = np.stack(
            (self.origin, self.origin + self.i_hat, self.origin + self.j_hat), axis=0
        )
        new_basis = Points(old_basis).rotate(rotation_matrix, centre).coords

        origin = new_basis[0, :]
        i_hat = new_basis[1, :] - origin
        j_hat = new_basis[2, :] - origin
        return self.__class__(origin, i_hat, j_hat)

    def isclose(self, other, atol=1e-8, rtol=0.0):
        """
        Compare two coordinate system.
        """
        return (
            np.allclose(self.origin, other.origin, rtol=rtol, atol=atol)
            and np.allclose(self.i_hat, other.i_hat, rtol=rtol, atol=atol)
            and np.allclose(self.j_hat, other.j_hat, rtol=rtol, atol=atol)
        )

    @property
    def basis_matrix(self):
        # i_hat, j_hat and k_hat stored in columns
        return np.stack((self.i_hat, self.j_hat, self.k_hat), axis=1)

    def copy(self):
        return self.__class__(self.origin.copy(), self.i_hat.copy(), self.j_hat.copy())


class Grid(Points):
    """
    Regularly spaced 3d grid

    Attributes
    ----------
    xvect: ndarray
        Unique points along first axis
    yvect: ndarray
        Unique points along second axis
    zvect: ndarray
        Unique points along third axis
    x: ndarray
        First coordinate of all points. Shape: ``(numx, numy, numz)``
    y: ndarray
        Second coordinate of all points. Shape: ``(numx, numy, numz)``
    z: ndarray
        Third coordinate of all points. Shape: ``(numx, numy, numz)``
    dx, dy, dz: float or None
        Exact distance between points. None if only one point along the axis
    numx, numy, numz, numpoints

    Parameters
    ----------
    xmin : float
    xmax : float
    xmin : float
    ymax : float
    zmin : float
    zmax  : float
    pixel_size: float
        *Approximative* distance between points to use. Either one or three floats.

    """

    __slots__ = ("coords", "name", "xvect", "yvect", "zvect")

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, pixel_size):
        try:
            dx, dy, dz = pixel_size
        except TypeError:
            dx = pixel_size
            dy = pixel_size
            dz = pixel_size

        if xmin == xmax:
            x = np.array([xmin])
        else:
            if xmin > xmax:
                warn("xmin > xmax in grid", ArimWarning)
            x = np.linspace(
                xmin, xmax, round((abs(xmax - xmin) + dx) / dx), dtype=s.FLOAT
            )

        if ymin == ymax:
            y = np.array([ymin], dtype=s.FLOAT)
        else:
            if ymin > ymax:
                warn("ymin > ymax in grid", ArimWarning)
            y = np.linspace(
                ymin, ymax, round((abs(ymax - ymin) + dy) / dy), dtype=s.FLOAT
            )

        if zmin == zmax:
            z = np.array([zmin], dtype=s.FLOAT)
        else:
            if zmin > zmax:
                warn("zmin > zmax in grid", ArimWarning)
            z = np.linspace(
                zmin, zmax, round((abs(zmax - zmin) + dz) / dz), dtype=s.FLOAT
            )

        all_coords = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        super().__init__(all_coords, "Grid")
        self.xvect = x
        self.yvect = y
        self.zvect = z

    @classmethod
    def grid_centred_at_point(
        cls, centre_x, centre_y, centre_z, size_x, size_y, size_z, pixel_size
    ):
        """
        Create a regularly spaced 3d grid centred around a point.

        The centre is a point of the grid, which imposes the number of points in any non-null direction is odd.

        Parameters
        ----------
        centre_x : float
        centre_y : float
        centre_z : float
        size_x : float
        size_y : float
        size_z : float
        pixel_size : float
            Approximate size

        Returns
        -------
        Grid

        """
        # Reminder:
        #   L = (N-1)*D
        #   D = L/(N-1)
        #   N = L/D + 1
        # The smallest odd integer above x is: math.ceil(x)|1
        assert size_x >= 0.0
        assert size_y >= 0.0
        assert size_z >= 0.0

        numpoints_x = math.ceil(size_x / pixel_size + 1) | 1
        numpoints_y = math.ceil(size_y / pixel_size + 1) | 1
        numpoints_z = math.ceil(size_z / pixel_size + 1) | 1

        # The pixel size is exactly size/(numpoints-1)
        try:
            dx = size_x / (numpoints_x - 1)
        except ZeroDivisionError:
            dx = size_x
        try:
            dy = size_y / (numpoints_y - 1)
        except ZeroDivisionError:
            dy = size_y
        try:
            dz = size_z / (numpoints_z - 1)
        except ZeroDivisionError:
            dz = size_z

        return cls(
            centre_x - size_x / 2,
            centre_x + size_x / 2,
            centre_y - size_y / 2,
            centre_y + size_y / 2,
            centre_z - size_z / 2,
            centre_z + size_z / 2,
            (dx, dy, dz),
        )

    def resample(self, new_pixel_size):
        """
        Returns a new Grid object with a new pixel size

        Parameters
        ----------
        new_pixel_size

        Returns
        -------
        Grid

        """
        return self.__class__(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
            new_pixel_size,
        )

    @property
    def xmin(self):
        return self.xvect[0]

    @property
    def xmax(self):
        return self.xvect[-1]

    @property
    def ymin(self):
        return self.yvect[0]

    @property
    def ymax(self):
        return self.yvect[-1]

    @property
    def zmin(self):
        return self.zvect[0]

    @property
    def zmax(self):
        return self.zvect[-1]

    @property
    def numx(self):
        return len(self.xvect)

    @property
    def numy(self):
        return len(self.yvect)

    @property
    def numz(self):
        return len(self.zvect)

    @property
    def dx(self):
        try:
            return self.xvect[1] - self.xvect[0]
        except IndexError:
            return None

    @property
    def dy(self):
        try:
            return self.yvect[1] - self.yvect[0]
        except IndexError:
            return None

    @property
    def dz(self):
        try:
            return self.zvect[1] - self.zvect[0]
        except IndexError:
            return None

    @property
    def as_points(self):
        """
        Returns the grid points as Points object of dimension 1 (flatten the grid points).
        """
        warn(DeprecationWarning("use method to_1d_points() instead"))
        return self.to_1d_points()

    def to_oriented_points(self):
        """
        Returns a 1d OrientedPoints from the grid points (assume default orientation)

        Returns
        -------
        OrientedPoints

        """
        return default_oriented_points(self.to_1d_points())


def spherical_coordinates_r(x, y, z, out=None):
    """radial distance"""
    return norm2(x, y, z, out=out)


def spherical_coordinates_theta(z, r, out=None):
    """polar angle"""
    return np.arccos(z / r, out=out)


def spherical_coordinates_phi(x, y, out=None):
    """azimuthal angle"""
    return np.arctan2(y, x, out=out)


def spherical_coordinates(x, y, z, r=None):
    """
    Compute the spherical coordinates (r, θ, φ) of points.

    r is positive or null. Theta is in [0, pi]. Phi is in [-pi, pi].

    Quoted from [Spherical coordinate system](https://en.wikipedia.org/wiki/Spherical_coordinate_system):

        Spherical coordinates (r, θ, φ) as commonly used in physics: radial distance r, polar angle θ (theta),
        and azimuthal angle φ (phi).


    Parameters
    ----------
    x : ndarray
    y : ndarray
    z : ndarray
    r : ndarray or None
        Computed on the fly is not provided.

    Returns
    -------
    r, theta, phi : SphericalCoordinates
        Three arrays with same shape as input.

    See Also
    --------
    Points.spherical_coordinates : corresponding function with the ``Points`` interface.

    """
    if r is None:
        r = spherical_coordinates_r(x, y, z)
    theta = spherical_coordinates_theta(z, r)
    phi = spherical_coordinates_phi(x, y)
    return SphericalCoordinates(r, theta, phi)


def rotation_matrix_x(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=float)


def rotation_matrix_y(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=float)


def rotation_matrix_z(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float)


def rotate(coords, rotation_matrix, centre=None):
    r"""
    Rotate these points given a rotation matrix and the centre.

    The rotation of a point OM (in column) is given by OM' such as:

        OM' := RotationMatrix x OM + Centre

    This function accepts multiple rotations: for example to have one rotation matrix per point of index ``idx``,
    ``rotation_matrix[idx]`` must be a 3x3 matrix.

    Parameters
    ----------
    coords : ndarray
        Coordinates to rotate. Shape: (\*shape_points, 3)
    rotation_matrix
        Shape: (3, 3) or (\*shape_points, 3, 3). Rotation matrices to apply: either one for all points or one per point.
    centre : ndarray, optional
        Centre of the rotation. This point is invariant by rotation.
        Shape: Shape: (3, 3) or (\*shape_points, 3).
        Default: centre = (0., 0., 0.)

    Returns
    -------
    rotated_points : ndarray
        Shape: (\*shape_points, 3
    """
    assert rotation_matrix.shape[-2:] == (3, 3)

    if centre is None:
        # Out[..., j] = Sum_i Rotation[...,j, i].In[...,i]
        rotated = np.einsum("...ji,...i->...j", rotation_matrix, coords)
    else:
        centre = np.asarray(centre)
        assert centre.shape[-1] == 3
        rotated = (
            np.einsum("...ji,...i->...j", rotation_matrix, coords - centre) + centre
        )
    return rotated


def to_gcs(coords_cs, bases, origins):
    r"""
    Convert the coordinates of points expressed in the basis/bases given as parameter to coordinates expressed in the
    global coordinate system.

    Warning: the bases must be **orthogonal**. No check is performed.

    Parameters
    ----------
    coords_cs : ndarray
        Shape: (\*shape_points, 3)
        Coordinates of the points in the basis.
    bases : ndarray
        Shape: (\*shape_points, 3, 3) or (3, 3)
        One or several orthogonal bases. For each basis, the coordinates of the basis vectors in the global
        coordinate system must be given row per row: i_hat, j_hat, k_hat.
    origins
        Shape: (\*shape_points, 3) or (3, 3)

    Returns
    -------
    coords_gcs : ndarray
        Shape: (\*shape_points, 3)
    """
    # OM' = Origin + OM_x * i_hat + OM_y * j_hat + OM_z * k_hat
    return np.einsum("...ij,...i->...j", bases, coords_cs) + origins


def from_gcs(points_gcs, bases, origins):
    r"""
    Convert the coordinates of points expressed in the global coordinate system to coordinates expressed
    in the basis/bases given as parameter.

    Warning: the bases must be **orthogonal**. No check is performed.

    Parameters
    ----------
    coords_gcs : ndarray
        Shape: (\*shape_points, 3)
        Coordinates of the points in the GCS.
    bases : ndarray
        Shape: (\*shape_points, 3, 3) or (3, 3)
        One or several orthogonal bases. For each basis, the coordinates of the basis vectors in the global
        coordinate system must be given row per row: i_hat, j_hat, k_hat.
    origins
        Shape: (\*shape_points, 3) or (3, 3)

    Returns
    -------
    coords_gcs : ndarray
        Shape: (\*shape_points, 3)
    """
    return np.einsum("...ji,...i->...j", bases, points_gcs - origins)


def rotation_matrix_ypr(yaw, pitch, roll):
    """Returns the rotation matrix (as a ndarray) from the yaw, pitch and roll
    in radians.

    This matrix corresponds to a intrinsic rotation around axes z, y, x respectively.

    https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    """
    return rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)


def are_points_close(points1, points2, atol=1e-8, rtol=0.0):
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
    return len(points1.shape) == len(points2.shape) and np.allclose(
        points1.coords, points2.coords, rtol=rtol, atol=atol
    )


def are_points_aligned(points, rtol=0.0, atol=1e-08):
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
    assert not np.allclose(
        AB, np.zeros(3)
    ), "this function does not work if the two first points are the same"

    # Cross product AC ^ AB, AD ^ AB, AE ^ AB...
    cross = np.cross(AM[1:, :], AB)

    return np.allclose(cross, np.zeros_like(cross), rtol=rtol, atol=atol)


def norm2(x, y, z, out=None):
    """
    Euclidean norm of a ndarray

    Parameters
    ----------
    x : ndarray
    y : ndarray
    z : ndarray
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
    x : ndarray
    y : ndarray
    z : ndarray
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

    assert np.isclose(norm2(*u_hat), 1.0)
    assert np.isclose(norm2(*v_hat), 1.0)
    assert np.isclose(norm2(*i_hat), 1.0)
    assert np.isclose(norm2(*j_hat), 1.0)

    assert np.allclose(i_hat @ j_hat, 0.0)
    assert np.allclose(u_hat @ v_hat, 0.0)

    k_hat = np.cross(i_hat, j_hat)
    w_hat = np.cross(u_hat, v_hat)

    baseDep = np.stack((i_hat, j_hat, k_hat), axis=1)
    baseArr = np.stack((u_hat, v_hat, w_hat), axis=1)

    # baseArr = M @ baseDep
    # <=> baseDep.T @ M.T = baseArr.T
    M = np.linalg.solve(baseDep.T, baseArr.T).T

    # assert np.allclose(M @ baseDep, baseArr)

    # Y = M @ (X - A) + B
    P = B - M @ A

    return M, P


def distance_pairwise(
    points1, points2, out=None, dtype=None, block_size=None, numthreads=None
):
    """
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
    """
    # Check dimensions and shapes
    try:
        (num1,) = points1.x.shape
        (num2,) = points2.x.shape
    except ValueError:
        raise InvalidDimension(
            "The dimension of the coordinates of points must be one."
        )
    if not (points1.x.shape == points1.y.shape == points1.z.shape):
        raise InvalidShape(
            "Must have: points1.x.shape == points1.y.shape == points1.z.shape"
        )
    if not (points2.x.shape == points2.y.shape == points2.z.shape):
        raise InvalidShape(
            "Must have: points2.x.shape == points2.y.shape == points2.z.shape"
        )

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=numthreads) as executor:
        for chunk1 in chunk_array((num1,), chunk_size):
            for chunk2 in chunk_array((num2,), chunk_size):
                chunk_tof = (chunk1[0], chunk2[0])

                futures.append(
                    executor.submit(
                        _distance_pairwise,
                        points1.x[chunk1],
                        points1.y[chunk1],
                        points1.z[chunk1],
                        points2.x[chunk2],
                        points2.y[chunk2],
                        points2.z[chunk2],
                        distance[chunk_tof],
                    )
                )
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()
    return distance


def is_orthonormal(basis):
    assert basis.shape == (3, 3)
    return basis.dtype.kind == "f" and np.allclose(  # must be real
        basis.T, np.linalg.inv(basis)
    )


def is_orthonormal_direct(basis):
    """
    Returns True if the basis is orthonormal and direct:
    Parameters
    ----------
    basis

    Returns
    -------

    """
    return is_orthonormal(basis) and np.allclose(
        np.cross(basis[0, :], basis[1, :]), basis[2, :]
    )


@numba.jit(nopython=True, nogil=True, cache=True)
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


def points_in_rectbox(
    x, y, z, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None
):
    """
    Returns points in the rectangular box.

    Some constraints can be ignored (unbounded box).

    Parameters
    ----------
    x : ndarray
        Coordinates x of the points to filter.
    y : ndarray
        Coordinates y of the points to filter.
    z : ndarray
        Coordinates z of the points to filter.
    xmin : float or None
    xmax : float or None
    ymin : float or None
    ymax : float or None
    zmin : float or None
    zmax : float or None

    Returns
    -------
    out : ndarray
        Array of bool whose shape is the one as x, y and z. For each entry: true if all the constraints are satisfied,
        false otherwise.

    Examples
    --------

    >>> points_in_rectbox(x, y, z, xmin=10, ymax=20, zmin=30, zmax=39)
    Returns points such as ``10 <= x`` and ``y <= 20`` and ``30 <= z <= zmax``.

    """
    if not (x.shape == y.shape == z.shape):
        raise ValueError("shape must be equal")
    out = np.ones(x.shape, dtype=bool)
    valid_ones = []
    if xmin is not None:
        valid_ones.append(xmin <= x)
    if ymin is not None:
        valid_ones.append(ymin <= y)
    if zmin is not None:
        valid_ones.append(zmin <= z)
    if xmax is not None:
        valid_ones.append(x <= xmax)
    if ymax is not None:
        valid_ones.append(y <= ymax)
    if zmax is not None:
        valid_ones.append(z <= zmax)
    for valid in valid_ones:
        np.logical_and(out, valid, out=out)
    return out


GCS = CoordinateSystem(
    origin=np.array((0.0, 0.0, 0.0)),
    i_hat=np.array((1.0, 0.0, 0.0)),
    j_hat=np.array((0.0, 1.0, 0.0)),
)


def make_contiguous_geometry(coords, numpoints, names=None, dtype=None):
    """
    Returns a list of OrientedPoints with length m which are uniquely named.
    Default naming convention defines the frontwall as the wall drawn between
    the first pair of points iff z=0.0; backwalls have constant z; sidewalls 
    have constant x; and otherwalls are anything else.

    Parameters
    ----------
    coords : ndarray[float]
        Walls drawn between each point in coords. Shape must be (m+1, 2) or
        (m+1, 3).
    numpoints : int or ndarray[int]
        Number of points which each wall is split into. If ndarray, shape must
        be (m,) a list of values for each wall individually.
    names : list[str] or None, optional
        List of names for each wall, which must be unique. If None, the default
        naming convention will be used.
    dtype : numpy.dtype, optional

    Returns
    -------
    list[OrientedPoints].

    """
    if dtype is None:
        dtype = s.FLOAT
    
    coords = np.squeeze(coords)
    if coords.shape[0] < 2:
        raise ValueError(
            "Not enough coordinates provided to draw lines for geometry."
        )
    if coords.shape[1] not in [2, 3]:
        raise ValueError("Coordinates should be 2D or 3D.")
    
    numpoints = np.squeeze(numpoints)
    if numpoints.shape[0] == 1:
        numpoints = numpoints[0] * np.ones(coords.shape[0] - 1)
    else:
        if numpoints.shape[0] != coords.shape[0] - 1:
            raise ValueError("Too many / few values of `numpoints` provided.")
        
    if names is not None:
        if len(names) != coords.shape[0] - 1:
            raise ValueError("Too many / few wall names provided.")
    else:
        bw_idx, sw_idx, ow_idx = 0, 0, 0
    
    walls = []
    for idx, (start, end) in enumerate(zip(coords[:-1, :], coords[1:, :])):
        if numpoints.shape[0] == 1:
            n = numpoints[0]
        else:
            n = numpoints[idx]
        
        if names is None:
            # Frontwall is first and has z == 0.0
            if idx == 0 and abs(start[-1]) < np.finfo(float).eps and abs(end[-1]) < np.finfo(float).eps:
                name = "frontwall"
            # Backwall has constant z.
            elif abs(start[-1] - end[-1]) < np.finfo(float).eps:
                name = "backwall_{}".format(bw_idx)
                bw_idx += 1
            # Sidewall has constant x.
            elif abs(start[0] - end[0]) < np.finfo(float).eps:
                name = "sidewall_{}".format(sw_idx)
                sw_idx += 1
            else:
                name = "otherwall_{}".format(ow_idx)
                ow_idx += 1
        
        walls.append(points_1d_wall(start, end, n, name=name, dtype=dtype))
        
    return walls


def points_1d_wall(start, end, numpoints, name=None, dtype=None):
    """
    Returns a set of regularly spaced points between `start` and `end`.
    
    Orientation will always have x_hat in the direction of wall start -> end,
    y_hat = j and z_hat = x_hat ^ j.

    Parameters
    ----------
    start : ndarray[float]
        1D array of length 2 or 3.
    end : ndarray[float]
        1D array of length 2 or 3.
    numpoints : int
    name : str or None, optional
    dtype : type or None, optional

    Returns
    -------
    OrientedPoints.

    """        
    if dtype is None:
        dtype = s.FLOAT
    start, end = np.squeeze(start), np.squeeze(end)
    if start.shape[0] == 2:
        start = np.asarray([start[0], 0.0, start[1]])
    if end.shape[0] == 2:
        end = np.asarray([end[0], 0.0, end[1]])
    
    # Make points and orientations
    points = Points.from_xyz(
        x = np.linspace(start[0], end[0], numpoints, dtype=dtype),
        y = np.linspace(start[1], end[1], numpoints, dtype=dtype),
        z = np.linspace(start[2], end[2], numpoints, dtype=dtype),
        name=name,
    )
    
    basis = CoordinateSystem(
        [0.0, 0.0, 0.0],
        (end - start) / np.linalg.norm(end - start), 
        [0.0, 1.0, 0.0],
    ).basis_matrix
    
    orientations_arr = np.broadcast_to(
        basis, (*points.shape, 3, 3)
    )
    orientations = Points(orientations_arr)
    
    return OrientedPoints(points, orientations)


def points_1d_wall_z(xmin, xmax, z, numpoints, y=0.0, name=None, is_block_above=True, dtype=None):
    """
    Returns a set of regularly spaced points between (xmin, y, z) and (xmax, y, z).

    Orientation of the point depends on `is_block_above`:
        (0., 0., 1.) if True (i.e. frontwall)
        (0., 0., -1.) if False (i.e. backwall)

    Parameters
    ----------
    xmin : float
    xmax : float
    z : float
    numpoints : int
    y : float
        Default 0
    name : str or None
    is_block_above : bool
    dtype : numpy.dtype

    Returns
    -------
    Oriented points

    """
    if dtype is None:
        dtype = s.FLOAT

    points = Points.from_xyz(
        x=np.linspace(xmin, xmax, numpoints, dtype=dtype),
        y=np.full((numpoints,), y, dtype=dtype),
        z=np.full((numpoints,), z, dtype=dtype),
        name=name,
    )

    orientations = default_orientations(points)
    # Rotate by pi radians in x-z plane if block is below.
    if not is_block_above:
        orientations = orientations.rotate([
            [ np.cos(np.pi), 0.0, np.sin(np.pi)],
            [           0.0, 1.0,           0.0],
            [-np.sin(np.pi), 0.0, np.cos(np.pi)],
        ])

    return OrientedPoints(points, orientations)


def default_orientations(points):
    """
    Returns the default orientations for unoriented points.

    Assign to each point the following orientation::

        x = (1, 0, 0)
        y = (0, 1, 0)
        z = (0, 0, 1)

    Parameters
    ----------
    points: Points

    Returns
    -------
    orientations : Points

    """
    # No need to create a full array because all values are the same: we cheat
    # using a broadcast array. This saves memory space and reduces access time.
    orientations_arr = np.broadcast_to(
        np.identity(3, dtype=points.dtype), (*points.shape, 3, 3)
    )
    orientations = Points(orientations_arr)
    return orientations


def combine_walls(walls, name=None):
    """
    Combines multiple walls into one as a simple concatenation. No checks are
    made that combination makes physical sense (i.e. walls are next to each
    other). Use at your own discretion. Duplicate points are (not) removed.

    Parameters
    ----------
    walls : list[OrientedPoints]
        
    name : str or None, optional
        New name for the wall. If None, the name of the first wall is used.

    Returns
    -------
    OrientedPoints.

    """
    if name is None:
        name = walls[0].points.name
    points = Points(np.concatenate(
        [wall.points.coords for wall in walls],
        axis=0,
    ), name=name)
    orientations = Points(np.concatenate(
        [wall.orientations.coords for wall in walls],
        axis=0,
    ))
    
    return OrientedPoints(points, orientations)


def default_oriented_points(points):
    """
    Returns OrientedPoints from Points assuming the default orientations.

    Parameters
    ----------
    points : Points

    Returns
    -------
    oriented_points : OrientedPOints

    """
    return OrientedPoints(points, default_orientations(points))


def points_from_probe(probe, name="Probe"):
    """
    Probe object to OrientedPoints (points and orientations).

    Parameters
    ----------
    probe
    name

    Returns
    -------
    OrientedPoints

    """
    points = probe.locations
    if name is not None:
        points.name = name

    orientations_arr = np.zeros((3, 3), dtype=points.dtype)
    orientations_arr[0] = probe.pcs.i_hat
    orientations_arr[1] = probe.pcs.j_hat
    orientations_arr[2] = probe.pcs.k_hat
    orientations = Points(np.broadcast_to(orientations_arr, (*points.shape, 3, 3)))

    return OrientedPoints(points, orientations)
