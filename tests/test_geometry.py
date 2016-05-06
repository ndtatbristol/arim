import math

import numpy as np
import pytest

import arim.geometry as g

DATASET_1 = dict(
    # set1:
    points1=g.Points(np.array([0, 1, 1], dtype=np.float),
                   np.array([0, 0, 0], dtype=np.float),
                   np.array([1, 0, 2], dtype=np.float),
                   'Points1'),

    # set 2:
    points2=g.Points(np.array([0, 1, 2], dtype=np.float),
                   np.array([0, -1, -2], dtype=np.float),
                   np.array([0, 0, 1], dtype=np.float),
                   'Points2'),
)


def test_are_points_aligned():
    n = 10
    z = np.arange(n, dtype=np.float64)

    theta = np.deg2rad(30)

    def make_points():
        p = g.Points(z * np.cos(theta),
                   z * np.sin(theta),
                   np.zeros((n,), dtype=np.float64))
        p.writeable = True
        return p

    points = make_points()
    are_aligned = g.are_points_aligned(points)
    assert are_aligned

    points = make_points()
    points.x[0] = 666
    are_aligned = g.are_points_aligned(points)
    assert not are_aligned

    points = make_points()
    points.y[-1] -= 0.01
    are_aligned = g.are_points_aligned(points)
    assert not are_aligned

    points = make_points()
    are_aligned = g.are_points_aligned(g.Points(*points[0:1]))
    assert are_aligned  # one point is always aligned

    points = make_points()
    are_aligned = g.are_points_aligned(g.Points(*points[0:2]))
    assert are_aligned  # two points are always aligned


def test_rotations():
    theta = np.deg2rad(30)
    identity = np.identity(3)

    # Check that rotations in one side and on the other side give the identity
    (rot_x, rot_y, rot_z) = (g.rotation_matrix_x,
                             g.rotation_matrix_y,
                             g.rotation_matrix_z)
    for rot in (rot_x, rot_y, rot_z):
        assert np.allclose(identity, rot(theta) @ rot(-theta))
        assert np.allclose(identity, rot(-theta) @ rot(theta))

    # Check the rotations of 90° are correct
    v = np.array((1, 2, 3), dtype=float)
    v_x = np.array((1, -3, 2), dtype=float)
    v_y = np.array((3, 2, -1), dtype=float)
    v_z = np.array((-2, 1, 3), dtype=float)
    phi = np.pi / 2
    assert np.allclose(v_x, rot_x(phi) @ v)
    assert np.allclose(v_y, rot_y(phi) @ v)
    assert np.allclose(v_z, rot_z(phi) @ v)


def test_rotate():
    """
    rotation of 30° around the axis y. Centre: (1, 2, 0)
    """
    # NB: the first point is the centre
    points = g.Points.from_2d_array((np.array([(1., 2., 0.), (1., 2., 3.)], dtype=float)))
    centre = np.array((1., 2., 0.))
    theta = np.deg2rad(30)
    rot = g.rotation_matrix_y(theta)

    out_points = points.rotate(rot, centre)
    assert len(points) == 2
    assert np.allclose(out_points[0], centre), "the centre must be invariant"
    assert np.allclose(out_points[1],
                       [1. + 3. * np.sin(theta), 2., 3. * np.cos(theta)])

    # this should be let all points invariant:
    centre = np.array((66., 77., 88.))
    out_points = points.rotate(np.identity(3), centre)
    assert len(points) == 2
    assert g.are_points_close(out_points, points), "all points must be invariant (no rotation)"


def test_translate():
    points = g.Points.from_2d_array((np.array([(1., 2., 0.), (1., 2., 3.)], dtype=float)))
    vector = np.array((-1., -2., -0.))

    out_points = points.translate(vector)
    assert isinstance(out_points, g.Points)
    assert len(points) == 2
    assert np.allclose(out_points[0], (0., 0., 0.))
    assert np.allclose(out_points[1], (0., 0., 3.))

    # this should be let all points invariant:
    out_points = points.translate(np.array((0., 0., 0.)))
    assert g.are_points_close(points, out_points), "all points must be invariant (no translation)"


def test_norm2():
    assert np.isclose(g.norm2(0., 0., 2.), 2.0)
    assert np.isclose(g.norm2(np.cos(0.3), np.sin(0.3), 0.), 1.0)

    x = np.array([np.cos(0.3), 1.0])
    y = np.array([np.sin(0.3), 0.0])
    z = np.array([0., 2.])
    assert np.allclose(g.norm2(x, y, z), [1.0, np.sqrt(5.0)])

    # using out:
    out = np.array(0.0, dtype=np.float)
    out1 = g.norm2(0., 0., 2., out=out)
    assert out is out1
    assert np.isclose(out, 2.0)


@pytest.mark.parametrize("shape", [(), (5,), (5, 4), (5, 6, 7)])
def test_norm2_2d(shape):
    x = np.random.uniform(size=shape)
    y = np.random.uniform(size=shape)
    z = np.zeros(shape)

    # without out
    out_2d = g.norm2_2d(x, y)
    out_3d = g.norm2(x, y, z)
    assert np.allclose(out_2d, out_3d, rtol=0.)

    # without out
    buf_2d = np.zeros(shape)
    buf_3d = np.zeros(shape)
    out_2d = g.norm2_2d(x, y, out=buf_2d)
    out_3d = g.norm2(x, y, z, out=buf_3d)
    assert buf_2d is out_2d
    assert buf_3d is out_3d
    assert np.allclose(out_2d, out_3d, rtol=0.)


_ISOMETRY_2D_DATA = [
    (np.array([0., 0.]), np.array([0., 1.]), np.array([2., 0.]), np.array([2., 1.])),
    (np.array([66., 0.]), np.array([66., 1.]), np.array([77., 0.]), np.array([77 + np.cos(30), np.sin(30)])),
    (np.array([66., 0.]), np.array([66., 1.]), np.array([77., 0.]), np.array([77 + np.cos(-30), np.sin(-30)])),
]


@pytest.mark.parametrize("points", _ISOMETRY_2D_DATA)
def test_direct_isometry_2d(points):
    A, B, Ap, Bp = points

    M, P = g.direct_isometry_2d(A, B, Ap, Bp)
    assert np.allclose(M @ A + P, Ap)
    assert np.allclose(M @ B + P, Bp)

    # Check barycentres
    k1 = 0.3
    k2 = 0.7
    bary = k1 * A + k2 * B
    assert np.allclose(M @ bary + P, k1 * Ap + k2 * Bp)

    # Do we preserve the orientation?
    rot90 = np.array([[0., -1.], [1., 0.]])
    C = rot90 @ (B - A) + A
    Cp2 = rot90 @ (Bp - Ap) + Ap
    assert np.allclose(M @ C + P, Cp2)


_ISOMETRY_3D_DATA = [
    (np.asarray((10., 0., 0.)), np.asarray((1., 0., 0)), np.asarray((0., 1., 0.)),
     np.asarray((0., 0., 66.)), np.asarray((np.cos(0.3), np.sin(0.3), 0)), np.asarray((-np.sin(0.3), np.cos(0.3), 0))),

    (np.asarray((10., 0., 0.)), np.asarray((1., 0., 0)), np.asarray((0., 0., 1.)),
     np.asarray((0., 0., 66.)), np.asarray((np.cos(0.3), np.sin(0.3), 0)), np.asarray((-np.sin(0.3), np.cos(0.3), 0))),

    (np.asarray((10., 11., 12.)), np.asarray((0., 1., 0)), np.asarray((0., 0., 1.)),
     np.asarray((22., 21., 20.)), np.asarray((np.cos(0.3), np.sin(0.3), 0)),
     np.asarray((-np.sin(0.3), np.cos(0.3), 0))),
]


@pytest.mark.parametrize("points", _ISOMETRY_3D_DATA)
def test_direct_isometry_3d(points):
    A, i_hat, j_hat, B, u_hat, v_hat = points

    k_hat = np.cross(i_hat, j_hat)
    w_hat = np.cross(u_hat, v_hat)

    M, P = g.direct_isometry_3d(A, i_hat, j_hat, B, u_hat, v_hat)

    # M is orthogonal
    assert np.allclose(M @ M.T, np.identity(3))

    assert np.allclose(M @ i_hat, u_hat)
    assert np.allclose(M @ j_hat, v_hat)
    assert np.allclose(M @ np.cross(i_hat, j_hat), np.cross(M @ i_hat, M @ j_hat))

    k1, k2, k3 = np.random.uniform(size=3)
    Q1 = A + k1 * i_hat + k2 * j_hat + k3 * k_hat
    Q2 = B + k1 * u_hat + k2 * v_hat + k3 * w_hat

    assert np.allclose(M @ Q1 + P, Q2)


def test_grid():
    xmin = -10e-3
    xmax = 10e-3
    dx = 1e-3

    ymin = 3e-3
    ymax = 3e-3
    dy = 1e-3

    zmin = 0
    zmax = -10e-3
    dz = 1e-3

    grid = g.Grid(xmin, xmax, ymin, ymax, zmin, zmax, (dx, dy, dz))
    assert len(grid.x) == 21
    assert grid.xmin == xmin
    assert grid.xmax == xmax
    assert grid.dx == dx

    assert len(grid.y) == 1
    assert grid.ymin == ymin
    assert grid.ymax == ymax
    assert grid.dy is None

    assert len(grid.z) == 11
    assert grid.zmin == zmin
    assert grid.zmax == zmax
    assert grid.dz == -dz

    assert grid.numpoints == 21 * 1 * 11

    points = grid.as_points
    points2 = grid.as_points
    assert points is points2
    assert isinstance(points, g.Points)
    assert len(points) == grid.numpoints


class TestPoints():
    def test_points(self):
        x = np.arange(5, dtype=np.float)
        y = np.arange(5, dtype=np.float) / 2
        z = np.arange(5, dtype=np.float) / 3
        points = g.Points(x, y, z, 'MyTestPoint')
        unnamed_points = g.Points(x, y, z)

        assert points.x is x
        assert points.y is y
        assert points.z is z
        str(points)
        repr(points)
        str(unnamed_points)
        repr(unnamed_points)
        assert len(points) == 5

        # check __getitem__
        for i in range(len(points)):
            xout, yout, zout = points[i]
            assert np.isclose(xout, x[i])
            assert np.isclose(yout, y[i])
            assert np.isclose(zout, z[i])

        # check iterable interface
        for (i, (xout, yout, zout)) in enumerate(points):
            xout, yout, zout = points[i]
            assert np.isclose(xout, x[i])
            assert np.isclose(yout, y[i])
            assert np.isclose(zout, z[i])

        # check method allclose
        assert points.allclose(points)

    def test_2d_arrays(self):
        arr = np.random.uniform(size=(12, 3))
        points = g.Points.from_2d_array(arr)
        points_arr = points.to_2d_array()
        assert np.allclose(arr, points_arr)

    def test_writeable(self):
        arr = np.random.uniform(size=(12, 3))
        points = g.Points.from_2d_array(arr)

        points.writeable = True
        assert points.writeable
        points.x[0] = 666.
        assert points.x[0] == 666.

        points.writeable = False
        assert not points.writeable
        with pytest.raises(ValueError):
            points.x[1] = 999.

    def test_from_one(self):
        p = (1., 2., 3.)
        points = g.Points.from_one(p)
        assert np.allclose(p, points[0])

    def test_spherical_coordinates(self):
        """
        Cf. https://commons.wikimedia.org/wiki/File:3D_Spherical.svg on 2016-03-16
        """
        # x, y, z
        points = g.Points.from_2d_array(np.array([[5., 0., 0.],
                                                [-5., 0., 0.],
                                                [0., 6., 0.],
                                                [0., -6., 0.],
                                                [0., 0., 7.],
                                                [0., 0., -7.]]))
        out = points.spherical_coordinates()

        # r, theta, phi
        expected = np.array([[5., np.pi / 2, 0.],
                             [5., np.pi / 2, np.pi],
                             [6., np.pi / 2, np.pi / 2],
                             [6., np.pi / 2, -np.pi / 2],
                             [7., 0., 0.],
                             [7., np.pi, .0],
                             ])

        assert np.allclose(out.r, expected[:, 0])
        assert np.allclose(out.theta, expected[:, 1])
        assert np.allclose(out.phi, expected[:, 2])


class TestCoordinateSystem:
    theta = np.deg2rad(30)

    def test_isclose(self, cs):
        assert cs.isclose(cs)
        assert not cs.isclose(g.GCS)

    @pytest.fixture()
    def cs(self):
        """
        Canonical cartesian coordinate system, with a rotation of theta around Oz.

        """
        i_hat = [np.cos(self.theta), np.sin(self.theta), 0.]
        j_hat = [-np.sin(self.theta), np.cos(self.theta), 0.]
        origin = [66., 77., 88.]
        return g.CoordinateSystem(origin, i_hat, j_hat)

    def test_cs(self, cs):
        assert np.allclose(cs.k_hat, [0., 0., 1.])

    def test_coordinate_system(self):
        i_hat = [1., 0., 0.]
        j_hat = [0., 1., 0.]
        origin = [66., 77., 88.]
        cs = g.CoordinateSystem(origin, i_hat, j_hat)

        assert np.allclose(cs.k_hat, [0., 0., 1.])

    def test_translate(self, cs):
        cs_expect = g.CoordinateSystem((0., 0., 0.), cs.i_hat, cs.j_hat)
        cs_out = cs.translate(-cs.origin)
        assert cs_expect.isclose(cs_out)

    def test_rotate(self, cs):
        # reverse the rotation that gave the coordinate system:
        rot = g.rotation_matrix_z(-self.theta)
        cs_expect = g.CoordinateSystem(cs.origin, (1., 0., 0.), (0., 1., 0.))
        cs_out = cs.rotate(rot, centre=cs.origin)
        assert cs_expect.isclose(cs_out)

        # Second case, with centre O:
        rot = g.rotation_matrix_z(np.pi)
        cs_expect = g.CoordinateSystem([-cs.origin[0], -cs.origin[1], +cs.origin[2]], -cs.i_hat, -cs.j_hat)
        cs_out = cs.rotate(rot, centre=None)
        assert cs_expect.isclose(cs_out)

    def test_convert_gcs(self):
        points = g.Points.from_2d_array(np.arange(12, dtype=np.float).reshape((4, 3)))

        gcs = g.GCS

        # conversion of GCS to GCS: must let the points invariant
        points_out = gcs.convert_from_gcs(points)
        assert g.are_points_close(points, points_out)

        points_out = gcs.convert_to_gcs(points)
        assert g.are_points_close(points, points_out)

    def test_convert_gcs2(self, cs):
        k1 = (0.5, 0.6, 0.7)
        k2 = (2., -0.6, 0.9)
        points_pcs = g.Points.from_2d_array(np.array((k1, k2)))
        p1 = cs.origin + k1[0] * cs.i_hat + k1[1] * cs.j_hat + k1[2] * cs.k_hat
        p2 = cs.origin + k2[0] * cs.i_hat + k2[1] * cs.j_hat + k2[2] * cs.k_hat
        points_gcs = g.Points.from_2d_array(np.array((p1, p2)))

        # Test GCS to PCS
        points_pcs_out = cs.convert_from_gcs(points_gcs)
        assert g.are_points_close(points_pcs_out, points_pcs)

        # Test PCS to GCS
        points_gcs_out = cs.convert_to_gcs(points_pcs)
        assert g.are_points_close(points_gcs_out, points_gcs)

    def test_from_gcs_pairwise(self, cs):
        assert isinstance(cs, g.CoordinateSystem)
        origins = g.Points(np.array([0., 0., 0.]),
                         np.array([0., 1., 2.]),
                         np.array([0., 3., 4.]))
        points_gcs = g.Points.from_2d_array(np.random.uniform(size=(10, 3)))

        # This is the tested function.
        x, y, z = cs.convert_from_gcs_pairwise(points_gcs, origins)
        assert x.shape == (len(points_gcs), len(origins))
        assert x.shape == y.shape == z.shape

        # Check the result are the same as if we convert the points "by hand".
        for (j, origin) in enumerate(origins):
            # Points in the j-th derived CS, computed by 'convert_from_gcs_pairwise':
            out_points = g.Points(x[..., j].copy(), y[..., j].copy(), z[..., j].copy())

            # Points in the j-th derived CS, computed by the regular 'convert_from_gcs':
            origin_gcs = cs.convert_to_gcs(g.Points.from_one(origin))[0]
            cs_derived = g.CoordinateSystem(origin_gcs, cs.i_hat, cs.j_hat)
            expected_points = cs_derived.convert_from_gcs(points_gcs)

            assert g.are_points_close(out_points, expected_points)

    def test_isometry_preserves_distance(self, cs):
        """
        ultimate test
        """
        points = g.Points.from_2d_array(np.arange(15, dtype=np.float).reshape((5, 3)))
        points_cs_init = cs.convert_from_gcs(points)

        # define a nasty isometry:
        centre = np.array((1.1, 1.2, 1.3))
        rotation = g.rotation_matrix_ypr(0.5, -0.6, 0.7)
        translation = np.array((66., -77., 0.))

        # move the coordinate system and the points:
        points_final = points.rotate(rotation, centre).translate(translation)
        cs_final = cs.rotate(rotation, centre).translate(translation)

        # check that the points are still the same
        points_cs_final = cs_final.convert_from_gcs(points_final)
        assert g.are_points_close(points_cs_init, points_cs_final)


EUCLIDEAN_DISTANCE_1 = None  # used for cache


@pytest.fixture(scope="module", params=[dict(block_size=None, numthreads=None),
                                        dict(block_size=1, numthreads=1),
                                        dict(block_size=1, numthreads=4),
                                        dict(block_size=10, numthreads=4), ])
def distance(request):
    """
    Compute the Euclidean distance using our package. Check that multithreading has no
    effect on the result.
    """
    global DATASET_1
    kwargs = request.param
    kwargs.update(DATASET_1)

    return g.distance_pairwise(**kwargs)


def mock_euclidean_distance(points1, points2):
    """
    naive implementation of computation of euclidean distance
    Compute once and cache result
    """
    global EUCLIDEAN_DISTANCE_1
    if EUCLIDEAN_DISTANCE_1 is None:
        distance = np.full((len(points1), len(points2)), 0, dtype=np.float)
        for i in range(len(points1)):
            for j in range(len(points2)):
                distance[i, j] = math.sqrt(((points1.x[i] - points2.x[j]) ** 2 +
                                            (points1.y[i] - points2.y[j]) ** 2 +
                                            (points1.z[i] - points2.z[j]) ** 2))
        EUCLIDEAN_DISTANCE_1 = distance
    return EUCLIDEAN_DISTANCE_1


def test_euclidean_distance(distance):
    mock_distance = mock_euclidean_distance(**DATASET_1)

    assert np.allclose(distance, mock_distance)


def test_euclidean_distance_advanced():
    mock_distance = mock_euclidean_distance(**DATASET_1)

    dtype = np.complex128  # weird though

    global DATASET_1
    distance = g.distance_pairwise(**DATASET_1, dtype=dtype)
    assert distance.dtype == dtype
    assert np.allclose(np.real(distance), mock_distance)

    distance = np.full((len(DATASET_1['points1']), len(DATASET_1['points2'])), 0.0, dtype=np.float)
    g.distance_pairwise(**DATASET_1, out=distance)  # write inplace
    assert np.allclose(distance, mock_distance)


class TestGeometryHelper:
    theta = np.deg2rad(30)

    @pytest.fixture
    def points1_pcs(self):
        return g.Points(x=np.arange(16, dtype=np.float),
                        y=np.zeros(16, dtype=np.float),
                        z=np.zeros(16, dtype=np.float),
                        name='Probe PCS')

    @pytest.fixture
    def pcs(self):
        return g.GCS.rotate(g.rotation_matrix_y(self.theta))

    @pytest.fixture
    def points1_gcs(self, points1_pcs, pcs):
        assert isinstance(pcs, g.CoordinateSystem)
        return pcs.convert_from_gcs(points1_pcs)

    @pytest.fixture
    def points2_gcs(self):
        return g.Points(x=np.linspace(-10., 10., 21),
                        y=np.zeros(21, dtype=np.float),
                        z=np.zeros(21, dtype=np.float),
                        name='Probe PCS')

    @pytest.fixture
    def geometry_helper_cache(self, points1_gcs, points2_gcs, pcs):
        return g.GeometryHelper(points1_gcs, points2_gcs, pcs, use_cache=True)

    @pytest.fixture
    def geometry_helper_nocache(self, points1_gcs, points2_gcs, pcs):
        return g.GeometryHelper(points1_gcs, points2_gcs, pcs, use_cache=False)

    @pytest.fixture(params=[geometry_helper_cache, geometry_helper_nocache])
    def geometry_helper(geom):
        return geom

    def test_basics(self, geometry_helper, points1_pcs):
        assert g.are_points_close(points1_pcs, geometry_helper.points1_pcs())

    def test_cache(self, geometry_helper_cache):
        geom = geometry_helper_cache
        dist = geom.distance_pairwise()
        dist2 = geom.distance_pairwise()
        assert dist is dist2
        geom.clear()

        dist3 = geom.distance_pairwise()
        assert dist is not dist3

        out = geom.points2_to_pcs_pairwise()
        out = geom.points2_to_pcs_pairwise_spherical()

        str(geom)
