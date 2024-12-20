import itertools
import math

import numpy as np
import pytest

import arim.geometry as g

DATASET_1 = dict(
    # set1:
    points1=g.Points.from_xyz(
        np.array([0, 1, 1], dtype=np.float64),
        np.array([0, 0, 0], dtype=np.float64),
        np.array([1, 0, 2], dtype=np.float64),
        "Points1",
    ),
    # set 2:
    points2=g.Points.from_xyz(
        np.array([0, 1, 2], dtype=np.float64),
        np.array([0, -1, -2], dtype=np.float64),
        np.array([0, 0, 1], dtype=np.float64),
        "Points2",
    ),
)


def test_are_points_aligned():
    n = 10
    z = np.arange(n, dtype=np.float64)

    theta = np.deg2rad(30)

    def make_points():
        p = g.Points.from_xyz(
            z * np.cos(theta), z * np.sin(theta), np.zeros((n,), dtype=np.float64)
        )
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
    are_aligned = g.are_points_aligned(g.Points(points[0:1]))
    assert are_aligned  # one point is always aligned

    points = make_points()
    are_aligned = g.are_points_aligned(g.Points(points[0:2]))
    assert are_aligned  # two points are always aligned


def test_rotations():
    """
    Test rotation_matrix_x, rotation_matrix_y and rotation_matrix_z
    """
    theta = np.deg2rad(30)
    identity = np.identity(3)

    # Check that rotations in one side and on the other side give the identity
    (rot_x, rot_y, rot_z) = (
        g.rotation_matrix_x,
        g.rotation_matrix_y,
        g.rotation_matrix_z,
    )
    for rot in (rot_x, rot_y, rot_z):
        assert np.allclose(identity, rot(theta) @ rot(-theta))
        assert np.allclose(identity, rot(-theta) @ rot(theta))

    # Check the rotations of 90° are correct
    v = np.array((1, 2, 3), dtype=np.float64)
    v_x = np.array((1, -3, 2), dtype=np.float64)
    v_y = np.array((3, 2, -1), dtype=np.float64)
    v_z = np.array((-2, 1, 3), dtype=np.float64)
    phi = np.pi / 2
    assert np.allclose(v_x, rot_x(phi) @ v)
    assert np.allclose(v_y, rot_y(phi) @ v)
    assert np.allclose(v_z, rot_z(phi) @ v)


def test_norm2():
    assert np.isclose(g.norm2(0.0, 0.0, 2.0), 2.0)
    assert np.isclose(g.norm2(np.cos(0.3), np.sin(0.3), 0.0), 1.0)

    x = np.array([np.cos(0.3), 1.0])
    y = np.array([np.sin(0.3), 0.0])
    z = np.array([0.0, 2.0])
    assert np.allclose(g.norm2(x, y, z), [1.0, np.sqrt(5.0)])

    # using out:
    out = np.array(0.0, dtype=np.float64)
    out1 = g.norm2(0.0, 0.0, 2.0, out=out)
    assert out is out1
    assert np.isclose(out, 2.0)


def test_is_orthonormal():
    assert g.is_orthonormal(np.identity(3))
    assert g.is_orthonormal_direct(np.identity(3))

    assert not (g.is_orthonormal(2.0 * np.identity(3)))
    assert not (g.is_orthonormal_direct(2.0 * np.identity(3)))

    a = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert g.is_orthonormal(a)
    assert g.is_orthonormal_direct(a)

    a = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert g.is_orthonormal(a)
    assert not (g.is_orthonormal_direct(a))


@pytest.mark.parametrize("shape", [(), (5,), (5, 4), (5, 6, 7)])
def test_norm2_2d(shape):
    rng = np.random.default_rng(seed=1234 + sum(shape))
    x = rng.uniform(size=shape)
    y = rng.uniform(size=shape)
    z = np.zeros(shape)

    # without out
    out_2d = g.norm2_2d(x, y)
    out_3d = g.norm2(x, y, z)
    assert np.allclose(out_2d, out_3d, rtol=0.0)

    # without out
    buf_2d = np.zeros(shape)
    buf_3d = np.zeros(shape)
    out_2d = g.norm2_2d(x, y, out=buf_2d)
    out_3d = g.norm2(x, y, z, out=buf_3d)
    assert buf_2d is out_2d
    assert buf_3d is out_3d
    assert np.allclose(out_2d, out_3d, rtol=0.0)


_ISOMETRY_2D_DATA = [
    (
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([2.0, 0.0]),
        np.array([2.0, 1.0]),
    ),
    (
        np.array([66.0, 0.0]),
        np.array([66.0, 1.0]),
        np.array([77.0, 0.0]),
        np.array([77 + np.cos(30), np.sin(30)]),
    ),
    (
        np.array([66.0, 0.0]),
        np.array([66.0, 1.0]),
        np.array([77.0, 0.0]),
        np.array([77 + np.cos(-30), np.sin(-30)]),
    ),
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
    rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    C = rot90 @ (B - A) + A
    Cp2 = rot90 @ (Bp - Ap) + Ap
    assert np.allclose(M @ C + P, Cp2)


_ISOMETRY_3D_DATA = [
    (
        np.asarray((10.0, 0.0, 0.0)),
        np.asarray((1.0, 0.0, 0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 66.0)),
        np.asarray((np.cos(0.3), np.sin(0.3), 0)),
        np.asarray((-np.sin(0.3), np.cos(0.3), 0)),
    ),
    (
        np.asarray((10.0, 0.0, 0.0)),
        np.asarray((1.0, 0.0, 0)),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((0.0, 0.0, 66.0)),
        np.asarray((np.cos(0.3), np.sin(0.3), 0)),
        np.asarray((-np.sin(0.3), np.cos(0.3), 0)),
    ),
    (
        np.asarray((10.0, 11.0, 12.0)),
        np.asarray((0.0, 1.0, 0)),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((22.0, 21.0, 20.0)),
        np.asarray((np.cos(0.3), np.sin(0.3), 0)),
        np.asarray((-np.sin(0.3), np.cos(0.3), 0)),
    ),
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

    rng = np.random.default_rng(seed=0)
    k1, k2, k3 = rng.uniform(size=3)
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

    zmin = -10e-3
    zmax = 0
    dz = 1e-3

    grid = g.Grid(xmin, xmax, ymin, ymax, zmin, zmax, (dx, dy, dz))
    assert grid.shape == (grid.numx, grid.numy, grid.numz)

    assert len(grid.xvect) == 21
    assert grid.xmin == xmin
    assert grid.xmax == xmax
    assert np.isclose(grid.dx, dx)
    assert grid.x.shape == (grid.numx, grid.numy, grid.numz)

    assert len(grid.yvect) == 1
    assert grid.ymin == ymin
    assert grid.ymax == ymax
    assert grid.dy is None
    assert grid.y.shape == (grid.numx, grid.numy, grid.numz)

    assert len(grid.zvect) == 11
    assert grid.zmin == zmin
    assert grid.zmax == zmax
    assert np.isclose(grid.dz, dz)
    assert grid.z.shape == (grid.numx, grid.numy, grid.numz)

    assert grid.numpoints == 21 * 1 * 11

    points = grid.to_1d_points()
    assert isinstance(points, g.Points)
    assert len(points) == grid.numpoints

    grid_p = grid.to_oriented_points()
    assert grid_p.points.shape == (grid.numpoints,)
    assert grid_p.orientations.shape == (grid.numpoints, 3)

    match = grid.points_in_rectbox(xmax=-9.5e-3)
    assert match.sum() == 1 * grid.numy * grid.numz


@pytest.mark.parametrize("pixel_size", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_grid_centred(pixel_size):
    centre_x = 15.0
    centre_y = 0.0
    centre_z = 10.0
    size_x = 5.0
    size_y = 0.0
    size_z = 10.0

    grid = g.Grid.grid_centred_at_point(
        centre_x, centre_y, centre_z, size_x, size_y, size_z, pixel_size
    )
    np.testing.assert_allclose(grid.xmin, 12.5)
    np.testing.assert_allclose(grid.xmax, 17.5)
    np.testing.assert_allclose(grid.yvect, [0.0])
    np.testing.assert_allclose(grid.zmin, 5.0)
    np.testing.assert_allclose(grid.zmax, 15.0)
    assert grid.dy is None
    assert grid.numx % 2 == 1
    assert grid.numy == 1
    assert grid.numz % 2 == 1
    idx = grid.closest_point(centre_x, centre_y, centre_z)
    np.testing.assert_allclose(
        [grid.x.flat[idx], grid.y.flat[idx], grid.z.flat[idx]],
        [centre_x, centre_y, centre_z],
    )


@pytest.mark.parametrize("pixel_size", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_grid_clone(pixel_size):
    xmin = -10e-3
    xmax = 10e-3

    ymin = 3e-3
    ymax = 3e-3

    zmin = -10e-3
    zmax = 0

    orig_grid = g.Grid(xmin, xmax, ymin, ymax, zmin, zmax, pixel_size)
    grid = g.Grid(
        orig_grid.xmin,
        orig_grid.xmax,
        orig_grid.ymin,
        orig_grid.ymax,
        orig_grid.zmin,
        orig_grid.zmax,
        (orig_grid.dx, orig_grid.dy, orig_grid.dz),
    )
    assert orig_grid.numx == grid.numx
    assert orig_grid.numy == grid.numy
    assert orig_grid.numz == grid.numz


# shape, name, size
points_parameters = [
    ((), "TestPoints", 1),
    ((5,), "TestPoints", 5),
    ((5, 6), "TestPoints", 30),
    ((), None, 1),
    ((5,), None, 5),
    ((5, 6), None, 30),
]
points_parameters_ids = [
    "one_named",
    "vect_named",
    "matrix_named",
    "one_unnamed",
    "vect_unnamed",
    "matrix_unnamed",
]


def test_aspoints():
    points = g.aspoints([1.0, 2.0, 3.0])
    assert points.shape == ()
    assert points.x == 1.0
    assert points.y == 2.0
    assert points.z == 3.0


class TestPoints:
    @pytest.fixture(scope="class", params=points_parameters, ids=points_parameters_ids)
    def points(self, request):
        """fixture points"""
        shape, name, size = request.param
        points, _ = self.make_points(shape, name, size)
        return points

    @staticmethod
    def make_points(shape, name, size):
        coords = np.arange(size * 3, dtype=np.float64).reshape((*shape, 3))
        raw_coords = np.copy(coords)

        points = g.Points(coords, name)
        return points, raw_coords

    @pytest.mark.parametrize(
        "shape, name, size", points_parameters, ids=points_parameters_ids
    )
    def test_points_basics(self, shape, name, size):
        """
        Test basics attributes/properties of Points.
        """
        points, raw_coords = self.make_points(shape, name, size)

        assert points.shape == shape
        assert points.ndim == len(shape)
        assert points.x.shape == shape
        assert points.y.shape == shape
        assert points.z.shape == shape
        np.testing.assert_allclose(points.coords, raw_coords)
        np.testing.assert_allclose(points.x, raw_coords[..., 0])
        np.testing.assert_allclose(points.y, raw_coords[..., 1])
        np.testing.assert_allclose(points.z, raw_coords[..., 2])

        assert points.size == size
        assert points.name == name

        if len(shape) == 0:
            with pytest.raises(TypeError):
                len(points)
        else:
            assert len(points) == shape[0]

        for idx, p in points.enumerate():
            np.testing.assert_allclose(p, (points.x[idx], points.y[idx], points.z[idx]))

            # test __getitem__:
            np.testing.assert_allclose(
                points[idx], (points.x[idx], points.y[idx], points.z[idx])
            )

        # test iterator
        for idx, p in zip(np.ndindex(shape), points):
            np.testing.assert_allclose(p, (points.x[idx], points.y[idx], points.z[idx]))
        assert len(list(iter(points))) == size

        # Test hashability
        d = {points: "toto"}
        assert d

        # Test str/rep
        str(points)
        repr(points)

    @pytest.mark.parametrize(
        "shape, name, size", points_parameters, ids=points_parameters_ids
    )
    def test_points_from_xyz(self, shape, name, size):
        points, raw_coords = self.make_points(shape, name, size)

        points2 = g.Points.from_xyz(points.x, points.y, points.z)
        np.testing.assert_allclose(points2.coords, points.coords)
        np.testing.assert_allclose(points2.x, points.x)
        np.testing.assert_allclose(points2.y, points.y)
        np.testing.assert_allclose(points2.z, points.z)

    def test_spherical_coordinates(self):
        """
        Cf. https://commons.wikimedia.org/wiki/File:3D_Spherical.svg on 2016-03-16
        """
        # x, y, z
        points = g.Points(
            np.array(
                [
                    [5.0, 0.0, 0.0],
                    [-5.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, -6.0, 0.0],
                    [0.0, 0.0, 7.0],
                    [0.0, 0.0, -7.0],
                ]
            )
        )
        out = points.spherical_coordinates()

        # r, theta, phi
        expected = np.array(
            [
                [5.0, np.pi / 2, 0.0],
                [5.0, np.pi / 2, np.pi],
                [6.0, np.pi / 2, np.pi / 2],
                [6.0, np.pi / 2, -np.pi / 2],
                [7.0, 0.0, 0.0],
                [7.0, np.pi, 0.0],
            ]
        )

        assert np.allclose(out.r, expected[:, 0])
        assert np.allclose(out.theta, expected[:, 1])
        assert np.allclose(out.phi, expected[:, 2])

    @pytest.mark.parametrize(
        "shape, name, size", points_parameters, ids=points_parameters_ids
    )
    def test_are_points_close(self, shape, name, size):
        """test geometry.are_points.close for different shapes"""
        points, raw_coords = self.make_points(shape, name, size)

        points2 = g.Points(raw_coords)

        assert g.are_points_close(points, points)
        assert g.are_points_close(points, points2)

        try:
            points2.x[0] += 666.0
        except IndexError:
            points2.x[()] += 666.0  # special case ndim=0
        assert not (g.are_points_close(points, points2))

        # use different shape
        assert not (g.are_points_close(points, g.Points([1, 2, 3])))

    @pytest.mark.parametrize(
        "shape, shape_directions",
        [[(), ()], [(5,), ()], [(5,), (5,)], [(5, 6), ()], [(5, 6), (5, 6)]],
        ids=["one_one", "vect_one", "vect_vect", "mat_one", "mat_mat"],
    )
    def test_points_translate(self, shape, shape_directions):
        """
        Test Points.translate for different shapes of Points and directions.

        """
        rng = np.random.default_rng(seed=100)
        coords = rng.uniform(size=((*shape, 3)))
        directions = rng.uniform(size=((*shape_directions, 3)))

        points = g.Points(coords.copy())

        out_points = points.translate(directions.copy())

        assert out_points.shape == shape, "the translated points have a different shape"
        assert isinstance(out_points, g.Points)

        idx_set = set()
        for (idx, in_p), out_p, idx_direction in itertools.zip_longest(
            points.enumerate(), out_points, np.ndindex(shape_directions), fillvalue=()
        ):
            idx_set.add(idx)
            assert in_p.shape == out_p.shape == directions[idx_direction].shape == (3,)
            expected = in_p + directions[idx_direction]
            np.testing.assert_allclose(
                out_p,
                expected,
                err_msg="translation failed for idx={} and idx_direction={}".format(
                    idx, idx_direction
                ),
            )
        assert len(idx_set) == points.size, "The test does not check all points"

        # this should be let all points invariant:
        out_points = points.translate(np.array((0.0, 0.0, 0.0)))
        assert g.are_points_close(
            points, out_points
        ), "all points must be invariant (no translation)"

    def test_norm2(self, points):
        """test Points.norm2"""
        norm = points.norm2()
        assert norm.shape == points.shape
        for idx, p in points.enumerate():
            x, y, z = p
            np.testing.assert_allclose(norm[idx], math.sqrt(x * x + y * y + z * z))

    def test_rotate_one_rotation(self, points):
        """Test Points.rotate() with one rotation for all points."""
        rot = g.rotation_matrix_ypr(*np.deg2rad([10, 20, 30]))

        # Case 1a: centre is None
        out_points = points.rotate(rot, centre=None)
        assert out_points.shape == points.shape

        for (idx, p_in), p_out in zip(points.enumerate(), out_points):
            expected = rot @ p_in
            np.testing.assert_allclose(
                p_out, expected, err_msg=f"rotation failed for idx={idx}"
            )

        # Case 1b: centre is [0., 0., 0.] (should give the same answers)
        out_points_b = points.rotate(rot, centre=np.array((0.0, 0.0, 0.0)))
        assert g.are_points_close(out_points, out_points_b)

        # Case 2: centre is not trivial
        centre = np.array((66.0, 77.0, 88.0))
        out_points = points.rotate(rot, centre=centre)
        assert out_points.shape == points.shape

        for (idx, p_in), p_out in zip(points.enumerate(), out_points):
            expected = rot @ (p_in - centre) + centre
            np.testing.assert_allclose(
                p_out, expected, err_msg=f"rotation failed for idx={idx}"
            )

    def test_rotate_multiple_rotations(self, points):
        """Test Points.rotate() with as many rotations as points

        Use multiple centres too.
        """
        # Create rotation matrix:
        rot_shape = (*points.shape, 3, 3)
        rot = np.zeros(rot_shape, dtype=points.dtype)
        rng = np.random.default_rng(seed=1000)
        for idx in np.ndindex(points.shape):
            rot[idx] = g.rotation_matrix_ypr(
                *rng.uniform(-2 * np.pi, 2 * np.pi, size=(3,))
            )

        # Case 1a: centre is None
        out_points = points.rotate(rot, centre=None)
        assert out_points.shape == points.shape

        for (idx, p_in), p_out in zip(points.enumerate(), out_points):
            expected = rot[idx] @ p_in
            np.testing.assert_allclose(
                p_out, expected, err_msg=f"rotation failed for idx={idx}"
            )

        # Case 1b: centre is [0., 0., 0.] (should give the same answers)
        out_points_b = points.rotate(rot, centre=np.array((0.0, 0.0, 0.0)))
        assert g.are_points_close(out_points, out_points_b)

        # Case 2: centre is not trivial
        centre = rng.uniform(size=(*points.shape, 3))
        out_points = points.rotate(rot, centre=centre)
        assert out_points.shape == points.shape

        # Check point per point:
        for (idx, p_in), p_out in zip(points.enumerate(), out_points):
            expected = rot[idx] @ (p_in - centre[idx]) + centre[idx]
            np.testing.assert_allclose(
                p_out, expected, err_msg=f"rotation failed for idx={idx}"
            )

    @pytest.mark.parametrize(
        ("points_shape, bases_shape, origins_shape"),
        [
            [(), (), ()],
            [(5,), (), ()],
            [(5,), (), (5,)],
            [(5,), (5,), ()],
            [(5,), (5,), (5,)],
            [(5, 6), (), ()],
            [(5, 6), (), (5, 6)],
            [(5, 6), (5, 6), ()],
            [(5, 6), (5, 6), (5, 6)],
        ],
        ids=[
            "one_one_one",
            "vect_one_one",
            "vect_one_vect",
            "vect_vect_one",
            "vect_vect_vect",
            "mat_one_one",
            "mat_one_mat",
            "mat_mat_one",
            "mat_mat_mat",
        ],
    )
    def test_convert_coordinates_one_basis(
        self, points_shape, bases_shape, origins_shape
    ):
        """
        Test the function to_gcs and from_gcs using various combinations of points, basis and origins.
        """
        # Generate points:
        rng = np.random.default_rng(seed=0)
        points_gcs = g.Points(rng.uniform(-10, 10.0, size=(*points_shape, 3)))

        # Generate bases:
        bases = np.zeros((*bases_shape, 3, 3))
        for idx in np.ndindex(bases_shape):
            bases[idx] = g.rotation_matrix_ypr(
                *rng.uniform(-2 * np.pi, 2 * np.pi, size=(3,))
            )
            assert g.is_orthonormal_direct(bases[idx])

        # Generate origins:
        origins = rng.uniform(-100, 100, size=(*origins_shape, 3))

        # PART 1/ GCS TO CS ===================================================================
        out_points_cs = points_gcs.from_gcs(bases, origins)
        self.compare_coordinates(points_gcs, out_points_cs, bases, origins)

        # PART 2/ CS TO GCS ===================================================================
        out_points_gcs = out_points_cs.to_gcs(bases, origins)
        self.compare_coordinates(out_points_gcs, out_points_cs, bases, origins)
        assert g.are_points_close(points_gcs, out_points_gcs)

    def compare_coordinates(self, coords_gcs, coords_cs, bases, origins):
        assert coords_gcs.shape == coords_cs.shape
        for (idx, p_gcs), p_cs in zip(coords_gcs.enumerate(), coords_cs):
            try:
                # Force a IndexError if idx has not a valid dim
                origin = origins[(*idx, slice(None))]
            except IndexError:
                origin = origins[()]
            try:
                # Force a IndexError if idx has not a valid dim
                basis = bases[(*idx, slice(None), slice(None))]
            except IndexError:
                basis = bases[()]
            assert origin.shape == (3,), "the test is badly written"
            assert basis.shape == (3, 3), "the test is badly written"
            assert p_gcs.shape == (3,), "the test is badly written"
            assert p_cs.shape == (3,), "the test is badly written"

            i_hat = basis[0, :]
            j_hat = basis[1, :]
            k_hat = basis[2, :]
            x, y, z = p_cs
            np.testing.assert_allclose(
                p_gcs, origin + x * i_hat + y * j_hat + z * k_hat
            )

    def test_points_in_rectbox(self):
        # 5 points, first col: x, second col: y, third col: z
        coords = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ]
        )
        points = g.Points(coords)

        np.testing.assert_equal(points.points_in_rectbox(), [True] * 5)
        np.testing.assert_equal(
            points.points_in_rectbox(xmin=3.0), [False, True, True, True, True]
        )
        np.testing.assert_equal(
            points.points_in_rectbox(xmin=3.0, xmax=11),
            [False, True, True, True, False],
        )
        np.testing.assert_equal(
            points.points_in_rectbox(xmin=3.0, zmax=8.0),
            [False, True, True, False, False],
        )

    def test_aspoints(self, points):
        assert g.aspoints(points) is points
        points2 = g.aspoints(points.coords)
        assert isinstance(points2, g.Points)
        assert np.allclose(points.coords, points2.coords)

    def test_to_1d_points(self, points):
        new_points = points.to_1d_points()
        assert new_points.ndim == 1


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
        i_hat = [np.cos(self.theta), np.sin(self.theta), 0.0]
        j_hat = [-np.sin(self.theta), np.cos(self.theta), 0.0]
        origin = [66.0, 77.0, 88.0]
        return g.CoordinateSystem(origin, i_hat, j_hat)

    def test_cs(self, cs):
        assert np.allclose(cs.k_hat, [0.0, 0.0, 1.0])

    def test_coordinate_system(self):
        i_hat = [1.0, 0.0, 0.0]
        j_hat = [0.0, 1.0, 0.0]
        origin = [66.0, 77.0, 88.0]
        cs = g.CoordinateSystem(origin, i_hat, j_hat)

        assert np.allclose(cs.k_hat, [0.0, 0.0, 1.0])

    def test_translate(self, cs):
        cs_expect = g.CoordinateSystem((0.0, 0.0, 0.0), cs.i_hat, cs.j_hat)
        cs_out = cs.translate(-cs.origin)
        assert cs_expect.isclose(cs_out)

    def test_rotate(self, cs):
        # reverse the rotation that gave the coordinate system:
        rot = g.rotation_matrix_z(-self.theta)
        cs_expect = g.CoordinateSystem(cs.origin, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        cs_out = cs.rotate(rot, centre=cs.origin)
        assert cs_expect.isclose(cs_out)

        # Second case, with centre O:
        rot = g.rotation_matrix_z(np.pi)
        cs_expect = g.CoordinateSystem(
            [-cs.origin[0], -cs.origin[1], +cs.origin[2]], -cs.i_hat, -cs.j_hat
        )
        cs_out = cs.rotate(rot, centre=None)
        assert cs_expect.isclose(cs_out)

    def test_convert_gcs(self):
        points = g.Points(np.arange(12, dtype=np.float64).reshape((4, 3)))

        gcs = g.GCS

        # conversion of GCS to GCS: must let the points invariant
        points_out = gcs.convert_from_gcs(points)
        assert g.are_points_close(points, points_out)

        points_out = gcs.convert_to_gcs(points)
        assert g.are_points_close(points, points_out)

    def test_convert_gcs2(self, cs):
        k1 = (0.5, 0.6, 0.7)
        k2 = (2.0, -0.6, 0.9)
        points_pcs = g.Points(np.array((k1, k2)))
        p1 = cs.origin + k1[0] * cs.i_hat + k1[1] * cs.j_hat + k1[2] * cs.k_hat
        p2 = cs.origin + k2[0] * cs.i_hat + k2[1] * cs.j_hat + k2[2] * cs.k_hat
        points_gcs = g.Points(np.array((p1, p2)))

        # Test GCS to PCS
        points_pcs_out = cs.convert_from_gcs(points_gcs)
        assert g.are_points_close(points_pcs_out, points_pcs)

        # Test PCS to GCS
        points_gcs_out = cs.convert_to_gcs(points_pcs)
        assert g.are_points_close(points_gcs_out, points_gcs)

    def test_from_gcs_pairwise(self, cs):
        assert isinstance(cs, g.CoordinateSystem)
        origins = g.Points.from_xyz(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 3.0, 4.0]),
        )
        rng = np.random.default_rng(seed=0)
        points_gcs = g.Points(rng.uniform(size=(10, 3)))

        # This is the tested function.
        x, y, z = cs.convert_from_gcs_pairwise(points_gcs, origins)
        assert x.shape == (len(points_gcs), len(origins))
        assert x.shape == y.shape == z.shape

        # Check the result are the same as if we convert the points "by hand".
        for j, origin in enumerate(origins):
            # Points in the j-th derived CS, computed by 'convert_from_gcs_pairwise':
            out_points = g.Points.from_xyz(
                x[..., j].copy(), y[..., j].copy(), z[..., j].copy()
            )

            # Points in the j-th derived CS, computed by the regular 'convert_from_gcs':
            origin_gcs = cs.convert_to_gcs(g.Points(origin))[()]
            cs_derived = g.CoordinateSystem(origin_gcs, cs.i_hat, cs.j_hat)
            expected_points = cs_derived.convert_from_gcs(points_gcs)

            assert g.are_points_close(out_points, expected_points)

    def test_isometry_preserves_distance(self, cs):
        """
        ultimate test
        """
        points = g.Points(np.arange(15, dtype=np.float64).reshape((5, 3)))
        points_cs_init = cs.convert_from_gcs(points)

        # define a nasty isometry:
        centre = np.array((1.1, 1.2, 1.3))
        rotation = g.rotation_matrix_ypr(0.5, -0.6, 0.7)
        translation = np.array((66.0, -77.0, 0.0))

        # move the coordinate system and the points:
        points_final = points.rotate(rotation, centre).translate(translation)
        cs_final = cs.rotate(rotation, centre).translate(translation)

        # check that the points are still the same
        points_cs_final = cs_final.convert_from_gcs(points_final)
        assert g.are_points_close(points_cs_init, points_cs_final)


EUCLIDEAN_DISTANCE_1 = None  # used for cache


@pytest.fixture(
    scope="module",
    params=[
        dict(block_size=None, numthreads=None),
        dict(block_size=1, numthreads=1),
        dict(block_size=1, numthreads=4),
        dict(block_size=10, numthreads=4),
    ],
)
def distance(request):
    """
    Compute the Euclidean distance using our package. Check that multithreading has no
    effect on the result.
    """
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
        distance = np.full((len(points1), len(points2)), 0, dtype=np.float64)
        for i in range(len(points1)):
            for j in range(len(points2)):
                distance[i, j] = math.sqrt(
                    (points1.x[i] - points2.x[j]) ** 2
                    + (points1.y[i] - points2.y[j]) ** 2
                    + (points1.z[i] - points2.z[j]) ** 2
                )
        EUCLIDEAN_DISTANCE_1 = distance
    return EUCLIDEAN_DISTANCE_1


def test_euclidean_distance(distance):
    mock_distance = mock_euclidean_distance(**DATASET_1)

    assert np.allclose(distance, mock_distance)


def test_euclidean_distance_advanced():
    mock_distance = mock_euclidean_distance(**DATASET_1)

    dtype = np.complex128  # weird though

    distance = g.distance_pairwise(**DATASET_1, dtype=dtype)
    assert distance.dtype == dtype
    assert np.allclose(np.real(distance), mock_distance)

    distance = np.full(
        (len(DATASET_1["points1"]), len(DATASET_1["points2"])), 0.0, dtype=np.float64
    )
    g.distance_pairwise(**DATASET_1, out=distance)  # write inplace
    assert np.allclose(distance, mock_distance)


def test_points_1d_wall_z():
    args = dict(xmin=10, xmax=20, numpoints=6, y=1, z=2, name="toto")
    points, orientations = g.points_1d_wall_z(**args)
    assert points.shape == (6,)
    assert orientations.shape == (6, 3)

    np.testing.assert_allclose(points.x, [10, 12, 14, 16, 18, 20])
    np.testing.assert_allclose(points.y, args["y"])
    np.testing.assert_allclose(points.z, args["z"])
    for i in range(6):
        np.testing.assert_allclose(orientations[i], np.eye(3))


def test_combine_walls():
    coords = np.asarray(
        [
            [-10.0, 0.0, 0.0],
            [-10.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 0.0, 0.0],
        ]
    )
    walls = g.make_contiguous_geometry(coords, numpoints=5)

    assert len(walls) == 3
    np.testing.assert_allclose(walls["wall_0"].points.x, -10)
    np.testing.assert_allclose(walls["wall_0"].points.y, 0)
    np.testing.assert_allclose(walls["wall_0"].points.z, [0.0, 2.5, 5.0, 7.5, 10.0])
    np.testing.assert_allclose(walls["wall_1"].points.x, [-10.0, -5.0, 0.0, 5.0, 10.0])
    np.testing.assert_allclose(walls["wall_1"].points.y, 0)
    np.testing.assert_allclose(walls["wall_1"].points.z, 10)
    np.testing.assert_allclose(walls["wall_2"].points.x, 10)
    np.testing.assert_allclose(walls["wall_2"].points.y, 0)
    np.testing.assert_allclose(walls["wall_2"].points.z, [10.0, 7.5, 5.0, 2.5, 0.0])

    single_contiguous_wall = g.combine_oriented_points(walls.values(), name="new_wall")

    assert single_contiguous_wall.points.shape == (13,)
    assert single_contiguous_wall.orientations.shape == (13, 3)

    np.testing.assert_allclose(
        single_contiguous_wall.points.x,
        [
            -10.0,
            -10.0,
            -10.0,
            -10.0,
            -10.0,
            -5.0,
            0.0,
            5.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ],
    )
    np.testing.assert_allclose(single_contiguous_wall.points.y, 0)
    np.testing.assert_allclose(
        single_contiguous_wall.points.z,
        [0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0, 10.0, 10.0, 7.5, 5.0, 2.5, 0.0],
    )

    left_and_right_wall = g.combine_oriented_points(
        [walls["wall_0"], walls["wall_2"]], name="opposite_wall"
    )

    assert left_and_right_wall.points.shape == (10,)
    assert left_and_right_wall.orientations.shape == (10, 3)

    np.testing.assert_allclose(
        left_and_right_wall.points.x,
        [-10.0, -10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    )
    np.testing.assert_allclose(left_and_right_wall.points.y, 0)
    np.testing.assert_allclose(
        left_and_right_wall.points.z,
        [0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 7.5, 5.0, 2.5, 0.0],
    )
