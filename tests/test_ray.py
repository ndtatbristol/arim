import math

import numpy as np
import pytest
from matplotlib import pyplot as plt

import arim
from arim import ray
import arim.geometry as g


def test_find_minimum_times():
    """

                A1      A2

        B1      B2        B3

                C1



    Returns
    -------

    """
    rt2 = math.sqrt(2.0)
    rt5 = math.sqrt(5.0)

    # Remark: B3 is a bit more on the right to have only one global minimum.

    # distance1[i, k] = distance between Ai and Bk
    distance1 = np.array([[rt2, 1.0, rt2 + 0.1], [rt5, rt2, 1.1]])

    # distance2[k, j] = distance between Ak and Cj
    distance2 = np.array([[rt2], [1.0], [rt2 + 0.1]])

    # Case 1
    # Best times: A1->B2->C1, A2->B2->C1.
    speed1 = 1.0
    speed2 = 1.0
    time1 = distance1 / speed1
    time2 = distance2 / speed2

    best_times, best_indices = ray.find_minimum_times(time1, time2)
    expected_indices = np.array([[1], [1]])

    assert np.allclose(best_times, np.array([[2.0], [1.0 + rt2]]))
    assert np.all(best_indices == expected_indices)

    # Case 2: medium 2 is very fast so spend the shortest possible distance in medium 1
    # Best times: A1->B2->C1, A2->B2->C1.
    speed1 = 1.0
    speed2 = 50.0
    time1 = distance1 / speed1
    time2 = distance2 / speed2
    best_times, best_indices = ray.find_minimum_times(time1, time2)
    expected_indices = np.array([[1], [2]])

    assert np.all(best_indices == expected_indices)


def test_find_minimum_times2():
    n = 300
    m = 301
    p = 302

    # The unique minimum of the i-th row of time_1 is on the i-th column and is 0.
    time_1 = np.fromfunction(lambda i, j: (j - i) % m, (n, m), dtype=float)

    # Each column of time_2 is constant
    time_2 = np.fromfunction(lambda i, j: j * m, (m, p), dtype=float)

    # Run the tested function:
    best_times, best_indices = ray.find_minimum_times(time_1, time_2)

    # Expected results:
    best_times_expected = np.fromfunction(lambda i, j: m * j, (n, p), dtype=float)
    best_indices_expected = np.fromfunction(lambda i, j: i, (n, p), dtype=int)

    assert np.allclose(best_times_expected, best_times)
    assert np.all(best_indices_expected == best_indices)


def test_fermat_path():
    s1 = ray.FermatPath(("frontwall", 1.234, "backwall"))
    s1_bis = ray.FermatPath(("frontwall", 1.234, "backwall"))
    assert s1 == s1_bis  # check hashability

    s2 = ray.FermatPath(("backwall", 2.0, "points1"))

    s12 = ray.FermatPath(("frontwall", 1.234, "backwall", 2.0, "points1"))
    assert s12 == (s1 + s2)
    assert isinstance(s1 + s2, ray.FermatPath)

    s1_rev = ray.FermatPath(("backwall", 1.234, "frontwall"))
    assert s1.reverse() == s1_rev
    with pytest.raises(ValueError):
        s2 + s1

    with pytest.raises(ValueError):
        ray.FermatPath((1, 2))

    with pytest.raises(ValueError):
        ray.FermatPath((1,))

    s3 = ray.FermatPath(("A", 1.0, "B", 2.0, "C", 3.0, "D"))
    head, tail = s3.split_head()
    assert head == ray.FermatPath(("A", 1.0, "B"))
    assert tail == ray.FermatPath(("B", 2.0, "C", 3.0, "D"))

    head, tail = s3.split_queue()
    assert head == ray.FermatPath(("A", 1.0, "B", 2.0, "C"))
    assert tail == ray.FermatPath(("C", 3.0, "D"))

    assert tuple(s1.points) == ("frontwall", "backwall")
    assert s1.velocities == (1.234,)
    assert s3.velocities == (1.0, 2.0, 3.0)

    assert s1.num_points_sets == 2
    assert s12.num_points_sets == 3


class TestRays4:
    """
    Test Rays for four interfaces.
    """

    # Number of points of interfaces A0, A1, A2
    d = 4
    numpoints = [4, 5, 6, 7]

    @pytest.fixture
    def path(self):
        interfaces = [
            g.Points.from_xyz(
                np.random.rand(n), np.random.rand(n), np.random.rand(n), "A{}".format(i)
            )
            for (i, n) in enumerate(self.numpoints)
        ]

        path = ray.FermatPath(
            (interfaces[0], 1.0, interfaces[1], 2.0, interfaces[2], 3.0, interfaces[3])
        )
        return path

    @pytest.fixture
    def interior_indices(self):
        dtype_indices = arim.settings.INT
        n, m, p, q = self.numpoints
        interior_indices_1 = (np.arange(n * q, dtype=dtype_indices) % m).reshape(n, q)
        interior_indices_2 = (np.arange(n * q, dtype=dtype_indices) % p).reshape(n, q)
        interior_indices = np.zeros((self.d - 2, n, q), dtype=dtype_indices)
        interior_indices[0, ...] = interior_indices_1
        interior_indices[1, ...] = interior_indices_2
        return interior_indices

    @pytest.fixture
    def rays(self, path, interior_indices):
        n, m, p, q = self.numpoints

        times = np.random.uniform(10.0, 20.0, size=(n, q))
        rays = ray.Rays(times, interior_indices, path)
        assert np.all(interior_indices == rays.interior_indices)
        return rays

    def test_rays_indices(self, rays):
        dtype_indices = rays.indices.dtype
        indices = rays.indices
        interior_indices = rays.interior_indices

        n, m, p, q = self.numpoints

        assert indices.dtype == interior_indices.dtype
        assert indices.shape == (self.d, n, q)

        assert np.all(
            indices[0, ...]
            == np.fromfunction(lambda i, j: i, (n, q), dtype=dtype_indices)
        )
        assert np.all(
            indices[-1, ...]
            == np.fromfunction(lambda i, j: j, (n, q), dtype=dtype_indices)
        )

        for k in range(self.d - 2):
            np.testing.assert_allclose(interior_indices[k, ...], indices[k + 1, ...])

    def test_expand_rays(self, interior_indices):
        dtype_indices = interior_indices.dtype
        n, _, _, q = self.numpoints
        r = q + 1
        indices_new_interface = (np.arange(n * r, dtype=dtype_indices) % q)[
            ::-1
        ].reshape((n, r))
        indices_new_interface = np.ascontiguousarray(indices_new_interface)

        expanded_indices = ray.Rays.expand_rays(interior_indices, indices_new_interface)
        assert expanded_indices.shape == (self.d - 1, n, r)

        for i in range(n):
            for j in range(r):
                # Index on the interface A(d-1):
                idx = indices_new_interface[i, j]
                for k in range(self.d - 2):
                    assert expanded_indices[k, i, j] == interior_indices[k, i, idx]
                assert expanded_indices[self.d - 2, i, j] == idx

    def test_rays_gone_through_extreme_points(self, rays):
        expected = np.full(rays.times.shape, False, dtype=bool)
        n, m, p, q = self.numpoints

        interior_indices = rays.interior_indices
        np.logical_or(interior_indices[0, ...] == 0, expected, out=expected)
        np.logical_or(interior_indices[0, ...] == (m - 1), expected, out=expected)
        np.logical_or(interior_indices[1, ...] == 0, expected, out=expected)
        np.logical_or(interior_indices[1, ...] == (p - 1), expected, out=expected)

        out = rays.gone_through_extreme_points()
        np.testing.assert_equal(out, expected)

    def test_fortran_rays(self, rays):
        rays_f = rays.to_fortran_order()
        assert rays_f.times.flags.fortran
        assert rays_f.indices.flags.fortran
        np.testing.assert_equal(rays_f.indices, rays.indices)
        np.testing.assert_almost_equal(rays_f.times, rays.times)

    def test_reverse_rays(self, rays):
        rev_rays = rays.reverse()

        n, m, p, q = self.numpoints

        assert rev_rays.times.shape == (q, n)
        assert rev_rays.indices.shape == (self.d, q, n)

        for i in range(n):
            for j in range(m):
                assert rays.times[i, j] == rev_rays.times[j, i]
                for k in range(self.d):
                    idx = rev_rays.indices[-k - 1, j, i]
                    expected_idx = rays.indices[k, i, j]
                    assert idx == expected_idx, "err for i={}, j={}, k={}".format(
                        i, j, k
                    )


class TestRays2:
    """
    Path of two interfaces. Use Rays' alternative constructor for this case.
    """

    d = 2
    numpoints = [4, 5]

    @pytest.fixture
    def path(self):
        interfaces = [
            g.Points.from_xyz(
                np.random.rand(n), np.random.rand(n), np.random.rand(n), "A{}".format(i)
            )
            for (i, n) in enumerate(self.numpoints)
        ]

        path = ray.FermatPath((interfaces[0], 2.0, interfaces[1]))
        return path

    @pytest.fixture
    def rays(self, path):
        """Test alternative constructor of Rays"""
        dtype_indices = arim.settings.INT
        n, m = self.numpoints
        times = np.random.uniform(10.0, 20.0, size=(n, m))
        rays = ray.Rays.make_rays_two_interfaces(times, path, dtype_indices)
        return rays

    def test_rays(self, rays):
        dtype_indices = rays.indices.dtype
        n, m = self.numpoints
        assert rays.indices.shape == (2, n, m)
        assert np.all(
            rays.indices[0, ...]
            == np.fromfunction(lambda i, j: i, (n, m), dtype=dtype_indices)
        )
        assert np.all(
            rays.indices[1, ...]
            == np.fromfunction(lambda i, j: j, (n, m), dtype=dtype_indices)
        )

    def test_expand_rays(self, rays):
        dtype_indices = rays.indices.dtype
        n, m = self.numpoints
        r = m + 1
        indices_new_interface = (np.arange(n * r, dtype=dtype_indices) % m)[
            ::-1
        ].reshape((n, r))
        indices_new_interface = np.ascontiguousarray(indices_new_interface)

        expanded_indices = ray.Rays.expand_rays(
            rays.interior_indices, indices_new_interface
        )
        assert expanded_indices.shape == (self.d - 1, n, r)

        for i in range(n):
            for j in range(r):
                # Index on the interface A(d-1):
                idx = indices_new_interface[i, j]
                for k in range(self.d - 2):
                    assert expanded_indices[k, i, j] == rays.interior_indices[k, i, idx]
                assert expanded_indices[self.d - 2, i, j] == idx

    def test_rays_gone_through_extreme_points(self, rays):
        n, m = self.numpoints
        out = rays.gone_through_extreme_points()
        assert out.shape == (n, m)
        assert np.any(np.logical_not(out))

    def test_fortran_rays(self, rays):
        rays_f = rays.to_fortran_order()
        assert rays_f.times.flags.fortran
        assert rays_f.indices.flags.fortran
        np.testing.assert_equal(rays_f.indices, rays.indices)
        np.testing.assert_almost_equal(rays_f.times, rays.times)


def test_fermat_solver():
    """
    Test Fermat solver by comparing it against a naive implementation.

    Check three and four interfaces.
    """
    n = 5
    m = 12  # number of points of interfaces B and C

    v1 = 99.0
    v2 = 130.0
    v3 = 99.0
    v4 = 50.0

    x_n = np.arange(n, dtype=float)
    x_m = np.linspace(-n, 2 * n, m)

    standoff = 11.1
    z = 66.6
    theta = np.deg2rad(30.0)
    interface_a = g.Points.from_xyz(
        x_n, standoff + x_n * np.sin(theta), np.full(n, z), "Interface A"
    )
    interface_b = g.Points.from_xyz(x_m, np.zeros(m), np.full(m, z), "Interface B")
    interface_c = g.Points.from_xyz(
        x_m, -((x_m - 5) ** 2) - 10.0, np.full(m, z), "Interface C"
    )

    path_1 = ray.FermatPath((interface_a, v1, interface_b, v2, interface_c))
    path_2 = ray.FermatPath(
        (interface_a, v1, interface_b, v3, interface_c, v4, interface_b)
    )

    # The test function must return a dictionary of Rays:
    solver = ray.FermatSolver([path_1, path_2])
    rays_dict = solver.solve()

    assert len(rays_dict) == 2
    for path in [path_1, path_2]:
        # Check Rays.path attribute:
        assert path in rays_dict
        assert rays_dict[path].fermat_path is path

        assert rays_dict[path].indices.shape == (path.num_points_sets, n, m)
        assert rays_dict[path].times.shape == (n, m)

        # Check the first and last points of the rays:
        indices = rays_dict[path].indices
        assert np.all(indices[0, ...] == np.fromfunction(lambda i, j: i, (n, m)))
        assert np.all(indices[-1, ...] == np.fromfunction(lambda i, j: j, (n, m)))

    # Check rays for path_1:
    for i in range(n):
        for j in range(m):
            min_tof = np.inf
            best_index = 0

            for k in range(m):
                tof = (
                    g.norm2(
                        interface_a.x[i] - interface_b.x[k],
                        interface_a.y[i] - interface_b.y[k],
                        interface_a.z[i] - interface_b.z[k],
                    )
                    / v1
                    + g.norm2(
                        interface_c.x[j] - interface_b.x[k],
                        interface_c.y[j] - interface_b.y[k],
                        interface_c.z[j] - interface_b.z[k],
                    )
                    / v2
                )
                if tof < min_tof:
                    min_tof = tof
                    best_index = k
            assert np.isclose(
                min_tof, rays_dict[path_1].times[i, j]
            ), "Wrong time of flight for ray (start={}, end={}) in path 1 ".format(i, j)
            assert (
                best_index == rays_dict[path_1].indices[1, i, j]
            ), "Wrong indices for ray (start={}, end={}) in path 1 ".format(i, j)

    # Check rays for path_2:
    for i in range(n):
        for j in range(m):
            min_tof = np.inf
            best_index_1 = 0
            best_index_2 = 0

            for k1 in range(m):
                for k2 in range(m):
                    tof = (
                        g.norm2(
                            interface_a.x[i] - interface_b.x[k1],
                            interface_a.y[i] - interface_b.y[k1],
                            interface_a.z[i] - interface_b.z[k1],
                        )
                        / v1
                        + g.norm2(
                            interface_c.x[k2] - interface_b.x[k1],
                            interface_c.y[k2] - interface_b.y[k1],
                            interface_c.z[k2] - interface_b.z[k1],
                        )
                        / v3
                        + g.norm2(
                            interface_b.x[j] - interface_c.x[k2],
                            interface_b.y[j] - interface_c.y[k2],
                            interface_b.z[j] - interface_c.z[k2],
                        )
                        / v4
                    )

                    if tof < min_tof:
                        min_tof = tof
                        best_index_1 = k1
                        best_index_2 = k2

            assert np.isclose(
                min_tof, rays_dict[path_2].times[i, j]
            ), "Wrong time of flight for ray (start={}, end={}) in path 2 ".format(i, j)
            assert (best_index_1, best_index_2) == tuple(
                rays_dict[path_2].indices[1:3, i, j]
            ), "Wrong indices for ray (start={}, end={}) in path 2 ".format(i, j)


def make_incoming_angles_results():
    no_tilt_ref_angles = np.array(
        [
            78.69006752597979,
            76.86597769360367,
            74.35775354279127,
            70.70995378081126,
            64.98310652189996,
            55.00797980144132,
            35.537677791974374,
            0.0,
            35.537677791974374,
            55.007979801441344,
            64.98310652189998,
            70.70995378081126,
            74.35775354279127,
            76.86597769360367,
            78.69006752597979,
        ]
    )
    no_tilt_ref_angles_flipped = 180 - no_tilt_ref_angles
    angles_dict = {}
    # Key order: dest_points_are_above, are_normals_zplus, use_tilt
    angles_dict[True, False, False] = no_tilt_ref_angles
    angles_dict[False, False, False] = no_tilt_ref_angles_flipped
    angles_dict[True, True, False] = no_tilt_ref_angles_flipped
    angles_dict[False, True, False] = no_tilt_ref_angles

    return angles_dict


RAY_GEOMETRY_CASES = ("are_normals_zplus", [True, False])


class TestRayGeometry:
    """
    There are two interfaces: the first one (source) has a unique single point in Omega.
    The second one has 12 points in a circle of centre Omega and radius 5 mm. These points
    are spaced by 30°.

    The coordinate system of the source point is obtained by rotation of 30° around GCS-y
    axis. Therefore its normal points towards the first destination points.

    Use --show-plots in CLI to visualise the set-up. The arrows are the normals Oz+.


    """

    circle_num = 12
    circle_theta = arim.ut.wrap_phase(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    circle_radius = 5e-2

    def make_case(self, are_normals_zplus):
        """
        Source point: omega
        Dest points: 12 points along a circle of centre omega and radius 5 (points are
        spaced by  30°)

        Parameters
        ----------
        are_normals_zplus

        Returns
        -------

        """
        omega = np.array((3.0, 0.0, 5.0)) * 1e-2
        src_points = g.Points(omega.reshape([1, 3]), name="source")
        src_basis = g.default_orientations(src_points)
        src_basis = src_basis.rotate(g.rotation_matrix_y(np.pi / 6))
        source_interface = arim.Interface(
            src_points, src_basis, are_normals_on_out_rays_side=are_normals_zplus
        )

        # circle:
        dst_points = g.Points(np.zeros([len(self.circle_theta), 3]), name="dest")
        dst_points.x[...] = omega[0]
        dst_points.y[...] = omega[1]
        dst_points.z[...] = omega[2]
        dst_points.x[...] += self.circle_radius * np.sin(self.circle_theta)
        dst_points.z[...] += self.circle_radius * np.cos(self.circle_theta)

        dst_basis = g.default_orientations(dst_points)

        dest_interface = arim.Interface(
            dst_points, dst_basis, are_normals_on_inc_rays_side=are_normals_zplus
        )

        material = arim.Material(1.0, metadata=dict(long_name="Dummy"))

        interfaces = [source_interface, dest_interface]

        # The i-th ray starts from the source and ends at the i-th destination point.
        shape = [len(source_interface.points), len(dest_interface.points)]
        ray_indices = np.zeros((0, *shape), arim.settings.INT)
        times = np.empty(shape, float)
        times.fill(np.nan)

        path = arim.Path(interfaces, [material], ["L"])
        ray = arim.ray.Rays(times, ray_indices, path.to_fermat_path())
        path.rays = ray
        ray_geometry = arim.ray.RayGeometry.from_path(path)
        return path, ray_geometry

    def test_ray_geometry_cache(self):
        path, ray_geometry = self.make_case(are_normals_zplus=False)

        assert len(ray_geometry._cache) == 0

        # Case: final result
        leg_points = ray_geometry.leg_points(0, is_final=True)
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_intermediate_results()
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_all_results()
        assert len(ray_geometry._cache) == 0

        # Case: intermediate result
        leg_points = ray_geometry.leg_points(0, is_final=False)
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_intermediate_results()
        assert len(ray_geometry._cache) == 0
        ray_geometry.clear_all_results()
        assert len(ray_geometry._cache) == 0

        # Case: intermediate result promoted to final result
        leg_points = ray_geometry.leg_points(0, is_final=False)
        assert len(ray_geometry._cache) == 1
        # Promotion to final result:
        leg_points = ray_geometry.leg_points(0, is_final=True)
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_intermediate_results()
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_all_results()
        assert len(ray_geometry._cache) == 0

    def test_plot_cases(self, show_plots):
        if show_plots:
            kwargs = dict(are_normals_zplus=False)
            path, ray_geometry = self.make_case(**kwargs)
            ax = aplt.plot_interfaces(
                path.interfaces,
                show_grid=True,
                show_orientations=True,
                markers=[".", "."],
            )
            for idx, (x, y, z) in enumerate(path.interfaces[1].points):
                ax.text(x, z, str(idx))
            plt.title("TestRayGeometry.test_plot_cases\nThe arrows are the (Oz+) axes.")
            plt.show()

    def test_precompute(self):
        path, ray_geometry = self.make_case(True)
        with ray_geometry.precompute():
            r1 = ray_geometry.inc_angle(1)
            r2 = ray_geometry.signed_inc_angle(1)
            assert len(ray_geometry._cache) > 2, "no intermediate results is stored"

        assert len(ray_geometry._cache) == 2, "intermediate results aren'ray flushed"

        # Check that caching is effective:
        assert ray_geometry.inc_angle(1) is r1, "caching is not effective"
        assert ray_geometry.signed_inc_angle(1) is r2, "caching is not effective"

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_cache(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        r = ray_geometry.leg_points(1)
        r2 = ray_geometry.leg_points(1)
        assert r is r2

        # there are two interfaces so interfaces[-1] is interfaces[1]
        r3 = ray_geometry.leg_points(-1)
        assert r is r3

        # RayGeometry without cache:
        ray_geometry = arim.ray.RayGeometry.from_path(path, use_cache=False)
        r = ray_geometry.leg_points(1)
        r2 = ray_geometry.leg_points(1)
        assert r is not r2

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_leg_points(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        shape = len(path.interfaces[0].points), len(path.interfaces[1].points)
        ray_geometry = arim.ray.RayGeometry.from_path(path)

        leg_points = ray_geometry.leg_points(0)
        for point in leg_points:
            np.testing.assert_allclose(point, path.interfaces[0].points[0])

        leg_points = ray_geometry.leg_points(1)
        for i, point in enumerate(leg_points):
            np.testing.assert_allclose(point, path.interfaces[1].points[i])

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_radius(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_radius(0) is None
        radius = ray_geometry.inc_leg_radius(1)
        assert radius.shape == (1, self.circle_num)
        np.testing.assert_allclose(radius, self.circle_radius)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_size(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_size(0) is None
        radius = ray_geometry.inc_leg_radius(1)
        size = ray_geometry.inc_leg_size(1)
        np.testing.assert_allclose(size, radius)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_azimuth(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_azimuth(0) is None
        azimuth = ray_geometry.inc_leg_azimuth(1)
        assert azimuth.shape == (1, self.circle_num)

        # Azimuth is 0 if the source point is in x > 0 (from the dest point),
        # Pi if the source point is in x < 0.
        azimuth = np.squeeze(azimuth)
        for i, theta in enumerate(np.unwrap(self.circle_theta)):
            if np.isclose(theta, 0.0) or np.isclose(theta, np.pi):
                # Depends on x=+eps or x=-eps
                assert np.isclose(azimuth[i], 0.0) or np.isclose(azimuth[i], np.pi)
            elif 0 < theta < np.pi:
                assert np.isclose(azimuth[i], np.pi)
            else:
                assert np.isclose(azimuth[i], 0.0)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_polar(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_azimuth(0) is None
        polar = ray_geometry.inc_leg_polar(1)
        assert polar.shape == (1, self.circle_num)
        expected_polar = np.abs(arim.ut.wrap_phase(self.circle_theta + np.pi))

        np.testing.assert_allclose(
            np.rad2deg(np.squeeze(polar)), np.rad2deg(expected_polar)
        )

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_angle(0) is None

        polar = ray_geometry.inc_leg_polar(1)
        angles = ray_geometry.inc_angle(1)
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(polar))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_signed_inc_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.signed_inc_angle(0) is None
        angles = ray_geometry.signed_inc_angle(1)
        assert angles.shape == (1, self.circle_num)

        expected_angles = arim.ut.wrap_phase(self.circle_theta + np.pi)
        angles = arim.ut.wrap_phase(np.squeeze(angles))
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_conventional_inc_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.conventional_inc_angle(0) is None
        angles = ray_geometry.conventional_inc_angle(1)
        assert angles.shape == (1, self.circle_num)

        # wrap in [-pi, pi] then takes the abs value
        expected_angles = np.abs(arim.ut.wrap_phase(self.circle_theta + np.pi))
        if not are_normals_zplus:
            expected_angles = np.pi - expected_angles

        angles = np.squeeze(angles)
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_leg_radius(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_leg_radius(1) is None
        radius = ray_geometry.out_leg_radius(0)
        assert radius.shape == (1, self.circle_num)
        np.testing.assert_allclose(radius, self.circle_radius)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_leg_azimuth(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_leg_azimuth(1) is None
        azimuth = ray_geometry.out_leg_azimuth(0)
        assert azimuth.shape == (1, self.circle_num)

        # Azimuth is 0 if the source point is in x > 0 (from the dest point),
        # Pi if the source point is in x < 0.
        azimuth = np.squeeze(azimuth)
        for i, theta in enumerate(np.unwrap(self.circle_theta)):
            if np.isclose(theta, np.pi / 6) or np.isclose(theta, 7 * np.pi / 6):
                # Depends on x=+eps or x=-eps
                assert np.isclose(azimuth[i], 0.0) or np.isclose(azimuth[i], np.pi)
            elif np.pi / 6 < theta < 7 * np.pi / 6:
                assert np.isclose(azimuth[i], 0.0)
            else:
                assert np.isclose(azimuth[i], np.pi)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_leg_polar(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_leg_azimuth(1) is None
        polar = ray_geometry.out_leg_polar(0)
        assert polar.shape == (1, self.circle_num)
        expected_polar = np.abs(arim.ut.wrap_phase(self.circle_theta - np.pi / 6))

        np.testing.assert_allclose(
            np.rad2deg(np.squeeze(polar)), np.rad2deg(expected_polar)
        )

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_angle(1) is None

        polar = ray_geometry.out_leg_polar(0)
        angles = ray_geometry.out_angle(0)
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(polar))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_signed_out_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.signed_out_angle(1) is None
        angles = ray_geometry.signed_out_angle(0)
        assert angles.shape == (1, self.circle_num)

        expected_angles = np.unwrap(arim.ut.wrap_phase(self.circle_theta - np.pi / 6))
        angles = np.unwrap(arim.ut.wrap_phase(np.squeeze(angles)))

        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_conventional_out_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.conventional_out_angle(1) is None
        angles = ray_geometry.conventional_out_angle(0)
        assert angles.shape == (1, self.circle_num)

        # wrap in [-pi, pi] then takes the abs value
        expected_angles = np.abs(arim.ut.wrap_phase(self.circle_theta - np.pi / 6))
        if not are_normals_zplus:
            expected_angles = np.pi - expected_angles

        # angles = np.unwrap(arim.ut.wrap_phase(np.squeeze(angles)))
        angles = np.squeeze(angles)
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))
