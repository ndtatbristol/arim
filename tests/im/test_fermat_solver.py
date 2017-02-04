# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:11:11 2016

@author: nb14908
"""

import numpy as np
import pytest

from arim.geometry import Points, norm2
from arim.im import fermat_solver as t
from arim.im import Rays
from arim import Material

def test_fermat_path():
    s1 = t.FermatPath(('frontwall', 1.234, 'backwall'))
    s1_bis = t.FermatPath(('frontwall', 1.234, 'backwall'))
    assert s1 == s1_bis  # check hashability

    s2 = t.FermatPath(('backwall', 2.0, 'points1'))

    s12 = t.FermatPath(('frontwall', 1.234, 'backwall', 2.0, 'points1'))
    assert s12 == (s1 + s2)
    assert isinstance(s1 + s2, t.FermatPath)

    s1_rev = t.FermatPath(('backwall', 1.234, 'frontwall'))
    assert s1.reverse() == s1_rev
    with pytest.raises(ValueError):
        s2 + s1

    with pytest.raises(ValueError):
        t.FermatPath((1, 2))

    with pytest.raises(ValueError):
        t.FermatPath((1,))

    s3 = t.FermatPath(('A', 1.0, 'B', 2.0, 'C', 3.0, 'D'))
    head, tail = s3.split_head()
    assert head == t.FermatPath(('A', 1.0, 'B'))
    assert tail == t.FermatPath(('B', 2.0, 'C', 3.0, 'D'))

    head, tail = s3.split_queue()
    assert head == t.FermatPath(('A', 1.0, 'B', 2.0, 'C'))
    assert tail == t.FermatPath(('C', 3.0, 'D'))

    assert tuple(s1.points) == ('frontwall', 'backwall')

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
        interfaces = [Points.from_xyz(np.random.rand(n),
                                      np.random.rand(n),
                                      np.random.rand(n), 'A{}'.format(i)) for (i, n) in enumerate(self.numpoints)]

        path = t.FermatPath((interfaces[0], 1.0, interfaces[1],
                             2.0, interfaces[2], 3.0, interfaces[3]))
        return path

    @pytest.fixture
    def interior_indices(self, dtype_indices):
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

        times = np.full((n, q), np.nan, dtype=np.double)

        rays = Rays(times, interior_indices, path)
        assert np.all(interior_indices == rays.interior_indices)
        return rays

    def test_rays_indices(self, rays):
        dtype_indices = rays.indices.dtype
        indices = rays.indices
        interior_indices = rays.interior_indices

        n, m, p, q = self.numpoints

        assert indices.dtype == interior_indices.dtype
        assert indices.shape == (self.d, n, q)

        assert np.all(indices[0, ...] == np.fromfunction(
            lambda i, j: i, (n, q), dtype=dtype_indices))
        assert np.all(indices[-1, ...] == np.fromfunction(lambda i,
                                                                 j: j, (n, q), dtype=dtype_indices))

        for k in range(self.d - 2):
            np.testing.assert_allclose(interior_indices[k, ...], indices[k + 1, ...])

    def test_expand_rays(self, interior_indices):
        dtype_indices = interior_indices.dtype
        n, _, _, q = self.numpoints
        r = q + 1
        indices_new_interface = (np.arange(n * r, dtype=dtype_indices) % q)[::-1].reshape((n, r))
        indices_new_interface = np.ascontiguousarray(indices_new_interface)

        expanded_indices = Rays.expand_rays(interior_indices, indices_new_interface)
        assert expanded_indices.shape == (self.d - 1, n, r)

        for i in range(n):
            for j in range(r):
                # Index on the interface A(d-1):
                idx = indices_new_interface[i, j]
                for k in range(self.d - 2):
                    assert expanded_indices[k, i, j] == interior_indices[k, i, idx]
                assert expanded_indices[self.d - 2, i, j] == idx

    def test_rays_gone_through_extreme_points(self, rays):
        expected = np.full(rays.times.shape, False, dtype=np.bool)
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


class TestRays2:
    """
    Path of two interfaces. Use Rays' alternative constructor for this case.
    """
    d = 2
    numpoints = [4, 5]

    @pytest.fixture
    def path(self):
        interfaces = [Points.from_xyz(np.random.rand(n),
                                      np.random.rand(n),
                                      np.random.rand(n), 'A{}'.format(i)) for (i, n) in enumerate(self.numpoints)]

        path = t.FermatPath((interfaces[0], 2.0, interfaces[1]))
        return path

    @pytest.fixture
    def rays(self, path, dtype_indices):
        """Test alternative constructor of Rays"""
        n, m = self.numpoints
        times = np.full((n, m), np.nan, dtype=np.double)
        rays = Rays.make_rays_two_interfaces(times, path, dtype_indices)
        return rays

    def test_rays(self, rays):
        dtype_indices = rays.indices.dtype
        n, m = self.numpoints
        assert rays.indices.shape == (2, n, m)
        assert np.all(rays.indices[0, ...] == np.fromfunction(lambda i, j: i, (n, m), dtype=dtype_indices))
        assert np.all(rays.indices[1, ...] == np.fromfunction(lambda i, j: j, (n, m), dtype=dtype_indices))

    def test_expand_rays(self, rays):
        dtype_indices = rays.indices.dtype
        n, m = self.numpoints
        r = m + 1
        indices_new_interface = (np.arange(n * r, dtype=dtype_indices) % m)[::-1].reshape((n, r))
        indices_new_interface = np.ascontiguousarray(indices_new_interface)

        expanded_indices = Rays.expand_rays(rays.interior_indices, indices_new_interface)
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


@pytest.fixture(scope="module", params=[np.uint])
def dtype_indices(request):
    return request.param


def test_fermat_solver():
    """
    Test Fermat solver by comparing it against a naive implementation.

    Check three and four interfaces.
    """
    n = 5
    m = 12  # number of points of interfaces B and C

    v1 = 99.
    v2 = 130.
    v3 = 99.
    v4 = 50.

    x_n = np.arange(n, dtype=float)
    x_m = np.linspace(-n, 2 * n, m)

    standoff = 11.1
    z = 66.6
    theta = np.deg2rad(30.)
    interface_a = Points.from_xyz(x_n, standoff + x_n * np.sin(theta), np.full(n, z), 'Interface A')
    interface_b = Points.from_xyz(x_m, np.zeros(m), np.full(m, z), 'Interface B')
    interface_c = Points.from_xyz(x_m, -(x_m - 5) ** 2 - 10., np.full(m, z), 'Interface C')

    path_1 = t.FermatPath((interface_a, v1, interface_b, v2, interface_c))
    path_2 = t.FermatPath((interface_a, v1, interface_b, v3, interface_c, v4, interface_b))

    # The test function must return a dictionary of Rays:
    solver = t.FermatSolver([path_1, path_2])
    rays_dict = solver.solve()

    assert len(rays_dict) == 2
    for path in [path_1, path_2]:
        # Check Rays.path attribute:
        assert path in rays_dict
        assert rays_dict[path].path is path

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
                tof = norm2(interface_a.x[i] - interface_b.x[k],
                            interface_a.y[i] - interface_b.y[k],
                            interface_a.z[i] - interface_b.z[k]) / v1 + \
                      norm2(interface_c.x[j] - interface_b.x[k],
                            interface_c.y[j] - interface_b.y[k],
                            interface_c.z[j] - interface_b.z[k]) / v2
                if tof < min_tof:
                    min_tof = tof
                    best_index = k
            assert np.isclose(min_tof, rays_dict[path_1].times[i, j]), \
                "Wrong time of flight for ray (start={}, end={}) in path 1 ".format(i, j)
            assert best_index == rays_dict[path_1].indices[1, i, j], \
                "Wrong indices for ray (start={}, end={}) in path 1 ".format(i, j)

    # Check rays for path_2:
    for i in range(n):
        for j in range(m):
            min_tof = np.inf
            best_index_1 = 0
            best_index_2 = 0

            for k1 in range(m):
                for k2 in range(m):
                    tof = norm2(interface_a.x[i] - interface_b.x[k1],
                                interface_a.y[i] - interface_b.y[k1],
                                interface_a.z[i] - interface_b.z[k1]) / v1 + \
                          norm2(interface_c.x[k2] - interface_b.x[k1],
                                interface_c.y[k2] - interface_b.y[k1],
                                interface_c.z[k2] - interface_b.z[k1]) / v3 + \
                          norm2(interface_b.x[j] - interface_c.x[k2],
                                interface_b.y[j] - interface_c.y[k2],
                                interface_b.z[j] - interface_c.z[k2]) / v4

                    if tof < min_tof:
                        min_tof = tof
                        best_index_1 = k1
                        best_index_2 = k2

            assert np.isclose(min_tof, rays_dict[path_2].times[i, j]), \
                "Wrong time of flight for ray (start={}, end={}) in path 2 ".format(i, j)
            assert (best_index_1, best_index_2) == tuple(rays_dict[path_2].indices[1:3, i, j]), \
                "Wrong indices for ray (start={}, end={}) in path 2 ".format(i, j)
