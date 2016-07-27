# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:11:11 2016

@author: nb14908
"""

import numpy as np
import pytest

from arim.geometry import Points, norm2
from arim.im import fermat_solver as t


def test_path():
    s1 = t.Path(('frontwall', 1.234, 'backwall'))
    s1_bis = t.Path(('frontwall', 1.234, 'backwall'))
    assert s1 == s1_bis # check hashability

    s2 = t.Path(('backwall', 2.0, 'points1'))

    s12 = t.Path(('frontwall', 1.234, 'backwall', 2.0, 'points1'))
    assert s12 == (s1 + s2)
    assert isinstance(s1 + s2, t.Path)

    s1_rev= t.Path(('backwall', 1.234, 'frontwall'))
    assert s1.reverse() == s1_rev
    with pytest.raises(ValueError):
        s2 + s1

    with pytest.raises(ValueError):
        t.Path((1, 2))

    with pytest.raises(ValueError):
        t.Path((1, ))

    s3 = t.Path(('A', 1.0, 'B', 2.0, 'C', 3.0, 'D'))
    head, tail = s3.split_head()
    assert head == t.Path(('A', 1.0, 'B'))
    assert tail == t.Path(('B', 2.0, 'C', 3.0, 'D'))
    
    head, tail = s3.split_queue()
    assert head == t.Path(('A', 1.0, 'B', 2.0, 'C'))
    assert tail == t.Path(('C', 3.0, 'D'))

    assert tuple(s1.points) == ('frontwall', 'backwall')

    assert s1.num_points_sets == 2
    assert s12.num_points_sets == 3

def test_make_empty_ray_indices():
    dtype = np.uint
    num1 = 5
    num2 = 3
    indices = t.make_empty_ray_indices(num1, num2, dtype)

    for (i, j) in zip(range(num1), range(num2)):
        assert indices[i, j, 0] == i
        assert indices[i, j, 1] == j

@pytest.fixture
def path():
    nprobe = 10
    nfrontwall = 11
    nbackwall = 12
    ngrid = 13
    probe = Points(np.random.rand(nprobe), np.random.rand(nprobe), np.random.rand(nprobe), 'Probe')
    frontwall = Points(np.random.rand(nfrontwall), np.random.rand(nfrontwall), np.random.rand(nfrontwall), 'Frontwall')
    backwall = Points(np.random.rand(nbackwall), np.random.rand(nbackwall), np.random.rand(nbackwall), 'Backwall')
    grid = Points(np.random.rand(ngrid), np.random.rand(ngrid), np.random.rand(ngrid), 'Grid')

    path = t.Path((probe, 1.0, frontwall, 2.0, backwall, 3.0, grid))
    return path

@pytest.fixture
def empty_rays(path):
    times = np.zeros((len(path[0]), len(path[-1])), dtype=np.float)
    indices = np.zeros((len(path[0]), len(path[-1]), path.num_points_sets), dtype=np.uint)

    rays = t.Rays(times, indices, path)
    return rays


def test_rays_gone_through_extreme_points(empty_rays):
    rays = empty_rays
    lenpoints = [len(x) for x in rays.path.points]

    # make all rays go through non extreme points:
    rays.indices[..., 1] = 5
    rays.indices[..., 2] = 5

    expected = np.full(rays.times.shape, False, dtype=np.bool)

    # points passing by first point
    rays.indices[2, 5, 1] = 0
    expected[2, 5] = True
    rays.indices[2, 6, 2] = 0
    expected[2, 6] = True

    # points passing by last point
    rays.indices[3, 5, 1] = lenpoints[1] - 1
    expected[3, 5] = True
    rays.indices[3, 7, 2] = lenpoints[2] - 1
    expected[3, 7] = True

    out = rays.gone_through_extreme_points()
    assert np.all(out == expected)

def test_assemble_rays():
    n = 20
    m = 30
    p = 40
    d1 = 3
    d2 = 4

    indices_head = np.arange(n*m*d1, dtype=np.uint).reshape((n, m, d1))
    indices_tail = np.arange(100, 100+p*m*d2, dtype=np.uint).reshape((m, p, d2))

    indices_at_interface = np.fromfunction(lambda i,j: (i+j)%m, (n, p), dtype=np.uint)

    # Function to test:
    indices_rays = t.assemble_rays(indices_head, indices_tail, indices_at_interface)
    print(indices_rays)

    #%%
    for i in range(n):
        for j in range(p):
            index_interface = indices_at_interface[i, j]

            # Populate out_min_indices:
            for d in range(d1 - 1):
                assert indices_rays[i, j, d] == indices_head[i, index_interface, d]
            assert indices_rays[i, j, d1 - 1] == index_interface
            for d in range(d2 - 1):
                assert indices_rays[i, j, d1 + d] == indices_tail[index_interface, j, d + 1]


def test_fermat_solver():
    """
    Test Fermat solver by comparing it against a naive implementation.

    Check three and four interfaces.
    """
    n = 5
    m = 12 # number of points of interfaces B and C

    v1 = 99.
    v2 = 130.
    v3 = 99.
    v4 = 50.

    x_n = np.arange(n, dtype=float)
    x_m = np.linspace(-n, 2*n, m)

    standoff = 11.1
    z = 66.6
    theta = np.deg2rad(30.)
    interface_a = Points(x_n, standoff + x_n*np.sin(theta), np.full(n, z), 'Interface A')
    interface_b = Points(x_m, np.zeros(m), np.full(m, z), 'Interface B')
    interface_c = Points(x_m, -(x_m-5)**2 - 10., np.full(m, z), 'Interface C')

    path_1 = t.Path((interface_a, v1, interface_b, v2, interface_c))
    path_2 = t.Path((interface_a, v1, interface_b, v3, interface_c, v4, interface_b))

    # The test function must return a dictionary of Rays:
    solver = t.FermatSolver([path_1, path_2])
    rays_dict = solver.solve()

    assert len(rays_dict) == 2
    for path in [path_1, path_2]:
        # Check Rays.path attribute:
        assert path in rays_dict
        assert rays_dict[path].path is path

        assert rays_dict[path].indices.shape == (n, m, path.num_points_sets)
        assert rays_dict[path].times.shape == (n, m)

        # Check the first and last points of the rays:
        indices = rays_dict[path].indices
        assert np.all(indices[..., 0] == np.fromfunction(lambda i,j: i, (n, m)))
        assert np.all(indices[..., -1] == np.fromfunction(lambda i,j: j, (n, m)))


    # Check rays for path_1:
    for i in range(n):
        for j in range(m):
            min_tof = np.inf
            best_index = 0

            for k in range(m):
                tof = norm2(interface_a.x[i] - interface_b.x[k],
                            interface_a.y[i] - interface_b.y[k],
                            interface_a.z[i] - interface_b.z[k]) / v1 +  \
                      norm2(interface_c.x[j] - interface_b.x[k],
                            interface_c.y[j] - interface_b.y[k],
                            interface_c.z[j] - interface_b.z[k]) / v2
                if tof < min_tof:
                    min_tof = tof
                    best_index = k
            assert np.isclose(min_tof, rays_dict[path_1].times[i, j]), \
                "Wrong time of flight for ray (start={}, end={}) in path 1 ".format(i,j)
            assert best_index == rays_dict[path_1].indices[i, j, 1], \
                "Wrong indices for ray (start={}, end={}) in path 1 ".format(i,j)

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
                                interface_a.z[i] - interface_b.z[k1]) / v1 +  \
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
                "Wrong time of flight for ray (start={}, end={}) in path 2 ".format(i,j)
            assert (best_index_1, best_index_2) == tuple(rays_dict[path_2].indices[i, j, 1:3]), \
                "Wrong indices for ray (start={}, end={}) in path 2 ".format(i,j)

