# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:11:11 2016

@author: nb14908
"""

import numpy as np
import pytest

from arim.geometry import Points
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
