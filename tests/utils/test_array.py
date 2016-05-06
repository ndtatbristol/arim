import numpy as np
import pytest

import arim.utils as u
from arim.exceptions import InvalidShape, InvalidDimension, NotAnArray

def test_linspace2():
    start = 10.
    num = 21
    step = 0.5
    end = start + (num - 1) * step

    # Standard case
    x = u.linspace2(start, step, num)
    assert len(x) == num
    assert x[0] == start
    assert x[-1] == end

    # Check dtype
    dtype = np.complex
    x = u.linspace2(start, step, num, dtype)
    assert len(x) == num
    assert x[0] == start
    assert x[-1] == end
    assert x.dtype == dtype

    # Roundoff errors?
    start = 10.
    num = 1300
    step = 1e-7
    end = start + (num - 1) * step

    x = u.linspace2(start, step, num)
    assert len(x) == num
    assert x[0] == start
    assert x[-1] == end


def test_get_shape_safely():
    shape = (3, 4, 5)
    x = np.arange(3 * 4 * 5).reshape(shape)

    assert u.get_shape_safely(x, 'x', shape) == shape
    assert u.get_shape_safely(x, 'x', (3, None, 5)) == shape
    assert u.get_shape_safely(x, 'x') == shape
    assert u.get_shape_safely(x, 'x', (None, None, None)) == shape

    with pytest.raises(InvalidShape):
        u.get_shape_safely(x, 'x', (3, 4, 666))

    with pytest.raises(InvalidDimension):
        u.get_shape_safely(x, 'x', (3, 4, 5, 6))

    with pytest.raises(NotAnArray):
        u.get_shape_safely(x.tolist(), 'x', (3, 4, 5))


def test_chunk_array():
    # 1D:
    x = np.arange(10)
    size = 3
    res = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    for (sel, w2) in zip(u.chunk_array(x.shape, size), res):
        w1 = x[sel]
        assert np.all(w1 == w2)

    # 1D:
    x = np.arange(9)
    size = 3
    res = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for (sel, w2) in zip(u.chunk_array(x.shape, size), res):
        w1 = x[sel]
        assert np.all(w1 == w2)

    # 2D dim 0:
    x = np.arange(20).reshape((10, 2))
    size = 3
    res = [
        x[0:3, :],
        x[3:6, :],
        x[6:9, :],
        x[9:, :]]
    for (sel, w2) in zip(u.chunk_array(x.shape, size), res):
        w1 = x[sel]
        assert np.all(w1 == w2)

    # 2D dim 1:
    x = np.arange(20).reshape((2, 10))
    size = 3
    res = [
        x[:, 0:3],
        x[:, 3:6],
        x[:, 6:9],
        x[:, 9:],
    ]
    for (sel, w2) in zip(u.chunk_array(x.shape, size, axis=1), res):
        w1 = x[sel]
        assert np.all(w1 == w2)

    # 3D dim 1:
    x = np.arange(5 * 10 * 3).reshape((5, 10, 3))
    size = 3
    res = [
        x[:, 0:3, :],
        x[:, 3:6, :],
        x[:, 6:9, :],
        x[:, 9:, :],
    ]
    for (sel, w2) in zip(u.chunk_array(x.shape, size, axis=1), res):
        w1 = x[sel]
        assert np.all(w1 == w2)


def test_smallest_uint_that_fits():
    assert u.smallest_uint_that_fits(2**8-1) is np.uint8
    assert u.smallest_uint_that_fits(2**8) is np.uint16
    assert u.smallest_uint_that_fits(2**64-1) is np.uint64
    

    