import enum
import logging

import numpy as np
import pytest

import arim.helpers
from arim.exceptions import InvalidShape, InvalidDimension, NotAnArray


def test_get_name():
    metadata = dict(long_name='Nicolas', short_name='Nic')
    assert arim.helpers.get_name(metadata) == 'Nicolas'

    del metadata['long_name']
    assert arim.helpers.get_name(metadata) == 'Nic'

    del metadata['short_name']
    assert isinstance(arim.helpers.get_name(metadata), str)


def test_parse_enum_constant():
    Foo = enum.Enum("Foo", "foo bar")

    assert arim.helpers.parse_enum_constant("foo", Foo) is Foo.foo
    assert arim.helpers.parse_enum_constant(Foo.foo, Foo) is Foo.foo
    assert arim.helpers.parse_enum_constant("bar", Foo) is Foo.bar
    assert arim.helpers.parse_enum_constant(Foo.bar, Foo) is Foo.bar

    with pytest.raises(ValueError):
        arim.helpers.parse_enum_constant("baz", Foo)
    with pytest.raises(ValueError):
        arim.helpers.parse_enum_constant(Foo, Foo)


def test_timeit(capsys):
    logger = logging.getLogger(__name__)
    with arim.helpers.timeit(logger=logger):
        1 + 1

    out, err = capsys.readouterr()
    assert out == ''
    assert err == ''

    with arim.helpers.timeit('Foobar'):
        1 + 1
    out, err = capsys.readouterr()
    assert out.startswith('Foobar')
    assert err == ''


def test_cache():
    cache = arim.helpers.Cache()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0

    cache['toto'] = 'titi'
    assert len(cache) == 1
    assert cache.hits == 0
    assert cache.misses == 0

    a = cache['toto']
    assert a == 'titi'
    assert len(cache) == 1
    assert cache.hits == 1
    assert cache.misses == 0

    a = cache.get('toto')
    assert a == 'titi'
    assert len(cache) == 1
    assert cache.hits == 2
    assert cache.misses == 0

    b = cache.get('foo', None)
    assert len(cache) == 1
    assert cache.hits == 2
    assert cache.misses == 1

    with pytest.raises(KeyError):
        b = cache['another_miss']
    assert len(cache) == 1
    assert cache.hits == 2
    assert cache.misses == 2

    # 'in' statement do not change the hits/misses count:
    'toto' in cache
    'tata' in cache
    assert len(cache) == 1
    assert cache.hits == 2
    assert cache.misses == 2

    str(cache)
    cache.stat()
    cache.clear()


def test_nocache():
    cache = arim.helpers.NoCache()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0

    cache['toto'] = 'titi'  # this should do nothing
    assert 'toto' not in cache
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0

    with pytest.raises(KeyError):
        a = cache['toto']
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 1

    a = cache.get('toto')
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 2

    # 'in' statement do not change the hits/misses count:
    'toto' in cache
    'tata' in cache
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 2

    str(cache)
    cache.stat()
    cache.clear()


def test_git_version():
    v = arim.helpers.get_git_version()
    assert isinstance(v, str)
    assert v != ''

    v_short = arim.helpers.get_git_version(short=True)
    assert v_short == v

    v_long = arim.helpers.get_git_version(short=False)
    assert isinstance(v_long, str)
    assert v_long != ''
    assert len(v_long) >= len(v_short)


def test_get_shape_safely():
    shape = (3, 4, 5)
    x = np.arange(3 * 4 * 5).reshape(shape)

    assert arim.helpers.get_shape_safely(x, 'x', shape) == shape
    assert arim.helpers.get_shape_safely(x, 'x', (3, None, 5)) == shape
    assert arim.helpers.get_shape_safely(x, 'x') == shape
    assert arim.helpers.get_shape_safely(x, 'x', (None, None, None)) == shape

    with pytest.raises(InvalidShape):
        arim.helpers.get_shape_safely(x, 'x', (3, 4, 666))

    with pytest.raises(InvalidDimension):
        arim.helpers.get_shape_safely(x, 'x', (3, 4, 5, 6))

    with pytest.raises(NotAnArray):
        arim.helpers.get_shape_safely(x.tolist(), 'x', (3, 4, 5))


def test_chunk_array():
    # 1D:
    x = np.arange(10)
    size = 3
    res = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    for (sel, w2) in zip(arim.helpers.chunk_array(x.shape, size), res):
        w1 = x[sel]
        assert np.all(w1 == w2)

    # 1D:
    x = np.arange(9)
    size = 3
    res = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for (sel, w2) in zip(arim.helpers.chunk_array(x.shape, size), res):
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
    for (sel, w2) in zip(arim.helpers.chunk_array(x.shape, size), res):
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
    for (sel, w2) in zip(arim.helpers.chunk_array(x.shape, size, axis=1), res):
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
    for (sel, w2) in zip(arim.helpers.chunk_array(x.shape, size, axis=1), res):
        w1 = x[sel]
        assert np.all(w1 == w2)


def test_smallest_uint_that_fits():
    assert arim.helpers.smallest_uint_that_fits(2 ** 8 - 1) is np.uint8
    assert arim.helpers.smallest_uint_that_fits(2 ** 8) is np.uint16
    assert arim.helpers.smallest_uint_that_fits(2 ** 64 - 1) is np.uint64


def test_sizeof_fmt():
    assert arim.helpers.sizeof_fmt(1) == '1.0 B'
    assert arim.helpers.sizeof_fmt(1024) == '1.0 KiB'
    assert arim.helpers.sizeof_fmt(2 * 1024) == '2.0 KiB'
    assert arim.helpers.sizeof_fmt(5 * 1024**2) == '5.0 MiB'
