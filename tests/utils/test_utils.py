import enum
import logging

import pytest

import arim.utils as u


def test_get_name():
    metadata = dict(long_name='Nicolas', short_name='Nic')
    assert u.get_name(metadata) == 'Nicolas'

    del metadata['long_name']
    assert u.get_name(metadata) == 'Nic'

    del metadata['short_name']
    assert isinstance(u.get_name(metadata), str)


def test_parse_enum_constant():
    Foo = enum.Enum("Foo", "foo bar")

    assert u.parse_enum_constant("foo", Foo) is Foo.foo
    assert u.parse_enum_constant(Foo.foo, Foo) is Foo.foo
    assert u.parse_enum_constant("bar", Foo) is Foo.bar
    assert u.parse_enum_constant(Foo.bar, Foo) is Foo.bar

    with pytest.raises(ValueError):
        u.parse_enum_constant("baz", Foo)
    with pytest.raises(ValueError):
        u.parse_enum_constant(Foo, Foo)


def test_timeit(capsys):
    logger = logging.getLogger(__name__)
    with u.timeit(logger=logger):
        1 + 1

    out, err = capsys.readouterr()
    assert out == ''
    assert err == ''

    with u.timeit('Foobar'):
        1 + 1
    out, err = capsys.readouterr()
    assert out.startswith('Foobar')
    assert err == ''


def test_cache():
    cache = u.Cache()
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
    cache = u.NoCache()
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