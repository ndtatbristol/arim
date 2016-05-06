import pytest

import arim.core as c


def test_cache():
    cache = c.Cache()
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
    cache = c.NoCache()
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
