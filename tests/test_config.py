import arim

def test_config():
    conf = arim.config.Config()
    conf['foo'] = 1
    conf['bar'] = 2
    conf['bar.baz'] = 3

    str(conf)
    repr(conf)

    assert conf['foo'] == 1
    assert conf['bar'] == 2

    assert conf.find_all('bar') == arim.config.Config([('bar', 2), ('bar.baz', 3)])
    assert conf.keys() == ['bar', 'bar.baz', 'foo']

