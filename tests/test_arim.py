import arim


def test_version():
    v = arim.__version__
    assert isinstance(v, str)
