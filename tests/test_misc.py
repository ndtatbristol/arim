import arim
import os

def test_git_version():
    v = arim.get_git_version()
    assert isinstance(v, str)
    assert v != ''

    v_short = arim.get_git_version(short=True)
    assert v_short == v

    v_long = arim.get_git_version(short=False)
    assert isinstance(v_long, str)
    assert v_long != ''
    assert len(v_long) >= len(v_short)


