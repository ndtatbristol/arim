import numpy as np
import pytest

import arim.utils as u
from arim.exceptions import InvalidShape, InvalidDimension, NotAnArray
from arim.enums import CaptureMethod


def test_fmc():
    numelements = 3
    tx2 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]

    tx, rx = u.fmc(numelements)

    shape = (numelements * numelements,)

    assert tx.shape == shape
    assert rx.shape == shape
    assert np.all(tx == tx2)
    assert np.all(rx == rx2)


def test_hmc():
    numelements = 3
    tx2 = [0, 0, 0, 1, 1, 2]
    rx2 = [0, 1, 2, 1, 2, 2]

    tx, rx = u.hmc(numelements)

    shape = (numelements * (numelements + 1) / 2,)

    assert tx.shape == shape
    assert rx.shape == shape
    assert np.all(tx == tx2)
    assert np.all(rx == rx2)

def test_infer_capture_method():
    # Valid HMC
    tx = [0, 0, 0, 1, 1, 2]
    rx = [0, 1, 2, 1, 2, 2]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.hmc

    # HMC with duplicate signals
    tx = [0, 0, 0, 1, 1, 2, 1]
    rx = [0, 1, 2, 1, 2, 2, 1]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.unsupported

    # HMC with missing signals
    tx = [0, 0, 0, 2, 1]
    rx = [0, 1, 2, 2, 1]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.unsupported

    # Valid HMC
    tx = [0, 1, 2, 1, 2, 2]
    rx = [0, 0, 0, 1, 1, 2]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.hmc

    # Something weird
    tx = [0, 1, 2, 1, 2, 2]
    rx = [0, 0, 0, 1]
    with pytest.raises(Exception):
        u.infer_capture_method(tx, rx)

    # Valid FMC
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.fmc

    # FMC with duplicate signals
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.unsupported

    # FMC with missing signals
    tx = [0, 0, 0, 1, 1, 1, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.unsupported

    # Negative values
    tx = [0, -1]
    rx = [0, 1]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.unsupported

    # Weird
    tx = [0, 5]
    rx = [0, 1]
    assert u.infer_capture_method(tx, rx) == CaptureMethod.unsupported

def test_get_name():
    metadata = dict(long_name='Nicolas', short_name='Nick')
    assert u.get_name(metadata) == 'Nicolas'

    del metadata['long_name']
    assert u.get_name(metadata) == 'Nick'

    del metadata['short_name']
    assert isinstance(u.get_name(metadata), str)

