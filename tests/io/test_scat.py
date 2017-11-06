import numpy as np
import pytest

from arim import io, scat

import arim.io.brain as mat
from arim.exceptions import InvalidShape, InvalidDimension, NotAnArray
from arim.core import CaptureMethod

from tests import helpers


def test_load_scat_from_matlab():
    # cf tests/data/scat/make_scat_matlab.m script
    fname = helpers.get_data_filename("scat/scat_matlab.mat")

    scat_obj = io.scat.load_scat_from_matlab(fname)

    assert isinstance(scat_obj, scat.ScatFromData)
    assert scat_obj.numfreq == 2
    assert scat_obj.numangles == 11
    assert 'LL' in scat_obj.orig_matrices
    assert 'TL' in scat_obj.orig_matrices

    np.testing.assert_allclose(scat_obj.orig_matrices['LL'][0, 0], 10j)
    np.testing.assert_allclose(scat_obj.orig_matrices['TL'][1, 0], 20j)


def test_load_scat():
    # matlab
    fname = helpers.get_data_filename("scat/scat_matlab.mat")
    scat_obj = io.scat.load_scat(fname)
    assert scat_obj.numfreq == 2
    assert scat_obj.numangles == 11
    assert 'LL' in scat_obj.orig_matrices
    assert 'TL' in scat_obj.orig_matrices
