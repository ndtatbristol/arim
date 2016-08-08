import numpy as np
import pytest

import arim
import arim.im as im
import arim.im.amplitudes
import arim.im.tfm
import arim.io
import arim.geometry as g
from arim.im import fermat_solver as t

from tests.helpers import get_data_filename


@pytest.fixture()
def probe():
    return arim.probes['ima_50_MHz_64_1d']


@pytest.fixture()
def grid():
    return g.Grid(-10., 10., 0., 0., 0., 15., 1.)


@pytest.fixture()
def frame():
    expdata_filename = get_data_filename("brain/exp_data.mat7.mat")
    frame = arim.io.load_expdata(expdata_filename)
    frame.probe = probe()
    return frame


@pytest.fixture()
def tfm(frame, grid):
    speed = 1.0
    return im.ContactTFM(speed, frame=frame, grid=grid)


class TestTFM:
    def test_contact_tfm(self, grid, frame):
        speed = 6300
        tfm = im.ContactTFM(speed, frame=frame, grid=grid)
        res = tfm.run()


class TestAmplitude:
    def test_uniform(self, frame, grid):
        amplitudes = arim.im.amplitudes.UniformAmplitudes(frame, grid)
        res = amplitudes()
        assert np.allclose(res, 1.)

    def test_directivity_finite_width_2d(self, frame, grid):
        amplitudes = arim.im.amplitudes.DirectivityFiniteWidth2D(frame, grid, speed=1.0)
        res = amplitudes()

    def test_multi_amplitudes(self, frame, grid):
        amp1 = arim.im.amplitudes.UniformAmplitudes(frame, grid)
        amp2 = arim.im.amplitudes.UniformAmplitudes(frame, grid)
        multi_amp = arim.im.amplitudes.MultiAmplitudes([amp1, amp2])
        assert np.allclose(multi_amp(), 1.)


def test_make_views():
    probe = g.Points(np.random.uniform(size=(10, 3)), 'Probe')
    frontwall = g.Points(np.random.uniform(size=(10, 3)), 'Frontwall')
    backwall = g.Points(np.random.uniform(size=(10, 3)), 'Backwall')
    grid = g.Points(np.random.uniform(size=(10, 3)), 'Grid')

    v_couplant = 1.0
    v_longi = 2.0
    v_shear = 3.0

    views = arim.im.tfm.MultiviewTFM.make_views(probe, frontwall, backwall, grid, v_couplant, v_longi, v_shear)

    assert len(views) == 21
    assert len(set([v.name for v in views])) == 21

    view = [v for v in views if v.name == 'LT-TL'][0]
    assert view.tx_path == view.rx_path

    view = [v for v in views if v.name == 'LT-LT'][0]
    assert view.tx_path == (probe, v_couplant, frontwall, v_longi, backwall, v_shear, grid)
    assert view.rx_path == (probe, v_couplant, frontwall, v_shear, backwall, v_longi, grid)