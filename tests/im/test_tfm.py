import numpy as np
import pytest
from unittest.mock import Mock, patch
import math
import copy

import arim
import arim.im as im
import arim.im.amplitudes
import arim.im.tfm, arim.ray
import arim.io
import arim.geometry as g

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

    def test_extrema_lookup_times_in_rectbox(self, grid, probe):
        frame = Mock(spec='tx rx metadata numscanlines probe')
        tx = [0, 0, 0, 1, 1, 1]
        rx = [0, 1, 2, 1, 1, 2]
        frame.tx = tx
        frame.rx = rx
        frame.numscanlines = len(tx)
        frame.probe = probe
        frame.is_complete_assuming_reciprocity = Mock(return_value=True)

        lookup_times_tx = np.zeros((grid.numpoints, len(tx)))
        lookup_times_rx = np.zeros((grid.numpoints, len(tx)))

        # scanline 5 (tx=1, rx=2) is the minimum time:
        grid_idx = 5
        lookup_times_tx[grid_idx, 5] = -1.5
        lookup_times_rx[grid_idx, 5] = -1.5
        # some noise:
        lookup_times_tx[grid_idx, 4] = -2.
        lookup_times_rx[grid_idx, 4] = -0.1

        # scanline 1 (tx=0, rx=1) is the maximum time:
        grid_idx = 3
        lookup_times_tx[grid_idx, 1] = 1.5
        lookup_times_rx[grid_idx, 1] = 1.5
        # some noise:
        lookup_times_tx[0, 0] = 2.
        lookup_times_rx[0, 0] = 0.1

        with patch.object(im.BaseTFM, 'get_lookup_times_tx',
                          return_value=lookup_times_tx):
            with patch.object(im.BaseTFM, 'get_lookup_times_rx',
                              return_value=lookup_times_rx):
                tfm = im.BaseTFM(frame, grid)
                out = tfm.extrema_lookup_times_in_rectbox()
        assert math.isclose(out.tmin, -3.)
        assert math.isclose(out.tmax, 3.)
        assert out.tx_elt_for_tmin == 1
        assert out.rx_elt_for_tmin == 2
        assert out.tx_elt_for_tmax == 0
        assert out.rx_elt_for_tmax == 1

    def test_simple_tfm(self, grid, frame):
        # Check that SimpleTFM and ContactTFM gives consistent values:
        frame_bak = copy.deepcopy(frame)
        grid_bak = copy.deepcopy(grid)

        speed = 6300
        tfm_contact = im.ContactTFM(speed, frame=frame, grid=grid)
        res_contact = tfm_contact.run()

        lookup_times_tx = tfm_contact.get_lookup_times_tx()
        lookup_times_rx = tfm_contact.get_lookup_times_rx()

        tfm = im.SimpleTFM(frame_bak, grid_bak, lookup_times_tx, lookup_times_rx)
        res = tfm.run()

        np.testing.assert_allclose(res, res_contact)


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


def test_multiview_tfm():
    # make probe
    probe = arim.Probe.make_matrix_probe(5, 0.5e-3, 1, np.nan, 1e6)
    probe.set_reference_element('first')
    probe.reset_position()
    probe.translate([0., 0., -1e-3])

    # make frame
    tx_arr, rx_arr = arim.ut.hmc(probe.numelements)
    time = arim.Time(.5e-6, 1 / 20e6, 100)
    np.random.seed(42)
    scanlines = np.random.rand(len(tx_arr), len(time))
    block = arim.Material(6300, 3100)
    frame = arim.Frame(scanlines, time, tx_arr, rx_arr, probe, arim.ExaminationObject(block))

    # prepare view LL-T in contact
    grid = arim.Grid(0., 0., 0., 0., 5e-3, 5e-3, np.nan)
    backwall = arim.geometry.points_1d_wall_z(-1e-3, 1e-3, 10e-3, 200)
    backwall_interface = arim.Interface(*backwall)
    probe_interface = arim.Interface(*probe.to_oriented_points())
    grid_interface = arim.Interface(*grid.to_oriented_points())

    path_LL = arim.Path([probe_interface, backwall_interface, grid_interface], [block, block], ['L', 'L'])
    path_T = arim.Path([probe_interface, grid_interface], [block], ['T'])
    view = arim.View(path_LL, path_T, 'LL-T')
    arim.ray.ray_tracing([view], convert_to_fortran_order=True)

    # check there is no time limit issues
    assert view.tx_path.rays.times.min() > time.start
    assert view.tx_path.rays.times.min() > time.start
    assert view.tx_path.rays.times.max() < time.end
    assert view.tx_path.rays.times.max() < time.end

    # make TFM
    tfm = im.SingleViewTFM(frame, grid, view)
    tfm.run()

    # Check this value is unchanged over time!
    expected = np.array([[[13.810041527100738]]])
    np.testing.assert_allclose(tfm.res, expected)
