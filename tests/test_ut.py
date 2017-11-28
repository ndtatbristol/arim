import numpy as np
import pytest

import arim
from arim import ut


def test_decibel():
    db = ut.decibel(0.01, reference=1.)
    assert np.allclose(db, -40.)

    arr = np.array([0.01, 0.1, 1., 10.])
    db = ut.decibel(arr)
    assert np.allclose(db, [-60., -40., -20., 0.])

    arr = np.array([0.01, 0.1, 1., 10.])
    db = ut.decibel(arr, reference=1.)
    assert np.allclose(db, [-40., -20., 0., 20.])

    arr = np.array([0.01, 0.1, 1., 10.])
    db, ref = ut.decibel(arr, return_reference=True)
    assert np.allclose(db, [-60., -40., -20., 0.])
    assert np.isclose(ref, 10.)

    arr = np.array([0.01, 0.1, 1., 10., np.nan])
    with np.errstate(all='raise'):
        db = ut.decibel(arr)
    assert np.isnan(db[-1])
    assert np.allclose(db[:-1], [-60., -40., -20., 0.])

    # Check argument neginf_values:
    arr = np.array([0., 0.01, 0.1, 1., 10., np.nan])
    with np.errstate(all='raise'):
        db = ut.decibel(arr, neginf_value=-666.)
    assert np.isnan(db[-1])
    assert np.allclose(db[:-1], [-666., -60., -40., -20., 0.])

    arr = np.array([0., 0.01, 0.1, 1., 10., np.nan])
    with np.errstate(all='raise'):
        db = ut.decibel(arr, neginf_value=None)
    assert np.isnan(db[-1])
    assert np.isneginf(db[0])
    assert np.allclose(db[1:-1], [-60., -40., -20., 0.])


def test_fmc():
    numelements = 3
    tx2 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]

    tx, rx = ut.fmc(numelements)

    shape = (numelements * numelements,)

    assert tx.shape == shape
    assert rx.shape == shape
    assert np.all(tx == tx2)
    assert np.all(rx == rx2)


def test_hmc():
    numelements = 3
    tx2 = [0, 0, 0, 1, 1, 2]
    rx2 = [0, 1, 2, 1, 2, 2]

    tx, rx = ut.hmc(numelements)

    shape = (numelements * (numelements + 1) / 2,)

    assert tx.shape == shape
    assert rx.shape == shape
    assert np.all(tx == tx2)
    assert np.all(rx == rx2)


def test_infer_capture_method():
    # Valid HMC
    tx = [0, 0, 0, 1, 1, 2]
    rx = [0, 1, 2, 1, 2, 2]
    assert ut.infer_capture_method(tx, rx) == 'hmc'

    # HMC with duplicate signals
    tx = [0, 0, 0, 1, 1, 2, 1]
    rx = [0, 1, 2, 1, 2, 2, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # HMC with missing signals
    tx = [0, 0, 0, 2, 1]
    rx = [0, 1, 2, 2, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # Valid HMC
    tx = [0, 1, 2, 1, 2, 2]
    rx = [0, 0, 0, 1, 1, 2]
    assert ut.infer_capture_method(tx, rx) == 'hmc'

    # Something weird
    tx = [0, 1, 2, 1, 2, 2]
    rx = [0, 0, 0, 1]
    with pytest.raises(Exception):
        ut.infer_capture_method(tx, rx)

    # Valid FMC
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert ut.infer_capture_method(tx, rx) == 'fmc'

    # FMC with duplicate signals
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # FMC with missing signals
    tx = [0, 0, 0, 1, 1, 1, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # Negative values
    tx = [0, -1]
    rx = [0, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # Weird
    tx = [0, 5]
    rx = [0, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'


def test_default_scanline_weights():
    # FMC
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    expected = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    np.testing.assert_almost_equal(ut.default_scanline_weights(tx, rx), expected)

    # FMC with dead-element 1
    tx = [0, 0, 2, 2]
    rx = [0, 2, 0, 2]
    expected = [1., 1., 1., 1.]
    np.testing.assert_almost_equal(ut.default_scanline_weights(tx, rx), expected)

    # HMC
    tx = [0, 0, 0, 1, 1, 2]
    rx = [0, 1, 2, 1, 2, 2]
    expected = [1., 2., 2., 1., 2., 1.]
    np.testing.assert_almost_equal(ut.default_scanline_weights(tx, rx), expected)

    # HMC with dead-element 1
    tx = [0, 0, 2]
    rx = [0, 2, 2]
    expected = [1., 2., 1., ]
    np.testing.assert_almost_equal(ut.default_scanline_weights(tx, rx), expected)

    # HMC again
    tx, rx = ut.hmc(30)
    expected = np.ones(len(tx))
    expected[tx != rx] = 2.
    np.testing.assert_almost_equal(ut.default_scanline_weights(tx, rx), expected)


def test_instantaneous_phase_shift():
    t = np.arange(300)
    f0 = 20
    theta = np.pi / 3
    sig = 12. * np.exp(1j * (2. * np.pi * f0 * t + theta))

    theta_computed = ut.instantaneous_phase_shift(sig, t, f0)
    np.testing.assert_allclose(theta_computed, theta)

    with pytest.warns(ut.UtWarning):
        theta_computed = ut.instantaneous_phase_shift(sig.real, t, f0)


def test_wrap_phase():
    res_phases = [
        # unwrapped, wrapped
        (np.pi, -np.pi),
        (-np.pi, -np.pi),
        (4.5 * np.pi, 0.5 * np.pi),
        (3.5 * np.pi, -0.5 * np.pi),
        (-4.5 * np.pi, -0.5 * np.pi),
    ]
    unwrapped, wrapped = zip(*res_phases)
    np.testing.assert_allclose(ut.wrap_phase(unwrapped), wrapped)


def test_make_timevect():
    # loop over different values to check numerical robustness
    num_list = list(range(30, 40)) + list(range(2000, 2020))
    for num in num_list:
        start = 300e-6
        step = 50e-9
        end = start + (num - 1) * step

        # Standard case without start
        x = ut.make_timevect(num, step)
        assert len(x) == num
        np.testing.assert_allclose(x[1] - x[0], step)
        np.testing.assert_allclose(x[0], 0.)
        np.testing.assert_allclose(x[-1], (num - 1) * step)

        # Standard case
        x = ut.make_timevect(num, step, start)
        np.testing.assert_allclose(x[1] - x[0], step)
        np.testing.assert_allclose(x[0], start)
        np.testing.assert_allclose(x[-1], end)

        # Check dtype
        dtype = np.complex
        x = ut.make_timevect(num, step, start, dtype)
        np.testing.assert_allclose(x[1] - x[0], step)
        np.testing.assert_allclose(x[0], start)
        np.testing.assert_allclose(x[-1], end)
        assert x.dtype == dtype


def test_filter_unique_views():
    unique_views = arim.ut.filter_unique_views([('AB', 'CD'), ('DC', 'BA'), ('X', 'YZ'),
                                                ('ZY', 'X')])
    assert unique_views == [('AB', 'CD'), ('X', 'YZ')]


def test_make_viewnames():
    L = 'L'
    T = 'T'
    LL = 'LL'

    viewnames = arim.ut.make_viewnames(['L', 'T'], tfm_unique_only=False)
    assert viewnames == [(L, L), (L, T), (T, L), (T, T)]

    viewnames = arim.ut.make_viewnames(['L', 'T'], tfm_unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T)]

    viewnames = arim.ut.make_viewnames(['L', 'T'], tfm_unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T)]

    # legacy IMAGING_MODES
    legacy_imaging_views = ["L-L", "L-T", "T-T",
                            "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L",
                            "TT-T",
                            "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                            "LT-LT", "LT-TL", "LT-TT",
                            "TL-LT", "TL-TT",
                            "TT-TT"]
    legacy_imaging_views = [tuple(view.split('-')) for view in legacy_imaging_views]

    viewnames = arim.ut.make_viewnames(['L', 'T', 'LL', 'LT', 'TL', 'TT'],
                                       tfm_unique_only=True)
    assert viewnames == legacy_imaging_views

    viewnames = arim.ut.make_viewnames(arim.ut.DIRECT_PATHS + arim.ut.SKIP_PATHS,
                                       tfm_unique_only=True)
    assert viewnames == legacy_imaging_views

    viewnames = arim.ut.make_viewnames(
        arim.ut.DIRECT_PATHS + arim.ut.SKIP_PATHS + arim.ut.DOUBLE_SKIP_PATHS,
        tfm_unique_only=True)
    assert viewnames[:21] == legacy_imaging_views
    assert len(viewnames) == 105

    viewnames = arim.ut.make_viewnames(
        arim.ut.DIRECT_PATHS + arim.ut.SKIP_PATHS + arim.ut.DOUBLE_SKIP_PATHS,
        tfm_unique_only=False)
    assert len(viewnames) == 14 * 14

    viewnames = arim.ut.make_viewnames(['L', 'T', 'LL'], tfm_unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T), (LL, L), (LL, T), (LL, LL)]


def test_reciprocal_viewname():
    assert ut.reciprocal_viewname('L-LT') == 'TL-L'


def test_rayleigh_wave():
    # steel
    v_l = 5900.
    v_t = 3200.

    v_r = ut.rayleigh_vel(v_l, v_t)
    v_r_expected = 2959.250291  # the value was not checked against literature

    np.testing.assert_allclose(v_r, v_r_expected)
