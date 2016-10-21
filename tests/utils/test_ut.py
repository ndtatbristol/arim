import numpy as np
import arim.utils.ut as ut


def test_decibel():
    arr = np.array([0.01, 0.1, 1.])
    db = ut.decibel(arr)
    assert np.allclose(db, [-40., -20, 0.])

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
