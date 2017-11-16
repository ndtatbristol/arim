import math
import pytest
import numpy as np
from collections import OrderedDict

import arim.geometry as g
from arim import Probe, ExaminationObject, Material, Time, Frame
import arim.im.das as das


def make_delay_and_sum_case1():
    """
    2 elements (x=0, y=0, z=0) and (x=1, y=0, z=0)
    1 reflector at (x=0, y=0, z=2)
    Speed: 10


    Parameters
    ----------
    self

    Returns
    -------

    """
    # very basic points1:
    locations = g.Points(np.array([(0, 0, 0), (1., 0, 0)], dtype=np.float))
    frequency = 1e6
    probe = Probe(locations, frequency)

    # examination object:
    vel = 10.
    material = Material(vel)
    examination_object = ExaminationObject(material)

    # scanlines
    time = Time(start=0.35, step=0.001, num=100)

    numscanlines = 3
    tx = np.array([0, 0, 1], dtype=np.int)
    rx = np.array([0, 1, 1], dtype=np.int)

    # Model a reflector at distance 2 from the first element, and sqrt(5) from the second
    rt5 = math.sqrt(5)
    times_of_flights = np.array([4.0 / vel, (rt5 + 2) / vel, (2 * rt5) / vel])
    scanlines = np.zeros((numscanlines, len(time)), dtype=np.float)

    for (i, val) in enumerate(times_of_flights):
        closest = np.abs(time.samples - val).argmin()
        scanlines[i, closest] = 5

    lookup_times_tx = (times_of_flights[tx == rx] / 2).reshape((1, 2))
    lookup_times_rx = lookup_times_tx.copy()
    scanline_weights = np.array([1.0, 2.0, 1.0])  # HMC
    amplitudes_tx = np.array([[1.0, 1.0]])
    amplitudes_rx = np.array([[1.0, 1.0]])
    focal_law = FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes_tx, amplitudes_rx, scanline_weights)

    frame = Frame(scanlines, time, tx, rx, probe, examination_object)

    return frame, focal_law


def test_delay_and_sum_basic():
    frame, focal_law = make_delay_and_sum_case1()

    res = das.delay_and_sum(frame, focal_law)
    assert res == 20.0


def _random_uniform(dtype, low=0., high=1., size=None):
    z = np.zeros(size, dtype)
    if np.issubdtype(dtype, np.complex):
        z.real = np.random.uniform(low, high, size)
        z.imag = np.random.uniform(low, high, size)
    elif np.issubdtype(dtype, np.float):
        z[...] = np.random.uniform(low, high, size)
    else:
        raise NotImplementedError
    return z


def make_delay_and_sum_case_random(dtype_float, dtype_data):
    locations = g.Points(np.array([(0, 0, 0), (1., 0, 0), (2., 0., 0.)], dtype=np.float))
    numelements = len(locations)
    frequency = 1e6
    probe = Probe(locations, frequency)

    # examination object:
    vel = 10.
    material = Material(vel)
    examination_object = ExaminationObject(material)

    # scanlines
    # time = Time(start=0.35, step=0.001, num=100)
    time = Time(start=0., step=0.001, num=100)

    tx = np.array([0, 0, 1, 1, 2, 2], dtype=np.int)
    rx = np.array([0, 1, 0, 1, 0, 1], dtype=np.int)
    numscanlines = len(tx)

    numpoints = 10

    start_lookup = time.start / 2
    stop_lookup = (time.end - time.step) / 2

    np.random.seed(31031596)
    scanlines = _random_uniform(dtype_data, 100., 101., size=(numscanlines, len(time)))
    amplitudes_tx = _random_uniform(dtype_data, 1.0, 1.1, size=(numpoints, numelements))
    amplitudes_rx = _random_uniform(dtype_data, -1.0, -1.1, size=(numpoints, numelements))
    scanline_weights = _random_uniform(dtype_data, size=(numscanlines))
    lookup_times_tx = _random_uniform(dtype_float, start_lookup, stop_lookup, (numpoints, numelements))
    lookup_times_rx = _random_uniform(dtype_float, start_lookup, stop_lookup, (numpoints, numelements))

    # Mess a bit lookup times to get out of bounds values:
    #lookup_times_tx[0, 0] = time.start / 2.
    #lookup_times_rx[1, 1] = time.end * 2.

    focal_law = FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes_tx, amplitudes_rx, scanline_weights)

    frame = Frame(scanlines, time, tx, rx, probe, examination_object)
    # import pdb; pdb.set_trace()

    return frame, focal_law


DATATYPES = OrderedDict()
# dtype_float, dtype_data
DATATYPES['f'] = (np.float32, np.float32)
DATATYPES['c'] = (np.float32, np.complex64)
DATATYPES['d'] = (np.float64, np.float64)
DATATYPES['z'] = (np.float64, np.complex128)

# DATATYPES = [
#     dict(code='f', dtype_float=np.float32, dtype_data=np.float32),
#     dict(code='c', dtype_float=np.float32, dtype_data=np.complex64),
#     dict(code='d', dtype_float=np.float64, dtype_data=np.float32),
#     dict(code='z', dtype_float=np.float64, dtype_data=np.complex128),
# ]


@pytest.fixture(params=['naive', 'numba', 'cpu'])
def das_func(request):
    return getattr(das, 'delay_and_sum_' + request.param)


@pytest.fixture(params=tuple(DATATYPES.values()), ids=tuple(DATATYPES.keys()))
def datatypes(request):
    return request.param


@pytest.fixture(params=(0., np.nan), ids=('fillvalue_0', 'fillvalue_nan'))
def fillvalue(request):
    return request.param


def test_delay_and_sum_all(das_func, datatypes, fillvalue):
    dtype_float, dtype_data = datatypes
    frame, focal_law = make_delay_and_sum_case_random(dtype_float, dtype_data)

    kwargs = dict(frame=frame, focal_law=focal_law, fillvalue=fillvalue)

    result = das_func(**kwargs)
    assert result.dtype == dtype_data

    reference_result = das.delay_and_sum_naive(**kwargs)

    # np.testing.assert_almost_equal(result, reference_result)
    assert np.allclose(result, reference_result, equal_nan=True)

    assert np.sum(np.isfinite(result)) > 0, "all nan!"
    assert np.count_nonzero(result) > 0, "all zeros!"
