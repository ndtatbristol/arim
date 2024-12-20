import contextlib
from collections import OrderedDict

import numpy as np
import pytest

import arim.geometry as g
import arim.im.das as das
import arim.im.tfm
from arim import ExaminationObject, Frame, Material, Probe, Time


def _random_uniform(rng: np.random.Generator, dtype, low=0.0, high=1.0, size=None):
    z = np.zeros(size, dtype)
    if np.issubdtype(dtype, np.complexfloating):
        z.real = rng.uniform(low, high, size)
        z.imag = rng.uniform(low, high, size)
    elif np.issubdtype(dtype, np.floating):
        z[...] = rng.uniform(low, high, size)
    else:
        raise NotImplementedError
    return z


def make_delay_and_sum_case_random(dtype_float, dtype_data, amplitudes="random"):
    locations = g.Points(
        np.array([(0, 0, 0), (1.0, 0, 0), (2.0, 0.0, 0.0)], dtype=float)
    )
    numelements = len(locations)
    frequency = 1e6
    probe = Probe(locations, frequency)

    # examination object:
    vel = 10.0
    material = Material(vel)
    examination_object = ExaminationObject(material)

    # timetraces
    # time = Time(start=0.35, step=0.001, num=100)
    time = Time(start=0.0, step=0.001, num=100)

    tx = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    rx = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    numtimetraces = len(tx)

    numpoints = 10

    start_lookup = time.start / 2
    stop_lookup = (time.end - time.step) / 2

    rng = np.random.RandomState(seed=31031596)
    timetraces = _random_uniform(
        rng, dtype_data, 100.0, 101.0, size=(numtimetraces, len(time))
    )
    amplitudes_tx = _random_uniform(
        rng, dtype_data, 1.0, 1.1, size=(numpoints, numelements)
    )
    amplitudes_rx = _random_uniform(
        rng, dtype_data, -1.0, -1.1, size=(numpoints, numelements)
    )
    timetrace_weights = _random_uniform(rng, dtype_data, size=(numtimetraces))
    lookup_times_tx = _random_uniform(
        rng, dtype_float, start_lookup, stop_lookup, (numpoints, numelements)
    )
    lookup_times_rx = _random_uniform(
        rng, dtype_float, start_lookup, stop_lookup, (numpoints, numelements)
    )
    if amplitudes == "random":
        amplitudes = arim.im.tfm.TxRxAmplitudes(amplitudes_tx, amplitudes_rx)
    elif amplitudes == "uniform":
        amplitudes = arim.im.tfm.TxRxAmplitudes(
            np.ones((numpoints, numelements), dtype_data),
            np.ones((numpoints, numelements), dtype_data),
        )
    elif amplitudes == "none":
        amplitudes = None
    else:
        raise ValueError

    # Mess a bit lookup times to get out of bounds values:
    # lookup_times_tx[0, 0] = time.start / 2.
    # lookup_times_rx[1, 1] = time.end * 2.

    focal_law = arim.im.tfm.FocalLaw(
        lookup_times_tx, lookup_times_rx, amplitudes, timetrace_weights
    )

    frame = Frame(timetraces, time, tx, rx, probe, examination_object)

    return frame, focal_law


DATATYPES = OrderedDict()
# dtype_float, dtype_data
DATATYPES["f32"] = (np.float32, np.float32)
DATATYPES["c64"] = (np.float32, np.complex64)
DATATYPES["f64"] = (np.float64, np.float64)
DATATYPES["c128"] = (np.float64, np.complex128)


@pytest.fixture(params=["naive", "numba"])
def das_func(request):
    return getattr(das, "delay_and_sum_" + request.param)


@pytest.fixture(params=tuple(DATATYPES.values()), ids=tuple(DATATYPES.keys()))
def datatypes(request):
    return request.param


@pytest.fixture(params=(0.0, np.nan), ids=("fillvalue_0", "fillvalue_nan"))
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


class TestDasDispatcher:
    def test_amplitudes_uniform_vs_noamp(self, datatypes):
        dtype_float, dtype_data = datatypes
        frame, focal_law = make_delay_and_sum_case_random(
            dtype_float, dtype_data, amplitudes="uniform"
        )

        result = das.delay_and_sum(frame, focal_law)
        assert result.dtype == dtype_data
        reference_result = result.copy()

        frame, focal_law = make_delay_and_sum_case_random(
            dtype_float, dtype_data, amplitudes="none"
        )
        result = das.delay_and_sum(frame, focal_law)
        assert result.dtype == dtype_data

        if dtype_data is np.float32:
            np.testing.assert_allclose(
                result, reference_result, equal_nan=True, rtol=1e-5
            )
        else:
            np.testing.assert_allclose(result, reference_result, equal_nan=True)

    def test_call_das(self, datatypes):
        dtype_float, dtype_data = datatypes
        frame, focal_law = make_delay_and_sum_case_random(
            dtype_float, dtype_data, amplitudes="none"
        )
        res = das.delay_and_sum(frame, focal_law, fillvalue=0.0)
        assert res.dtype == dtype_data
        res = das.delay_and_sum(frame, focal_law, fillvalue=np.nan)
        assert res.dtype == dtype_data
        res = das.delay_and_sum(frame, focal_law, interpolation="nearest")
        assert res.dtype == dtype_data
        res = das.delay_and_sum(frame, focal_law, interpolation=("nearest",))
        assert res.dtype == dtype_data
        res = das.delay_and_sum(frame, focal_law, interpolation="linear")
        assert res.dtype == dtype_data
        res = das.delay_and_sum(frame, focal_law, interpolation=("linear",))
        assert res.dtype == dtype_data
        res = das.delay_and_sum(frame, focal_law, interpolation=("lanczos", 3))
        assert res.dtype == dtype_data
        if dtype_data == np.complex128:
            # If complex128, run normally
            not_impl_typing = contextlib.nullcontext()
        else:
            # Otherwise, run but expect NotImplementedTyping exception
            not_impl_typing = pytest.raises(das.NotImplementedTyping)
        with not_impl_typing:
            res = das.delay_and_sum(
                frame, focal_law, aggregation="median", interpolation="nearest"
            )
            assert res.dtype == dtype_data
            res = das.delay_and_sum(
                frame, focal_law, aggregation="median", interpolation=("lanczos", 3)
            )
            assert res.dtype == dtype_data
            res = das.delay_and_sum(
                frame,
                focal_law,
                aggregation=("huber", 1.5),
                interpolation=("lanczos", 3),
            )
            assert res.dtype == dtype_data

        frame, focal_law = make_delay_and_sum_case_random(
            dtype_float, dtype_data, amplitudes="random"
        )
        res = das.delay_and_sum(frame, focal_law, fillvalue=0.0)
        assert res is not None
        res = das.delay_and_sum(frame, focal_law, fillvalue=np.nan)
        assert res is not None
        res = das.delay_and_sum(frame, focal_law, interpolation="nearest")
        assert res is not None
        res = das.delay_and_sum(frame, focal_law, interpolation="linear")
        assert res is not None
