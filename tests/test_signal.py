import numpy as np
import scipy.signal
import pytest

from arim import signal
from arim import Time


def test_butterworth_bandpass():
    dt = 1 / 25e6
    time = Time(0, dt, 1000)

    f1 = 0.5e-6
    f2 = 1.0e-6

    # Mix f1 and f2
    x_raw = np.sin(time.samples * 2 * np.pi * f1)
    x_all = x_raw + np.sin(time.samples * 2 * np.pi * f2)

    # Objective: get f1 without f2
    filt = signal.ButterworthBandpass(order=3, cutoff_min=0, cutoff_max=(f1 + f2) / 2,
                                      time=time)

    # This should work without error:
    str(filt)
    repr(filt)

    x_filt = filt(x_all)

    assert np.allclose(x_filt, x_raw, atol=0.001)


def test_composed_filters():
    class MultiplyBy2(signal.Filter):
        def __call__(self, arr):
            return arr * 2.0

    class Add3(signal.Filter):
        def __call__(self, arr):
            return arr + 3.0

    class Substract1(signal.Filter):
        def __call__(self, arr):
            return arr - 1.0

    x = 5.0

    multiply2 = MultiplyBy2()
    add3 = Add3()
    substract1 = Substract1()

    assert isinstance(multiply2 + add3, signal.ComposedFilter)
    assert isinstance(add3 + multiply2, signal.ComposedFilter)
    assert isinstance(add3 + (multiply2 + substract1), signal.ComposedFilter)
    assert isinstance((add3 + multiply2) + substract1, signal.ComposedFilter)
    assert len(add3 + multiply2 + substract1) == 3

    assert multiply2(x) == 10.0
    assert add3(x) == 8.0
    assert (multiply2 + add3)(x) == 16.0
    assert (add3 + multiply2)(x) == 13.0
    assert ((add3 + multiply2) + substract1)(x) == 11.0
    assert (add3 + (multiply2 + substract1))(x) == 11.0

    composed = multiply2 + add3


@pytest.mark.parametrize("shape,axis", [(11, -1), (10, -1), ((20, 10), -1), ((20, 11), -1),
                                        ((20, 10), 0), ((21, 10), 0), (1, -1)])
def test_rfft_to_hilbert(shape, axis):
    x = np.random.uniform(size=shape)

    y = np.fft.rfft(x, axis=axis)
    n = np.atleast_1d(shape)[axis]

    z = signal.rfft_to_hilbert(y, n, axis=axis)
    z_desired = scipy.signal.hilbert(x, axis=axis)

    np.testing.assert_allclose(z, z_desired)
