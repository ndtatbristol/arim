import numpy as np
import pytest
import scipy.signal

from arim import Time, signal


def test_butterworth_bandpass():
    dt = 1 / 25e6
    time = Time(0, dt, 1000)

    f1 = 0.5e-6
    f2 = 1.0e-6

    # Mix f1 and f2
    x_raw = np.sin(time.samples * 2 * np.pi * f1)
    x_all = x_raw + np.sin(time.samples * 2 * np.pi * f2)

    # Objective: get f1 without f2
    filt = signal.ButterworthBandpass(
        order=3, cutoff_min=f1 / 2, cutoff_max=(f1 + f2) / 2, time=time
    )

    # This should work without error:
    str(filt)
    repr(filt)

    x_filt = filt(x_all)

    assert np.allclose(x_filt, x_raw, atol=0.001)


def test_gaussian_bandpass():
    dt = 1 / 5e6
    time = Time(0, dt, 20)

    f0 = 1e6

    # np.random.seed(123)
    # x = np.cos(2 * np.pi * f0 * time.samples) + np.random.uniform(-1., 1., size=len(time))
    x = np.array(
        [
            1.39293837,
            -0.11870434,
            -1.35531409,
            -0.70638746,
            0.74795493,
            0.84621292,
            1.27054539,
            -0.43935752,
            -0.84715319,
            0.09325203,
            0.68635603,
            0.76711641,
            -0.93187251,
            -1.6896612,
            0.10510551,
            1.47599081,
            -0.32599954,
            -1.45811348,
            -0.74591425,
            0.37267217,
        ]
    )

    half_bandwidth = 0.9
    filt = signal.Gaussian(len(time), f0, f0 * half_bandwidth, time)

    # This should work without error:
    str(filt)
    repr(filt)

    x_filt = filt(x)

    x_filt_ref = np.array(
        [
            0.62591615 + 0.16094794j,
            0.05282459 + 0.63433225j,
            -0.56150642 + 0.26702298j,
            -0.4464416 - 0.40377986j,
            0.17969486 - 0.55062043j,
            0.55312903 - 0.06575002j,
            0.27794359 + 0.46086085j,
            -0.31166247 + 0.42185181j,
            -0.49724934 - 0.14825403j,
            -0.00762617 - 0.52544998j,
            0.52277535 - 0.15681544j,
            0.30321322 + 0.48852171j,
            -0.4158227 + 0.43818666j,
            -0.54502171 - 0.30731385j,
            0.1758916 - 0.61337811j,
            0.64375842 + 0.03294709j,
            0.11810955 + 0.6382037j,
            -0.5917696 + 0.27483013j,
            -0.42575264 - 0.49615411j,
            0.34959627 - 0.55018928j,
        ]
    )
    np.testing.assert_allclose(x_filt, x_filt_ref)

    x_filt_2d = filt([x, 2 * x, 3 * x])
    np.testing.assert_allclose(x_filt_2d, [x_filt_ref, 2 * x_filt_ref, 3 * x_filt_ref])


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
    assert composed(x) == 16.0


@pytest.mark.parametrize(
    "shape,axis",
    [
        (11, -1),
        (10, -1),
        ((20, 10), -1),
        ((20, 11), -1),
        ((20, 10), 0),
        ((21, 10), 0),
        (1, -1),
    ],
)
def test_rfft_to_hilbert(shape, axis):
    x = np.random.uniform(size=shape)

    y = np.fft.rfft(x, axis=axis)
    n = np.atleast_1d(shape)[axis]

    z = signal.rfft_to_hilbert(y, n, axis=axis)
    z_desired = scipy.signal.hilbert(x, axis=axis)

    np.testing.assert_allclose(z, z_desired)
