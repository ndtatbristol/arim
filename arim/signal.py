"""
Module for signal processing.

"""

from enum import Enum
import cmath
import numba

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import scipy.fftpack

__all__ = [
    "Filter",
    "ButterworthBandpass",
    "ComposedFilter",
    "Hilbert",
    "NoFilter",
    "Abs",
    "Gaussian",
    "rfft_to_hilbert",
    "timeshift_spectra",
]


class Filter:
    """
    Abstract filter.

    To implement a new filter, create a derived class and implement the following method such as:

      - ``__init__`` initialiases the filter (take as many arguments as required),
      - ``__call__`` actually does something on the data (take as argument the data to filter),
      - ``__str__`` returns a description of the filter.

    Filters can be composed by using the ``+`` operator.

    """

    def __add__(self, inner_filter):
        """Composition operator for Filter objects."""
        return ComposedFilter(self, inner_filter)

    def __call__(self, *args, **kwargs):
        """Apply the filter on data; to implement in derived class."""
        raise NotImplementedError

    def __str__(self):
        """Description of the filter; to implement in derived class."""
        return "Unspecified filter"


class NoFilter(Filter):
    """
    A filter that does nothing (return data unchanged).
    """

    def __call__(self, arr):
        return arr

    def __str__(self):
        return "No filter"


class ComposedFilter(Filter):
    """
    Composed filter.

    When called, this filter applies each of its subfilters on the data.
    """

    def __init__(self, outer_filters, inner_filters):
        try:
            # If outer_filters is a composed filter:
            outer_ops = outer_filters.ops
        except AttributeError:
            # If outer_filters is a single filter:
            outer_ops = [outer_filters]

        try:
            inner_ops = inner_filters.ops
        except AttributeError:
            inner_ops = [inner_filters]

        self.ops = outer_ops + inner_ops

    def __len__(self):
        return len(self.ops)

    def __call__(self, arr, **kwargs):
        """

        Parameters
        ----------
        arr
            Array to process
        kwargs: dictionary
            Arguments to pass to the __call__ method of each part of the composed filter. Must be indexed by
            the instance of the filter.

        Returns
        -------
        filtered_arr

        """
        out = arr
        for op in reversed(self.ops):
            try:
                op_kwargs = kwargs.pop(op)
            except KeyError:
                out = op(out)
            else:
                out = op(out, **op_kwargs)
        if len(kwargs) != 0:
            raise ValueError("Unexpected keys: {}".format(kwargs.keys()))
        return out

    def __str__(self):
        return "\n".join([str(op) for op in self.ops])


class ButterworthBandpass(Filter):
    """
    Butterworth bandpass filter.

    Parameters
    ----------
    order : int
        Order of the filter
    cutoff_min, cutoff_max : float
        Cutoff frequencies in Hz.
    time : arim.Time
        Time object. This filter can be used only on data sampled consistently with the attribute
    ``time``.

    """

    def __init__(self, order, cutoff_min, cutoff_max, time):
        nyquist = 0.5 / time.step
        cutoff_min = cutoff_min * 1.0
        cutoff_max = cutoff_max * 1.0

        Wn = np.array([cutoff_min, cutoff_max]) / nyquist

        self.order = order
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

        self.b, self.a = butter(order, Wn, btype="bandpass")

    def __str__(self):
        return "{} [{:.1f}, {:.1f}] MHz order {}".format(
            self.__class__.__qualname__,
            self.cutoff_min * 1e-6,
            self.cutoff_max * 1e-6,
            self.order,
        )

    def __call__(self, arr, axis=-1, **kwargs):
        """
        Apply the filter on array with ``scipy.signal.filtfilt`` (zero-phase filtering).

        Parameters
        ----------
        arr
        axis
        kwargs: extra arguments for

        Returns
        -------
        filtered_arr

        """
        return np.ascontiguousarray(filtfilt(self.b, self.a, arr, axis=axis, **kwargs))

    def __repr__(self):
        return "<{} at {}>".format(str(self), hex(id(self)))


class Hilbert(Filter):
    """
    Returns the analytical signal, i.e. ``signal + i * hilbert_signal`` where
    ``hilbert_signal`` is the Hilbert transform of ``signal``.
    """

    def __call__(self, arr, axis=-1):
        return hilbert(arr, axis=axis)

    def __str__(self):
        return "Hilbert transform"


class Abs(Filter):
    """
    Returns the absolute value of a signal.
    """

    def __call__(self, arr):
        return np.abs(arr)

    def __str__(self):
        return "Absolute value"


class Gaussian(Filter):
    """ 
    Gaussian Filter - As applied in BRAIN **BUT** default is zero outside of filter region, BRAIN is not.

    Return the analytical signal
    
    Parameters
    ----------
    nsamples : int
        ``len(time)``
    centre_freq : float
        In Hz
    half_bandwidth : float
        In Hz
    time : arim.Time
        Time object. This filter can be used only on data sampled consistently with the attribute
    ``time``.
    force_zero : bool
        If True (default), the spectrum amplitudes below ``-db_down`` will be
        replaced by exactly zero.
    db_down : float

    """

    def __init__(
        self, nsamples, centre_freq, half_bandwidth, time, force_zero=True, db_down=40.0
    ):

        fract = np.power(10, -db_down / 20.0)
        max_freq = 1.0 / (time.step)
        peak_pos_fract = centre_freq / max_freq
        half_width_fract = half_bandwidth / max_freq
        r = np.arange(nsamples) / (nsamples - 1) - peak_pos_fract
        r1 = half_width_fract / (np.sqrt(-np.log(fract)))
        self.samples = nsamples
        self.centre_freq = centre_freq
        self.half_bandwidth = half_bandwidth
        self.max_freq = max_freq
        self.filter_window = np.exp(-np.power(r / r1, 2))
        # print('Gaussian')
        if force_zero:
            self.filter_window[self.filter_window < fract] = 0

    def __str__(self):
        return "{} [{:.1f}, {:.1f}] MHz order {}".format(
            self.__class__.__qualname__,
            self.max_freq * 1e-6,
            self.half_bandwidth * 1e-6,
            self.max_freq * 1e-6,
        )

    def __call__(self, arr):
        arr = np.asarray(arr)
        # broadcast window to (1, 1, ..., numsamples)
        window = np.array(self.filter_window, ndmin=arr.ndim)
        return np.fft.ifft(np.fft.fft(arr) * window)


def rfft_to_hilbert(xf, n, axis=-1):
    """
    Convert the Fourier transform of a real signal to the analytic signal.

    This is equivalent but faster than doing::

        scipy.signal.hilbert(np.fft.irfft(xf, n))

    where typically ::

        xf = np.fft.rfft(x)
        n = len(xf)

    Convert the positive frequency part as the spectrum, as obtained with ``numpy.fft.rfft``,

    Parameters
    ----------
    xf : ndarray
        Input array
    n : int
        Length of the time domain signal
    axis : int
        Default: -1

    Returns
    -------
    out : complex ndarray

    """
    # cf code of https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    if xf.ndim == 0:
        h = 1.0
    else:
        h = np.zeros(xf.shape[axis])
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2

    if xf.ndim > 1:
        ind = [np.newaxis] * xf.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    return scipy.fftpack.ifft(h * xf, n, axis)


@numba.guvectorize(
    [(numba.float64[:], numba.complex128[:], numba.float64[:], numba.complex128[:])],
    "(),(),(numfreq)->(numfreq)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _timeshift_spectra_singlef(delays, unshifted_x, freq_array, out=None):
    for freq_idx in range(freq_array.shape[0]):
        out[freq_idx] = (
            cmath.exp(-2j * np.pi * freq_array[freq_idx] * delays[0]) * unshifted_x[0]
        )


@numba.guvectorize(
    [(numba.float64[:], numba.complex128[:], numba.float64[:], numba.complex128[:])],
    "(),(numfreq),(numfreq)->(numfreq)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _timeshift_spectra_multif(delays, unshifted_x, freq_array, out=None):
    for freq_idx in range(freq_array.shape[0]):
        out[freq_idx] = (
            cmath.exp(-2j * np.pi * freq_array[freq_idx] * delays[0])
            * unshifted_x[freq_idx]
        )


def timeshift_spectra(unshifted_x, delays, freq_array):
    """Time-shift spectra in frequency domain

    Case ``num_x_freq=numfreq``: returns::

        X(omega) exp(-i omega delay)

    Case ``num_x_freq=1``: returns::

        X(omega_0) exp(-i omega delay)


    Parameters
    ----------
    unshifted_x : ndarray
        Shape (shape1, num_x_freq)
    delays : ndarray
        Shape (shape1)
    freq_array : ndarray
        Shape (numfreq)

    Returns
    -------
    shifted_x
        Shape (shape1, numfreq)

    """
    num_tf_freq = unshifted_x.shape[-1]

    if num_tf_freq == 1:
        return _timeshift_spectra_singlef(delays, unshifted_x[..., 0], freq_array)
    else:
        return _timeshift_spectra_multif(delays, unshifted_x, freq_array)
