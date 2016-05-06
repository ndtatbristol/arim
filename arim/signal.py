"""
Module for signal processing.

"""

from enum import Enum

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

__all__ = ['Filter', 'ButterworthBandpass', 'ComposedFilter', 'Hilbert', 'NoFilter', 'Abs']


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
        return 'No filter'


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

        self.b, self.a = butter(order, Wn, btype='bandpass')

    def __str__(self):
        return '{} [{:.1f}, {:.1f}] MHz order {}'.format(self.__class__.__qualname__,
                                                         self.cutoff_min * 1e-6, self.cutoff_max * 1e-6, self.order)

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
        return 'Hilbert transform'

class Abs(Filter):
    """
    Returns the absolute value of a signal.
    """
    def __call__(self, arr):
        return np.abs(arr)

    def __str__(self):
        return "Absolute value"
