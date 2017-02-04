import math
from itertools import zip_longest

import numpy as np

from ..exceptions import NotAnArray, InvalidDimension, InvalidShape

__all__ = ['linspace2', 'get_shape_safely', 'chunk_array', 'smallest_uint_that_fits']


def linspace2(start, step, num, dtype=None):
    """
    Return a linearly spaced vector.

    Parameters
    ----------
    start : scalar
        Starting value of the sequence.
    step : float, optional
        Size of spacing between samples.
    num : int
        Number of samples to generate.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    samples : ndarray
        Linearly spaced vector ``[start, stop]`` where ``end = start + (num - 1)*step``


    Examples
    --------
    >>> linspace2(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])

    Notes
    -----

    Adapted from ``numpy.linspace``
    (License: http://www.numpy.org/license.html)

    """
    num = int(num)
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)

    # Convert float/complex array scalars to float
    start = start * 1.
    step = step * 1.

    dt = np.result_type(start, step)
    if dtype is None:
        dtype = dt

    y = np.arange(0, num, dtype=dt)

    if num > 1:
        y *= step

    y += start

    return y.astype(dtype, copy=False)


def get_shape_safely(array, array_name, expected_shape=None):
    """
    Return the shape of an array.
    Raise ``NotAnArray`` if the so-called array has no attribute shape.

    If an expected is given, check that the array shape is indeed compatible. ``expected_shape`` must be a tuple
    of integers or 'None'. If 'None' is given for a dimension, this dimension is ignored.

    """
    try:
        shape = array.shape
    except AttributeError:
        raise NotAnArray(array_name)

    if expected_shape is None:
        return shape

    # Check shape if expected_shape was provided:
    if len(shape) != len(expected_shape):
        raise InvalidDimension.message_auto(array_name, len(expected_shape), len(shape))
    for (dim, (expected_size, current_size)) in enumerate(zip(expected_shape, shape),
                                                          start=1):
        if expected_size is None:
            continue
        if expected_size != current_size:
            raise InvalidShape(
                "Array '{}' must have a size of {} (current: {}) for its dimension {}."
                .format(array_name, expected_size, current_size, dim))

    return shape


def chunk_array(array_shape, block_size, axis=0):
    """Yield selectors to split a array into multiple chunk.

        >>> x = np.arange(10)
        >>> for sel in chunk_array(x.shape, 3):
        ...     print(x[sel])
        [0 1 2]
        [3 4 5]
        [6 7 8]
        [9]


    Parameters
    ----------
    array_shape : tuple
        Shape of the array to split.
    block_size : iterable or int
        Number of items in each block (except the latest which might have less).
    axis : int, optional
        Split axis. Default: 0

    """
    ndim = len(array_shape)
    axis = list(range(ndim))[axis]  # works if axis is positive or negative
    length = array_shape[axis]

    numchunks = math.ceil(length / block_size)

    if axis == 0:
        for i in range(numchunks):
            yield (slice(i * block_size, (i + 1) * block_size), ...)
    elif axis == (ndim - 1):
        for i in range(numchunks):
            yield (..., slice(i * block_size, (i + 1) * block_size))
    else:
        fillers = (slice(None),) * axis
        for i in range(numchunks):
            yield (*fillers, slice(i * block_size, (i + 1) * block_size), ...)


def smallest_uint_that_fits(max_value):
    """Return the smallest unsigned integer datatype (dtype) such as all numbers
    between 0 and 'max_value' can be stored without overflow."""
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        allowed_max_value = np.iinfo(dtype).max
        if max_value <= allowed_max_value:
            return dtype
    return TypeError("Cannot stored '{}' with numpy (max: '{}')"
                     .format(max_value, allowed_max_value))
