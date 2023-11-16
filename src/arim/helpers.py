"""
Helper functions
"""

import logging
import math
import os
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from warnings import warn

import numpy as np

from .exceptions import ArimWarning, InvalidDimension, InvalidShape, NotAnArray


def get_name(metadata):
    """Return the name of an object based on the dictionary metadata. By preference: long_name, short_name, 'Unnamed'"""
    name = metadata.get("long_name", None)
    if name is not None:
        return name

    name = metadata.get("short_name", None)
    if name is not None:
        return name

    return "Unnamed"


def parse_enum_constant(enum_constant_or_name, enum_type):
    """
    Return the enumerated constant corresponding to 'enum_constant_or_name', which
    can be either this constant or a its name (string).
    """
    if isinstance(enum_constant_or_name, enum_type):
        return enum_constant_or_name
    else:
        try:
            return enum_type[enum_constant_or_name]
        except KeyError:
            raise ValueError(
                "Expected a constant of enum '{enum_type}', got '{x}' instead".format(
                    x=enum_constant_or_name, enum_type=enum_type
                )
            )


@contextmanager
def timeit(name="Computation", logger=None, log_level=logging.INFO):
    """
    A context manager for timing some code.

    Parameters
    ----------
    name : str
        Name of the computation
    logger : logging.Logger or None
        Logger where to write the elapsed time. If None (default), use function ``print()``
    log_level : int
        Level logger (used only if a logger is given).

    Returns
    -------
    None

    Examples
    --------
    ::

        >>> with arim.helpers.timeit('Simple addition'):
        ...     1 + 1
        Simple addition performed in 570.20 ns

    Using a logger::
        >>> with arim.helpers.timeit('Simple addition', logger=logger):
        >>>     1 + 1

    """
    default_timer = time.perf_counter
    tic = default_timer()
    yield
    elapsed = default_timer() - tic

    if elapsed < 1e-6:
        elapsed = elapsed * 1e9
        unit = "ns"
    elif elapsed < 1e-3:
        elapsed = elapsed * 1e6
        unit = "us"
    elif elapsed < 1:
        elapsed = elapsed * 1000
        unit = "ms"
    else:
        unit = "s"

    msg_format = "{name} performed in {elapsed:.2f} {unit}"
    msg = msg_format.format(name=name, elapsed=elapsed, unit=unit)

    if logger is None:
        print(msg)
    else:
        logger.log(log_level, msg)


class Cache(dict):
    """
    Dict-like which keeps track of which values were retrieved and how many
    times.

    Attributes
    ----------
    counter: Counter
    hits: int

    """

    def __init__(self):
        self.counter = Counter()
        self.hits = 0
        self.misses = 0
        super().__init__()

    def clear(self):
        super().clear()
        self.counter.clear()
        self.hits = 0
        self.misses = 0

    def __getitem__(self, key):
        # Preventively, we consider we have a miss until we are sure we got a hits.
        self.misses += 1
        out = super().__getitem__(key)  # this line may raise an exception

        # At this point no exception was raised so it's a hits:
        self.misses -= 1
        self.hits += 1
        self.counter.update([key])

        return out

    def __setitem__(self, key, value):
        if key in self:
            msg = f"Reassigning a cached value: key={key}"
            warn(msg, ArimWarning, stacklevel=2)
        super().__setitem__(key, value)

    def stat(self):
        print(
            "{}: {} values cached, {} hits, {} misses".format(
                self.__class__.__name__, len(self), self.hits, self.misses
            )
        )
        print(f"\tBest cached: {self.counter.most_common()}")

    def get(self, key, default=None):
        out = super().get(key, default)
        if out is default:
            self.misses += 1
        else:
            self.hits += 1
        return out


class NoCache(Cache):
    """
    Looks like a cache but actually unable to retain anything.
    """

    def __init__(self):
        self.ignored = 0
        super().__init__()

    def __setitem__(self, key, value):
        self.ignored += 1


def get_git_version(short=True):
    """
    Returns the current git revision as a string. Returns an empty string
    if git is not available or if the library is not not in a repository.
    """
    curdir = os.getcwd()
    filedir, _ = os.path.split(__file__)
    os.chdir(filedir)

    if short:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
    else:
        cmd = ["git", "rev-parse", "HEAD"]

    try:
        githash = subprocess.check_output(cmd)
        githash = githash.decode("ascii").strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        githash = ""

    os.chdir(curdir)
    return githash


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
    for dim, (expected_size, current_size) in enumerate(
        zip(expected_shape, shape), start=1
    ):
        if expected_size is None:
            continue
        if expected_size != current_size:
            raise InvalidShape(
                "Array '{}' must have a size of {} (current: {}) for its dimension {}.".format(
                    array_name, expected_size, current_size, dim
                )
            )

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
    return TypeError(
        f"Cannot stored '{max_value}' with numpy (max: '{allowed_max_value}')"
    )


def sizeof_fmt(num, suffix="B"):
    """
    Human-readable memory size.

    Adapted from https://stackoverflow.com/a/1094933/2996578
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)  # noqa
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)  # noqa
