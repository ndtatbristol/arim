import time
from contextlib import contextmanager
import sys
import logging

__all__ = ['get_name', 'parse_enum_constant', 'timeit']


def get_name(metadata):
    """Return the name of an object based on the dictionary metadata. By preference: long_name, short_name, 'Unnamed'
    """
    name = metadata.get('long_name', None)
    if name is not None:
        return name

    name = metadata.get('short_name', None)
    if name is not None:
        return name

    return 'Unnamed'


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
                    x=enum_constant_or_name, enum_type=enum_type))


@contextmanager
def timeit(name='Computation', logger=None, log_level=logging.INFO):
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

        >>> with arim.utils.timeit('Simple addition'):
        ...     1 + 1
        Simple addition performed in 570.20 ns

    Using a logger::
        >>> with arim.utils.timeit('Simple addition', logger=logger):
        >>>     1 + 1

    """
    if sys.platform == 'win32':
        # On Windows, the best timer is time.clock
        default_timer = time.clock
    else:
        # On most other platforms the best timer is time.time
        default_timer = time.time

    tic = default_timer()
    yield
    elapsed = default_timer() - tic

    if elapsed < 1e-6:
        elapsed = elapsed * 1e9
        unit = 'ns'
    elif elapsed < 1e-3:
        elapsed = elapsed * 1e6
        unit = 'us'
    elif elapsed < 1:
        elapsed = elapsed * 1000
        unit = 'ms'
    else:
        unit = 's'

    msg_format = '{name} performed in {elapsed:.2f} {unit}'
    msg = msg_format.format(name=name, elapsed=elapsed, unit=unit)

    if logger is None:
        print(msg)
    else:
        logger.log(log_level, msg)
