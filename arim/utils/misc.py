import os
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
import sys
import logging
from warnings import warn

from arim.exceptions import ArimWarning

__all__ = ['get_name', 'parse_enum_constant', 'timeit', 'Cache', 'NoCache',
           'get_git_version']


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
            msg = "Reassigning a cached value: key={}".format(key)
            warn(msg, ArimWarning)
        super().__setitem__(key, value)

    def stat(self):
        print("{}: {} values cached, {} hits, {} misses"
              .format(self.__class__.__name__, len(self), self.hits, self.misses))
        print("\tBest cached: {}".format(self.counter.most_common()))

    def get(self, key, default=None):
        out = super(Cache, self).get(key, default)
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
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
    else:
        cmd = ['git', 'rev-parse', 'HEAD']

    try:
        githash = subprocess.check_output(cmd)
        githash = githash.decode('ascii').strip()
    except (FileNotFoundError, subprocess.CalledProcessError)  as e:
        githash = ''

    os.chdir(curdir)
    return githash
