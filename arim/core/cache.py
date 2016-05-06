from warnings import warn
from collections import Counter

from ..exceptions import ArimWarning

__all__ = ['Cache', 'NoCache']


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
