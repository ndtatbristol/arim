import collections.abc
from collections import OrderedDict, namedtuple

ProbeMaker = namedtuple("ProbeMaker", "make short_name long_name")


class ProbeRegistry(collections.abc.Mapping):
    """
    Dict-like that create a new object at each call. This prevents accidental changes in a probe.
    """

    def __init__(self):
        self._makers = OrderedDict()

    def register(self, probe_maker):
        key = probe_maker.short_name
        if key in self:
            raise KeyError(
                f"Key '{key}' already exist in registry. Unregister it first to rewrite it"
            )
        self._makers[key] = probe_maker

    def unregister(self, key):
        del self._makers[key]

    def __len__(self):
        return len(self._makers)

    def __iter__(self):
        return iter(self._makers)

    def __getitem__(self, item):
        return self._makers[item].make()

    def __str__(self):
        s = "Available probes:\n"
        s += "----------------\n"
        for key, maker in self._makers.items():
            s += f"  - {maker.long_name} (id: {maker.short_name})\n"
        return s

    def keys(self):
        return self._makers.keys()
