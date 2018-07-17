"""
Helper for configuring scripts
"""

import pprint
import re
import copy
import collections

__all__ = ["Config"]


class Config(dict):
    """
    Configuration object

    A dictionary object that shows its values by alphabetical order.

    Notes
    -----
    Adapted from matplotlib.RcParams (BSD License)

    """

    def __repr__(self):
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(dict(self), indent=1, width=80 - indent).split("\n")
        repr_indented = ("\n" + " " * indent).join(repr_split)
        return "{0}({1})".format(class_name, repr_indented)

    def __str__(self):
        return "\n".join("{0}: {1}".format(k, v) for k, v in sorted(self.items()))

    def keys(self):
        """
        Return sorted list of keys.
        """
        k = list(super().keys())
        k.sort()
        return k

    def values(self):
        """
        Return values in order of sorted keys.
        """
        return [self[k] for k in self.keys()]

    def find_all(self, pattern):
        """
        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        .. note::

            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.

        """
        pattern_re = re.compile(pattern)
        return self.__class__(
            (key, value) for key, value in self.items() if pattern_re.search(key)
        )

    def copy(self):
        """
        Returns a deep copy of the object.
        """
        return copy.deepcopy(self)

    def merge(self, conf):
        """
        Merge the dict-like parameter into the current object.
        This is a recursive update.

        Parameters
        ----------
        conf : dict or None
            Dictionary or Config object. If None, do nothing.

        Returns
        -------
        None

        Notes
        -----
        Adapted from `configobj <https://github.com/DiffSK/configobj/>`_, license BSD 3-clause
        """
        if conf is None:
            return self
        return recursive_dict_merge(self, conf)


def recursive_dict_merge(base_dict, top_dict):
    """
    Merge `top_dict` to `base_dict`. This is a recursive version of::

    base_dict.update(top_dict)
    """
    for key, val in list(top_dict.items()):
        if (
            key in base_dict
            and isinstance(base_dict[key], collections.Mapping)
            and isinstance(val, collections.Mapping)
        ):
            recursive_dict_merge(base_dict[key], val)
        else:
            base_dict[key] = val
