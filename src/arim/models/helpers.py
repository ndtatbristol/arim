from collections import OrderedDict
from itertools import product

from .. import core as c
from .. import ut


def make_views_from_paths(paths_dict, tfm_unique_only=False):
    """
    Returns 'View' objects for the case of a block in immersion.

    Consut all possible views that can be constructed with the paths given as argument.

    If unique only ``unique_only`` is false,

    Parameters
    ----------
    paths_dict : Dict[Path]
        Key: path names (exemple: 'L', 'LT'). Values: :class:`Path`
    tfm_unique_only : bool
        Default: False. If True, returns only the views that give *different* imaging
        results with TFM (AB-CD and DC-BA give the same imaging result).

    Returns
    -------
    views: OrderedDict[Views]

    """
    views = OrderedDict()
    for tx_path, rx_path in product(paths_dict.values(), paths_dict.values()):
        view_name = f"{tx_path.longname} - {rx_path.reverse_longname}"
        if tfm_unique_only and (f"{rx_path.longname} - {tx_path.reverse_longname}" in views.keys()):
            continue
        views[view_name] = c.View(tx_path, rx_path, view_name)
        
    return views
