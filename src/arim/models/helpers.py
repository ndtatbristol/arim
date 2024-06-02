from collections import OrderedDict

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
    viewnames = ut.make_viewnames(paths_dict.keys(), tfm_unique_only=tfm_unique_only)
    views = OrderedDict()
    for view_name_tuple in viewnames:
        tx_name, rx_name = view_name_tuple
        view_name = "{}-{}".format(tx_name, rx_name)
    
        tx_path = paths_dict[tx_name]
        # to get the receive path: return the string of the corresponding transmit path
        rx_path = paths_dict[rx_name[::-1]]
    
        views[view_name] = c.View(tx_path, rx_path, view_name)
    return views
