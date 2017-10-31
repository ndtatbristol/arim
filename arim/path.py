"""
Objects and helpers related to paths and interfaces.

Remark: Interface and Path objects are defined in arim.core
"""
from .core import Mode

# Order by length then by lexicographic order
# Remark: independent views for one array (i.e. consider that view AB-CD is the
# same as view DC-BA).t
IMAGING_MODES = ["L-L", "L-T", "T-T",
                 "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L", "TT-T",
                 "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                 "LT-LT", "LT-TL", "LT-TT",
                 "TL-LT", "TL-TT",
                 "TT-TT"]

DIRECT_PATHS = ['L', 'T']
SKIP_PATHS = ['LL', 'LT', 'TL', 'TT']
DOUBLE_SKIP_PATHS = ['LLL', 'LLT', 'LTL', 'LTT', 'TLL', 'TLT', 'TTL', 'TTT']

L = Mode.L
T = Mode.T


def viewname_order(tx_rx_tuple):
    """
    The views are sorted in ascending order with the following criteria (in this order):

    1) the total number of legs,
    2) the maximum number of legs for transmit and receive paths,
    3) the number of legs for receive path,
    4) the number of legs for transmit path,
    5) lexicographic order for transmit path,
    6) lexicographic order for receive path.

    Parameters
    ----------
    tx_rx_tuple

    Returns
    -------
    order_tuple

    """
    tx, rx = tx_rx_tuple
    return (len(tx) + len(rx), max(len(tx), len(rx)), len(rx), len(tx), tx, rx)


def filter_unique_views(viewnames):
    """
    Remove views that would give the same result because of time reciprocity
    (under linear assumption). Order is unchanged.

    Parameters
    ----------
    viewnames : list[tuple[str]]

    Returns
    -------
    list[tuple[str]]

    Examples
    --------

    >>> filter_unique_views([('AB', 'CD'), ('DC', 'BA'), ('X', 'YZ'), ('ZY', 'X')])
    ... [('AB', 'CD'), ('X', 'YZ')]

    """
    unique_views = []
    seen_so_far = set()
    for view in viewnames:
        tx, rx = view
        rev_view = (rx[::-1], tx[::-1])
        if rev_view in seen_so_far:
            continue
        else:
            seen_so_far.add(view)
            unique_views.append(view)
    return unique_views


def make_viewnames(pathnames, unique_only=True, order_func=viewname_order):
    """
    Parameters
    ----------
    pathnames : list[str]
    unique_only : bool
        If True, consider Default True.
    order_func : func

    Returns
    -------
    list[tuple[str]

    """
    viewnames = []
    for tx in pathnames:
        for rx in pathnames:
            viewnames.append((tx, rx))

    if order_func is not None:
        viewnames = list(sorted(viewnames, key=viewname_order))

    if unique_only:
        viewnames = filter_unique_views(viewnames)

    return viewnames
