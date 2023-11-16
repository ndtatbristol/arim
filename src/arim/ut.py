"""
Toolbox of functions for ultrasonic testing/acoustics.
"""
# Only function that does not require any arim-specific logic should be put here.
# This module must be kept free of any arim dependencies because so that it could be used
# without arim.

import warnings

import numpy as np


class UtWarning(UserWarning):
    pass


def fmc(numelements):
    """
    Return all pairs of elements for a FMC.
    HMC as performed by Brain.

    Returns
    -------
    tx : ndarray [numelements^2]
        Transmitter for each timetrace: 0, 0, ..., 1, 1, ...
    rx : ndarray
        Receiver for each timetrace: 1, 2, ..., 1, 2, ...
    """
    numelements = int(numelements)
    elements = np.arange(numelements)

    # 0 0 0    1 1 1    2 2 2
    tx = np.repeat(elements, numelements)

    # 0 1 2    0 1 2    0 1 2
    rx = np.tile(elements, numelements)
    return tx, rx


def hmc(numelements):
    """
    Return all pairs of elements for a HMC.
    HMC as performed by Brain (rx >= tx)

    Returns
    -------
    tx : ndarray [numelements^2]
        Transmitter for each timetrace: 0, 0, 0, ..., 1, 1, 1, ...
    rx : ndarray
        Receiver for each timetrace: 0, 1, 2, ..., 1, 2, ...
    """
    numelements = int(numelements)
    elements = np.arange(numelements)

    # 0 0 0    1 1    2
    tx = np.repeat(elements, range(numelements, 0, -1))

    # 0 1 2    0 1    2
    rx = np.zeros_like(tx)
    take_n_last = np.arange(numelements, 0, -1)
    start = 0
    for n in take_n_last:
        stop = start + n
        rx[start:stop] = elements[-n:]
        start = stop
    return tx, rx


def infer_capture_method(tx, rx):
    """
    Infers the capture method from the indices of transmitters and receivers.

    Returns: 'hmc', 'fmc', 'unsupported'

    Parameters
    ----------
    tx : list
        One per timetrace
    rx : list
        One per timetrace

    Returns
    -------
    capture_method : string
    """
    numelements = max(np.max(tx), np.max(rx)) + 1
    assert len(tx) == len(rx)

    # Get the unique combinations tx/rx of the input.
    # By using set, we ignore the order of the combinations tx/rx.
    combinations = set(zip(tx, rx))

    # Could it be a HMC? Most frequent case, go first.
    # Remark: HMC can be made with tx >= rx or tx <= rx. Check both.
    tx_hmc, rx_hmc = hmc(numelements)
    combinations_hmc1 = set(zip(tx_hmc, rx_hmc))
    combinations_hmc2 = set(zip(rx_hmc, tx_hmc))

    if (len(tx_hmc) == len(tx)) and (
        (combinations == combinations_hmc1) or (combinations == combinations_hmc2)
    ):
        return "hmc"

    # Could it be a FMC?
    tx_fmc, rx_fmc = fmc(numelements)
    combinations_fmc = set(zip(tx_fmc, rx_fmc))
    if (len(tx_fmc) == len(tx)) and (combinations == combinations_fmc):
        return "fmc"

    # At this point we are hopeless
    return "unsupported"


def default_timetrace_weights(tx, rx):
    """
    timetrace weights for TFM.

    Consider a timetrace obtained by the transmitter i and the receiver j; this
    timetrace is denoted (i,j). If the response matrix contains both (i, j) and (j, i),
    the corresponding timetrace weight is 1. Otherwise, the timetrace weight is 2.

    Example: for a FMC, all timetrace weights are 1.
    Example: for a HMC, timetrace weights for the pulse-echo timetraces are 1,
    timetrace weights for the non-pulse-echo timetraces are 2.

    Remark: the function does not check if there are duplicated signals.

    Parameters
    ----------
    tx : list[int] or ndarray
        tx[i] is the index of the transmitter (between 0 and numelements-1) for
        the i-th timetrace.
    rx : list[int] or ndarray
        rx[i] is the index of the receiver (between 0 and numelements-1) for
        the i-th timetrace.

    Returns
    -------
    timetrace_weights : ndarray

    """
    if len(tx) != len(rx):
        raise ValueError("tx and rx must have the same lengths (numtimetraces)")
    numtimetraces = len(tx)

    # elements_pairs contains (tx[0], rx[0]), (tx[1], rx[1]), etc.
    elements_pairs = {*zip(tx, rx)}
    timetrace_weights = np.ones(numtimetraces)
    for this_tx, this_rx, timetrace_weight in zip(
        tx, rx, np.nditer(timetrace_weights, op_flags=["readwrite"])
    ):
        if (this_rx, this_tx) not in elements_pairs:
            timetrace_weight[...] = 2.0
    return timetrace_weights


def default_scanline_weights(tx, rx):
    warnings.warn(
        DeprecationWarning(
            "default_scanline_weights is deprecated. Use default_timetrace_weights"
        )
    )
    return default_timetrace_weights(tx, rx)


def decibel(arr, reference=None, neginf_value=-1000.0, return_reference=False):
    """
    Return 20*log10(abs(arr) / reference)

    If reference is None, use:

        reference := max(abs(arr))

    Parameters
    ----------
    arr : ndarray
        Values to convert in dB.
    reference : float or None
        Reference value for 0 dB. Default: None
    neginf_value : float or None
        If not None, convert -inf dB values to this parameter. If None, -inf
        dB values are not changed.
    return_max : bool
        Default: False.

    Returns
    -------
    arr_db
        Array in decibel.
    arr_max: float
        Return ``max(abs(arr))``. This value is returned only if return_max is true.

    """
    # Disable warnings messages for log10(0.0)
    arr_abs = np.abs(arr)

    if arr_abs.shape == ():
        orig_shape = ()
        arr_abs = arr_abs.reshape((1,))
    else:
        orig_shape = None

    if reference is None:
        reference = np.nanmax(arr_abs)
    else:
        assert reference > 0.0

    with np.errstate(divide="ignore"):
        arr_db = 20 * np.log10(arr_abs / reference)

    if neginf_value is not None:
        arr_db[np.isneginf(arr_db)] = neginf_value

    if orig_shape is not None:
        arr_db = arr_db.reshape(orig_shape)

    if return_reference:
        return arr_db, reference
    else:
        return arr_db


def wrap_phase(phases):
    """Return a phase in [-pi, pi[

    http://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    """
    phases = np.asarray(phases)
    return (phases + np.pi) % (2 * np.pi) - np.pi


def instantaneous_phase_shift(analytic_sig, time_vect, carrier_frequency):
    """
    For a signal $x(ray) = A * exp(i (2 pi f_0 ray + phi(ray)))$, returns phi(ray) in [-pi, pi[.

    Parameters
    ----------
    analytic_sig: ndarray
    time_vect: ndarray
    carrier_frequency: float

    Returns
    -------
    phase_shift

    """
    analytic_sig = np.asarray(analytic_sig)
    dtype = analytic_sig.dtype
    if dtype.kind != "c":
        warnings.warn(
            "Expected an analytic (complex) signal, got {}. Use a Hilbert "
            "transform to get the analytic signal.".format(dtype),
            UtWarning,
            stacklevel=2,
        )
    phase_correction = 2 * np.pi * carrier_frequency * time_vect
    phase = wrap_phase(np.angle(analytic_sig) - phase_correction)
    return phase


def make_timevect(num, step, start=0.0, dtype=None):
    """
    Return a linearly spaced time vector.

    Remark: using this method is preferable to ``numpy.arange(start, start + num * step, step``
    which may yield an incorrect number of samples due to numerical inaccuracy.

    Parameters
    ----------
    num : int
        Number of samples to generate.
    step : float, optional
        Time step (time between consecutive samples).
    start : scalar
        Starting value of the sequence. Default: 0.
    dtype : numpy.dtype
        Optional, the type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    samples : ndarray
        Linearly spaced vector ``[start, stop]`` where ``end = start + (num - 1)*step``

    Examples
    --------
    >>> make_timevect(10, .1)
    array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
    >>> make_timevect(10, .1, start=1.)
    array([ 1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9])

    Notes
    -----

    Adapted from ``numpy.linspace``
    (License: http://www.numpy.org/license.html ; 3 clause BSD)

    """
    if not isinstance(num, int):
        raise TypeError("num must be an integer (got {})".format(type(num)))
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)

    # Convert float/complex array scalars to float
    start = start * 1.0
    step = step * 1.0

    dt = np.result_type(start, step)
    if dtype is None:
        dtype = dt

    y = np.arange(0, num, dtype=dt)

    if num > 1:
        y *= step

    y += start

    return y.astype(dtype, copy=False)


def reciprocal_viewname(viewname):
    """
    Return the name of the reciprocal view

    Parameters
    ----------
    viewname : str

    Returns
    -------
    reciprocal_viewname : str

    Examples
    --------
    >>> reciprocal_viewname('L-LT')
    'TL-L'

    """
    tx_path, rx_path = viewname.split("-")
    return rx_path[::-1] + "-" + tx_path[::-1]


IMAGING_MODES = [
    "L-L",
    "L-T",
    "T-T",
    "LL-L",
    "LL-T",
    "LT-L",
    "LT-T",
    "TL-L",
    "TL-T",
    "TT-L",
    "TT-T",
    "LL-LL",
    "LL-LT",
    "LL-TL",
    "LL-TT",
    "LT-LT",
    "LT-TL",
    "LT-TT",
    "TL-LT",
    "TL-TT",
    "TT-TT",
]
DIRECT_PATHS = ["L", "T"]
SKIP_PATHS = ["LL", "LT", "TL", "TT"]
DOUBLE_SKIP_PATHS = ["LLL", "LLT", "LTL", "LTT", "TLL", "TLT", "TTL", "TTT"]


def default_viewname_order(tx_rx_tuple):
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
    Returns the view names that that give different results in linear imaging.

    If views AB-CD and DC-BA are in 'viewnames' in this order, DC-BA will not
    be in the filtered list. Order is unchanged.

    Remove views that would give the same result because of time reciprocity
    (under linear assumption).

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


def make_viewnames(pathnames, tfm_unique_only=False, order_func=default_viewname_order):
    """
    Make all view names from the paths given as arguments.

    Parameters
    ----------
    pathnames : list[str]
    tfm_unique_only : bool
        Default: False. If True, returns only the views that give *different* imaging
        results with TFM (AB-CD and DC-BA give the same imaging result).
    order_func : func
        Function for sorting the views.

    Returns
    -------
    list[tuple[str]

    """
    viewnames = []
    for tx in pathnames:
        for rx in pathnames:
            viewnames.append((tx, rx))

    if order_func is not None:
        viewnames = list(sorted(viewnames, key=default_viewname_order))

    if tfm_unique_only:
        viewnames = filter_unique_views(viewnames)

    return viewnames


def rayleigh_vel(longitudinal_vel, transverse_vel):
    """
    Approximate Rayleigh velocitiy.

    Parameters
    ----------
    longitudinal_vel : float
    transverse_vel : float

    Returns
    -------
    rayleigh_vel : float

    Notes
    -----
    [Freund98] Freund, L. B.. 1998. `Dynamic Fracture Mechanics`.
    Cambridge University Press. p. 83. ISBN 978-0521629225.


    """
    poisson = (longitudinal_vel**2 - 2 * transverse_vel**2) / (
        2 * (longitudinal_vel**2 - transverse_vel**2)
    )
    if poisson <= 0:
        raise ValueError
    return transverse_vel * (0.862 + 1.14 * poisson) / (1 + poisson)
