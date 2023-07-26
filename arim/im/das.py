"""
Delay-and-sum functions

Contains different implementation of delay-and-sum algorithm.

.. currentmodule:: arim.im.das

Examples
--------

::

    res = das.delay_and_sum(frame, focal_law, fillvalue=0.0)
    res = das.delay_and_sum(frame, focal_law, fillvalue=np.nan)
    res = das.delay_and_sum(frame, focal_law, interpolation="nearest")
    res = das.delay_and_sum(frame, focal_law, interpolation="nearest", aggregation="median")
    res = das.delay_and_sum(frame, focal_law, interpolation="nearest", aggregation=("huber", 1.5))
    res = das.delay_and_sum(frame, focal_law, interpolation="linear")
    res = das.delay_and_sum(frame, focal_law, interpolation=("lanczos", 3))
    res = das.delay_and_sum(frame, focal_law, interpolation=("lanczos", 3), aggregation="median")


Data structures
---------------

- ``lookup_times_tx``: ndarray of shape (numgridpoints, numtx)
- ``lookup_times_rx``: ndarray of shape (numgridpoints, numrx)
- ``amplitudes_tx``: ndarray of shape (numgridpoints, numtx)
- ``amplitudes_tx``: ndarray of shape (numgridpoints, numrx)
- ``amplitudes``: TxRxAmplitudes or ndarray (numgridpoints, numtimetraces) or None

"""

import math
import logging

import numba
import numpy as np

from . import geomed, huber
from ..config import USE_PARALLEL

logger = logging.getLogger(__name__)


class NotImplementedTyping(NotImplementedError, TypeError):
    pass


def _check_shapes(frame, focal_law):
    numtimetraces = frame.numtimetraces
    numpoints, numtx = focal_law.lookup_times_tx.shape
    _, numrx = focal_law.lookup_times_rx.shape

    if focal_law.amplitudes is not None:
        assert focal_law.amplitudes.amplitudes_tx.shape == (numpoints, numtx)
        assert focal_law.amplitudes.amplitudes_rx.shape == (numpoints, numrx)
        assert focal_law.amplitudes.amplitudes_tx.flags.c_contiguous
        assert focal_law.amplitudes.amplitudes_rx.flags.c_contiguous

    assert frame.tx.shape == (numtimetraces,)
    assert frame.rx.shape == (numtimetraces,)

    assert focal_law.lookup_times_tx.flags.c_contiguous
    assert focal_law.lookup_times_rx.flags.c_contiguous
    assert frame.timetraces.flags.c_contiguous
    assert frame.tx.flags.c_contiguous
    assert frame.rx.flags.c_contiguous


def delay_and_sum_numba(
    frame,
    focal_law,
    fillvalue=0.0,
    interpolation="nearest",
    aggregation="mean",
    result=None,
):
    """
    Delay-and-sum function for non-uniform amplitudes

    Parameters
    ----------
    frame : Frame
    focal_law : FocalLaw
    fillvalue : float
        Default: 0.
    interpolation : str
        Interpolation of timetraces between samples. "linear", "nearest"
    aggregation : str
        Only "mean" supported.
    result : ndarray
        Write on it if provided.

    Returns
    -------
    result : ndarray (numpoints, )

    """
    # numtimetraces = frame.numtimetraces
    numpoints, _ = focal_law.lookup_times_tx.shape

    _check_shapes(frame, focal_law)

    weighted_timetraces = focal_law.weigh_timetraces(frame.timetraces)
    dtype_float, dtype_amp, dtype_data = _infer_datatypes(
        weighted_timetraces, focal_law, result
    )

    aggregation = aggregation.lower()
    if aggregation != "mean":
        raise NotImplementedError

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)

    if interpolation.lower() == "nearest":
        delay_and_sum_function = _delay_and_sum_amplitudes_nearest
    elif interpolation.lower() == "linear":
        delay_and_sum_function = _delay_and_sum_amplitudes_linear
    else:
        raise ValueError("invalid 'interpolation' argument")

    logger.debug("Delay-and-sum function: {}".format(delay_and_sum_function.__name__))

    delay_and_sum_function(
        weighted_timetraces,
        frame.tx,
        frame.rx,
        focal_law.lookup_times_tx,
        focal_law.lookup_times_rx,
        focal_law.amplitudes.amplitudes_tx,
        focal_law.amplitudes.amplitudes_rx,
        frame.time.step,
        frame.time.start,
        fillvalue,
        result,
    )
    return result


def _infer_datatypes(
    timetraces, focal_law, result, dtype_float=None, dtype_amp=None, dtype_data=None
):
    """
    Returns
    -------
    dtype_float: datatype of times only values (time)
    dtype_amp: datatype of TFM amplitudes
    dtype_data: datatype that suits both the input timetraces and the results

    """
    if dtype_float is None:
        dtype_float = np.result_type(
            focal_law.lookup_times_tx, focal_law.lookup_times_rx
        )
    if dtype_amp is None:
        if focal_law.amplitudes is None:
            has_amp = False
            dtype_amp = None
        else:
            has_amp = True
            dtype_amp = focal_law.amplitudes.dtype
    if dtype_data is None:
        if has_amp:
            data_arrays = [timetraces, dtype_amp]
        else:
            data_arrays = [timetraces]
        if result is not None:
            data_arrays.append(result)
        dtype_data = np.result_type(*data_arrays)
    return dtype_float, dtype_amp, dtype_data


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_amplitudes_nearest(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    amplitudes_tx,
    amplitudes_rx,
    dt,
    t0,
    fillvalue,
    result,
):
    """
    Numba implementation of the delay and sum algorithm, using nearest time point
    match.

    Parameters
    ----------
    weighted_timetraces : ndarray [numtimetraces x numsamples]
    lookup_times_tx : ndarray [numpoints x numtx]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numrx]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes_tx : ndarray [numpoints x numtx]
    amplitudes_rx : ndarray [numpoints x numrx]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numtimetraces]
        Mapping between the timetraces and the transmitter/receiver.
        Values: integers in [0, numtx[ and [0, numrx] respectively

    Returns
    -------
    None
    """
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        res_tmp = 0.0

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            lookup_index = round((lookup_time - t0) / dt)

            if lookup_index < 0 or lookup_index >= numsamples:
                res_tmp += fillvalue
            else:
                res_tmp += (
                    amplitudes_tx[point, tx[scan]]
                    * amplitudes_rx[point, rx[scan]]
                    * weighted_timetraces[scan, lookup_index]
                )
        result[point] = res_tmp / numtimetraces


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_amplitudes_linear(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    amplitudes_tx,
    amplitudes_rx,
    dt,
    t0,
    fillvalue,
    result,
):
    """
    Numba implementation of the delay and sum algorithm, using linear
    interpolation for time point.

    Parameters
    ----------
    weighted_timetraces : ndarray [numtimetraces x numsamples]
    lookup_times_tx : ndarray [numpoints x numtx]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numrx]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes_tx : ndarray [numpoints x numtx]
    amplitudes_rx : ndarray [numpoints x numrx]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numtimetraces]
        Mapping between the timetraces and the transmitter/receiver.
        Values: integers in [0, numtx[ and [0, numrx] respectively

    Returns
    -------
    None
    """
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        res_tmp = 0.0
        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            loc1 = (lookup_time - t0) / dt
            lookup_index = int(loc1)
            frac1 = loc1 - lookup_index
            lookup_index1 = lookup_index + 1

            if lookup_index < 0 or lookup_index1 >= numsamples:
                res_tmp += fillvalue
            else:
                lscanVal = weighted_timetraces[scan, lookup_index]
                lscanVal1 = weighted_timetraces[scan, lookup_index1]
                lscanUseVal = lscanVal + frac1 * (lscanVal1 - lscanVal)
                res_tmp += (
                    amplitudes_tx[point, tx[scan]]
                    * amplitudes_rx[point, rx[scan]]
                    * lscanUseVal
                )
        result[point] = res_tmp / numtimetraces


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_amplitudes_linear(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    amplitudes_tx,
    amplitudes_rx,
    dt,
    t0,
    fillvalue,
    result,
):
    """
    Numba implementation of the delay and sum algorithm, using linear
    interpolation for time point.

    Parameters
    ----------
    weighted_timetraces : ndarray [numtimetraces x numsamples]
    lookup_times_tx : ndarray [numpoints x numtx]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numrx]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes_tx : ndarray [numpoints x numtx]
    amplitudes_rx : ndarray [numpoints x numrx]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numtimetraces]
        Mapping between the timetraces and the transmitter/receiver.
        Values: integers in [0, numtx[ and [0, numrx] respectively

    Returns
    -------
    None
    """
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        res_tmp = 0.0
        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            loc1 = (lookup_time - t0) / dt
            lookup_index = int(loc1)
            frac1 = loc1 - lookup_index
            lookup_index1 = lookup_index + 1

            if lookup_index < 0 or lookup_index1 >= numsamples:
                res_tmp += fillvalue
            else:
                lscanVal = weighted_timetraces[scan, lookup_index]
                lscanVal1 = weighted_timetraces[scan, lookup_index1]
                lscanUseVal = lscanVal + frac1 * (lscanVal1 - lscanVal)
                res_tmp += (
                    amplitudes_tx[point, tx[scan]]
                    * amplitudes_rx[point, rx[scan]]
                    * lscanUseVal
                )
        result[point] = res_tmp / numtimetraces


def delay_and_sum_numba_noamp(
    frame,
    focal_law,
    fillvalue=0.0,
    interpolation="nearest",
    aggregation="mean",
    result=None,
):
    """
    Delay and sum with uniform amplitudes

    Parameters
    ----------
    frame
    focal_law
    fillvalue
    interpolation : str
        Interpolation of timetraces between samples. 'linear', 'nearest', ('lanczos', a)
    result

    Returns
    -------
    result : ndarray (numpoints, )

    """
    numpoints, _ = focal_law.lookup_times_tx.shape

    _check_shapes(frame, focal_law)

    weighted_timetraces = focal_law.weigh_timetraces(frame.timetraces)
    dtype_float, dtype_amp, dtype_data = _infer_datatypes(
        weighted_timetraces, focal_law, result
    )

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)

    invdt = 1 / frame.time.step
    t0 = frame.time.start

    if isinstance(interpolation, str):
        interpolation_name = interpolation.lower()
        interpolation_args = ()
    else:
        interpolation_name = interpolation[0].lower()
        interpolation_args = interpolation[1:]

    if isinstance(aggregation, str):
        aggregation_name = aggregation.lower()
        aggregation_args = ()
    else:
        aggregation_name = aggregation[0].lower()
        aggregation_args = aggregation[1:]

    if aggregation_name == "mean":
        if interpolation_name == "nearest":
            das_func = _delay_and_sum_noamp
            assert len(interpolation_args) == 0
        elif interpolation_name == "linear":
            das_func = _delay_and_sum_noamp_linear
            assert len(interpolation_args) == 0
        elif interpolation_name == "lanczos":
            das_func = _delay_and_sum_noamp_lanczos
            assert len(interpolation_args) == 1
        else:
            raise ValueError("invalid interpolation")
    elif aggregation_name == "median":
        if dtype_data != np.complex_:
            raise NotImplementedTyping
        if interpolation_name == "lanczos":
            das_func = _delay_and_sum_noamp_median_lanczos
            assert len(interpolation_args) == 1
        elif interpolation_name == "nearest":
            das_func = _delay_and_sum_noamp_median_nearest
            assert len(interpolation_args) == 0
        else:
            raise NotImplementedError
    elif aggregation_name == "huber":
        if dtype_data != np.complex_:
            raise NotImplementedTyping
        if interpolation_name == "lanczos":
            das_func = _delay_and_sum_noamp_huber_lanczos
            assert len(interpolation_args) == 1
        else:
            raise NotImplementedError

    das_func(
        weighted_timetraces,
        frame.tx,
        frame.rx,
        focal_law.lookup_times_tx,
        focal_law.lookup_times_rx,
        invdt,
        t0,
        fillvalue,
        *interpolation_args,
        *aggregation_args,
        result
    )
    return result


# todo: add cache=True if it becomes compatible with parallel=True (numba)
@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_noamp(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    invdt,
    t0,
    fillvalue,
    result,
):
    # Mean aggregation, nearest interpolation
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        res_tmp = 0.0

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            lookup_index = round((lookup_time - t0) * invdt)

            if lookup_index < 0 or lookup_index >= numsamples:
                res_tmp += fillvalue
            else:
                res_tmp += weighted_timetraces[scan, lookup_index]
        result[point] = res_tmp / numtimetraces


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_noamp_median_nearest(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    invdt,
    t0,
    fillvalue,
    result,
):
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        datapoints = np.empty(numtimetraces, weighted_timetraces.dtype)

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            lookup_index = round((lookup_time - t0) * invdt)

            if lookup_index < 0 or lookup_index >= numsamples:
                datapoints[scan] = fillvalue
            else:
                datapoints[scan] = weighted_timetraces[scan, lookup_index]
        # I don't know how to statically cast complex64 to float32, and
        # complex128 to float64 :(
        # Have to impose dtype meanwhile.
        res, _ = geomed.geomed(datapoints.view(np.float_).reshape((numtimetraces, 2)))
        result[point] = res.view(np.complex_)[0]


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_noamp_linear(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    invdt,
    t0,
    fillvalue,
    result,
):
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        res_tmp = 0.0

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )

            lookup_index_exact = (lookup_time - t0) * invdt
            lookup_index_left = int(lookup_index_exact)
            lookup_index_right = lookup_index_left + 1
            frac = lookup_index_exact - lookup_index_left

            if lookup_index_left < 0 or lookup_index_right >= numsamples:
                res_tmp += fillvalue
            else:
                scan_val_left = weighted_timetraces[scan, lookup_index_left]
                scan_val_right = weighted_timetraces[scan, lookup_index_right]
                res_tmp += (1 - frac) * scan_val_left + frac * scan_val_right

        result[point] = res_tmp / numtimetraces


@numba.jit(nopython=True, cache=True, fastmath=True)
def sinc(x):
    if x == 0:
        return 1.0
    else:
        return math.sin(math.pi * x) / (math.pi * x)


@numba.jit(nopython=True, cache=True, fastmath=True)
def lanczos_interpolation(t, x, a):
    i_min = math.floor(t) - a + 1
    i_max = math.floor(t) + a + 1  # +1 because of how range() works
    n = len(x)
    out = 0.0
    for i in range(i_min, i_max):
        out += x[i % n] * sinc(t - i) * sinc((t - i) / a)
    return out


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_noamp_lanczos(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    invdt,
    t0,
    fillvalue,
    a,
    result,
):
    """
    Delay and sum with Lanczos interpolation with factor 'a' of the timetraces.

    https://en.wikipedia.org/wiki/Lanczos_resampling
    """
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        res_tmp = 0.0

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )

            lookup_index = (lookup_time - t0) * invdt

            if lookup_index < 0 or lookup_index >= numsamples:
                res_tmp += fillvalue
            else:
                res_tmp += lanczos_interpolation(
                    lookup_index, weighted_timetraces[scan], a
                )

        result[point] = res_tmp / numtimetraces


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_noamp_median_lanczos(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    invdt,
    t0,
    fillvalue,
    a,
    result,
):
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        datapoints = np.empty(numtimetraces, weighted_timetraces.dtype)

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )

            lookup_index = (lookup_time - t0) * invdt

            if lookup_index < 0 or lookup_index >= numsamples:
                datapoints[scan] = fillvalue
            else:
                datapoints[scan] = lanczos_interpolation(
                    lookup_index, weighted_timetraces[scan], a
                )
        # I don't know how to statically cast complex64 to float32, and
        # complex128 to float64 :(
        # Have to impose dtype meanwhile.
        res, _ = geomed.geomed(datapoints.view(np.float_).reshape((numtimetraces, 2)))
        result[point] = res.view(np.complex_)[0]


@numba.jit(nopython=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def _delay_and_sum_noamp_huber_lanczos(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    invdt,
    t0,
    fillvalue,
    a,
    tau,
    result,
):
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        datapoints = np.empty(numtimetraces, weighted_timetraces.dtype)

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )

            lookup_index = (lookup_time - t0) * invdt

            if lookup_index < 0 or lookup_index >= numsamples:
                datapoints[scan] = fillvalue
            else:
                datapoints[scan] = lanczos_interpolation(
                    lookup_index, weighted_timetraces[scan], a
                )
        # I don't know how to statically cast complex64 to float32, and
        # complex128 to float64 :(
        # Have to impose dtype meanwhile.
        res, _ = huber.huber_m_estimate(
            datapoints.view(np.float_).reshape((numtimetraces, 2)), tau
        )
        result[point] = res.view(np.complex_)[0]


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _general_delay_and_sum_nearest(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    amplitudes,
    dt,
    t0,
    fillvalue,
    result,
):
    """
    Numba implementation of the delay and sum algorithm using nearest
    interpolation for time point.

    Amplitudes are defined per timetrace instead of per element. This function is
    therefore more general but more memory-hungry.

    One amplitude per timetrace.

    Parameters
    ----------
    weighted_timetraces : ndarray [numtimetraces x numsamples]
    lookup_times_tx : ndarray [numpoints x numtx]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numrx]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes : ndarray [numpoints x numtimetraces]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numtimetraces]
        Mapping between the timetraces and the transmitter/receiver.
        Values: integers in [0, numtx[ and [0, numrx]

    Returns
    -------
    None
    """
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            lookup_index = int((lookup_time - t0) / dt + 0.5)

            if lookup_index < 0 or lookup_index >= numsamples:
                result[point] += fillvalue
            else:
                result[point] += (
                    amplitudes[point, scan] * weighted_timetraces[scan, lookup_index]
                )
        result[point] /= numtimetraces


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _general_delay_and_sum_linear(
    weighted_timetraces,
    tx,
    rx,
    lookup_times_tx,
    lookup_times_rx,
    amplitudes,
    dt,
    t0,
    fillvalue,
    result,
):
    """
    Numba implementation of the delay and sum algorithm, using linear
    interpolation for time point.

    Amplitudes are defined per timetrace instead of per element. This function is
    therefore more general but more memory-hungry.

    One amplitude per timetrace.

    Parameters
    ----------
    weighted_timetraces : ndarray [numtimetraces x numsamples]
    lookup_times_tx : ndarray [numpoints x numtx]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numrx]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes : ndarray [numpoints x numtimetraces]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numtimetraces]
        Mapping between the timetraces and the transmitter/receiver.
        Values: integers in [0, numtx[ and [0, numrx]

    Returns
    -------
    None
    """
    numtimetraces, numsamples = weighted_timetraces.shape
    numpoints, _ = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            loc1 = (lookup_time - t0) / dt
            lookup_index = int(loc1)
            frac1 = loc1 - lookup_index
            lookup_index1 = lookup_index + 1

            if lookup_index < 0 or lookup_index1 >= numsamples:
                result[point] += fillvalue
            else:
                lscanVal = weighted_timetraces[scan, lookup_index]
                lscanVal1 = weighted_timetraces[scan, lookup_index1]
                lscanUseVal = lscanVal + frac1 * (lscanVal1 - lscanVal)
                result[point] += amplitudes[point, scan] * lscanUseVal
        result[point] /= numtimetraces


def delay_and_sum_naive(
    frame, focal_law, fillvalue=0.0, result=None, interpolate_position="nearest"
):
    """
    Pure-Python implementation of delay and sum.

    This is a very slow implementation, use for test only.
    """
    numtimetraces = frame.numtimetraces
    numpoints, _ = focal_law.lookup_times_tx.shape

    from . import tfm

    assert isinstance(focal_law.amplitudes, tfm.TxRxAmplitudes)

    _check_shapes(frame, focal_law)
    weighted_timetraces = focal_law.weigh_timetraces(frame.timetraces)
    _, _, dtype_data = _infer_datatypes(weighted_timetraces, focal_law, result)

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)
    if interpolate_position != "nearest":
        raise NotImplementedError

    lookup_times_tx = focal_law.lookup_times_tx
    lookup_times_rx = focal_law.lookup_times_rx
    amplitudes_tx = focal_law.amplitudes.amplitudes_tx
    amplitudes_rx = focal_law.amplitudes.amplitudes_rx
    timetrace_weights = focal_law.timetrace_weights
    if timetrace_weights is None:
        timetrace_weights = np.ones(numtimetraces)
    timetraces = frame.timetraces
    tx = frame.tx
    rx = frame.rx
    t0 = frame.time.start
    dt = frame.time.step
    numsamples = len(frame.time)

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numtimetraces):
            lookup_time = (
                lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            )
            lookup_index = int(round((lookup_time - t0) / dt))

            if lookup_index < 0 or lookup_index >= numsamples:
                result[point] += fillvalue
            else:
                result[point] += (
                    timetrace_weights[scan]
                    * amplitudes_tx[point, tx[scan]]
                    * amplitudes_rx[point, rx[scan]]
                    * timetraces[scan, lookup_index]
                )
        result[point] /= numtimetraces
    return result


def delay_and_sum(frame, focal_law, *args, **kwargs):
    """
    Delay-and-sum timetraces

    .. warning:

        Not all combinations of parameters are implemented.

    Parameters
    ----------
    frame
    focal_law
    fillvalue : float
        Value to use outside of time limits of the timetrace.
    interpolation : str
        "nearest", "linear", "lanczos"
    aggregation : str
        "mean", "median"

    Returns
    -------
    result : ndarray (numpoints, )


    See Also
    --------
    :func:`delay_and_sum_numba`
    :func:`delay_and_sum_numba_noamp`

    """
    from . import tfm

    if isinstance(focal_law.amplitudes, tfm.TxRxAmplitudes):
        return delay_and_sum_numba(frame, focal_law, *args, **kwargs)
    elif focal_law.amplitudes is None:
        return delay_and_sum_numba_noamp(frame, focal_law, *args, **kwargs)
    else:
        raise NotImplementedError
