import numba
import numpy as np
from collections import OrderedDict
import logging

from ... import settings as s
from . import _delay_and_sum_cpu
from concurrent.futures import ThreadPoolExecutor
from ...helpers import chunk_array

_DATATYPES = OrderedDict()
_DATATYPES['f'] = (np.float32, np.float32)
_DATATYPES['c'] = (np.float32, np.complex64)
_DATATYPES['d'] = (np.float64, np.float64)
_DATATYPES['z'] = (np.float64, np.complex128)

logger = logging.getLogger(__name__)


def _check_shapes(frame, focal_law):
    numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape

    assert focal_law.lookup_times_rx.shape == (numpoints, numelements)
    if focal_law.amplitudes is not None:
        assert focal_law.amplitudes.amplitudes_tx.shape == (numpoints, numelements)
        assert focal_law.amplitudes.amplitudes_rx.shape == (numpoints, numelements)
        assert focal_law.amplitudes.amplitudes_tx.flags.c_contiguous
        assert focal_law.amplitudes.amplitudes_rx.flags.c_contiguous

    assert frame.tx.shape == (numscanlines,)
    assert frame.rx.shape == (numscanlines,)

    assert focal_law.lookup_times_tx.flags.c_contiguous
    assert focal_law.lookup_times_rx.flags.c_contiguous
    assert frame.scanlines.flags.c_contiguous
    assert frame.tx.flags.c_contiguous
    assert frame.rx.flags.c_contiguous


def delay_and_sum_numba(frame, focal_law, fillvalue=0., result=None, block_size=None,
                        numthreads=None, interpolate_position='nearest'):
    """
    Delay-and-sum function using Numba compiler.

    Works on CPU, multi-threading using the standard ``concurrent.futures`` library
    (not as fast as openmp).

    Parameters
    ----------
    frame : Frame
    focal_law : FocalLaw
    fillvalue : float
        Default: 0.
    result : ndarray
        Write on it if provided.
    block_size : int
        Block size for multithreading. Use arim.settings if not provided
    numthreads : int
        Number of threads for multithreading. Use arim.settings if not provided
    interpolate_position : string
        'nearest' or 'linear'. Default: nearest

    Returns
    -------

    """
    # numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape

    _check_shapes(frame, focal_law)

    weighted_scanlines = focal_law.weigh_scanlines(frame.scanlines)
    dtype_float, dtype_amp, dtype_data = _infer_datatypes(weighted_scanlines, focal_law, result)

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)

    # Parameters for multithreading:
    if block_size is None:
        block_size = s.BLOCK_SIZE_DELAY_AND_SUM
    if numthreads is None:
        numthreads = s.NUMTHREADS

    futures = []
    if interpolate_position.lower() == 'nearest':
        delay_and_sum_function = _delay_and_sum_amplitudes_nearest
    elif interpolate_position.lower() == 'linear':
        delay_and_sum_function = _delay_and_sum_amplitudes_linear
    else:
        raise ValueError("invalid 'interpolate_position'")

    logger.debug("Delay-and-sum function: {}".format(delay_and_sum_function.__name__))

    with ThreadPoolExecutor(max_workers=numthreads) as executor:
        for chunk in chunk_array((numpoints, ...), block_size):
            futures.append(executor.submit(
                delay_and_sum_function,
                weighted_scanlines, frame.tx, frame.rx,
                focal_law.lookup_times_tx[chunk],
                focal_law.lookup_times_rx[chunk],
                focal_law.amplitudes.amplitudes_tx[chunk],
                focal_law.amplitudes.amplitudes_rx[chunk],
                frame.time.step, frame.time.start, fillvalue,
                result[chunk]))
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()

    return result


def _infer_datatypes(scanlines, focal_law, result, dtype_float=None, dtype_amp=None,
                     dtype_data=None):
    """
    Returns
    -------
    dtype_float: datatype of times only values (time)
    dtype_amp: datatype of TFM amplitudes
    dtype_data: datatype that suits both the input scanlines and the results

    """
    if dtype_float is None:
        dtype_float = np.result_type(focal_law.lookup_times_tx, focal_law.lookup_times_rx)
    if dtype_amp is None:
        if focal_law.amplitudes is None:
            has_amp = False
            dtype_amp = None
        else:
            has_amp = True
            dtype_amp = focal_law.amplitudes.dtype
    if dtype_data is None:
        if has_amp:
            data_arrays = [scanlines, dtype_amp]
        else:
            data_arrays = [scanlines]
        if result is not None:
            data_arrays.append(result)
        dtype_data = np.result_type(*data_arrays)
    return dtype_float, dtype_amp, dtype_data


@numba.jit(nopython=True, nogil=True, cache=True)
def _delay_and_sum_amplitudes_nearest(weighted_scanlines, tx, rx, lookup_times_tx,
                                      lookup_times_rx, amplitudes_tx,
                                      amplitudes_rx, dt, t0, fillvalue, result):
    """
    Numba implementation of the delay and sum algorithm, using nearest time point
    match.

    Parameters
    ----------
    weighted_scanlines : ndarray [numscanlines x numsamples]
    lookup_times_tx : ndarray [numpoints x numelements]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numelements]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes_tx : ndarray [numpoints x numelements]
    amplitudes_rx : ndarray [numpoints x numelements]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numscanlines]
        Mapping between the scanlines and the transmitter/receiver.
        Values: integers in [0, numelements[

    Returns
    -------
    None
    """
    numscanlines, numsamples = weighted_scanlines.shape
    numpoints, numelements = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[
                point, rx[scan]]
            # lookup_index = round((lookup_time - t0) / dt)
            lookup_index = int((lookup_time - t0) / dt + 0.5)

            if lookup_index < 0 or lookup_index >= numsamples:
                result[point] += fillvalue
            else:
                result[point] += amplitudes_tx[point, tx[scan]] * amplitudes_rx[
                    point, rx[scan]] \
                                 * weighted_scanlines[scan, lookup_index]


@numba.jit(nopython=True, nogil=True, cache=True)
def _delay_and_sum_amplitudes_linear(weighted_scanlines, tx, rx, lookup_times_tx,
                                     lookup_times_rx, amplitudes_tx,
                                     amplitudes_rx, dt, t0, fillvalue, result):
    """
    Numba implementation of the delay and sum algorithm, using linear
    interpolation for time point.

    Parameters
    ----------
    weighted_scanlines : ndarray [numscanlines x numsamples]
    lookup_times_tx : ndarray [numpoints x numelements]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numelements]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes_tx : ndarray [numpoints x numelements]
    amplitudes_rx : ndarray [numpoints x numelements]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numscanlines]
        Mapping between the scanlines and the transmitter/receiver.
        Values: integers in [0, numelements[

    Returns
    -------
    None
    """
    numscanlines, numsamples = weighted_scanlines.shape
    numpoints, numelements = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[
                point, rx[scan]]
            loc1 = (lookup_time - t0) / dt
            lookup_index = int(loc1)
            frac1 = loc1 - lookup_index
            lookup_index1 = lookup_index + 1

            if lookup_index < 0 or lookup_index1 >= numsamples:
                result[point] += fillvalue
            else:
                lscanVal = weighted_scanlines[scan, lookup_index]
                lscanVal1 = weighted_scanlines[scan, lookup_index1]
                lscanUseVal = lscanVal + frac1 * (lscanVal1 - lscanVal)
                result[point] += amplitudes_tx[point, tx[scan]] \
                                 * amplitudes_rx[point, rx[scan]] * lscanUseVal


def delay_and_sum_numba_noamp(frame, focal_law, fillvalue=0., result=None):
    """
    Delay and sum with uniform amplitudes

    Parameters
    ----------
    frame
    focal_law
    fillvalue
    result

    Returns
    -------
    result

    """
    numpoints, numelements = focal_law.lookup_times_tx.shape

    _check_shapes(frame, focal_law)

    weighted_scanlines = focal_law.weigh_scanlines(frame.scanlines)
    dtype_float, dtype_amp, dtype_data = _infer_datatypes(weighted_scanlines, focal_law, result)

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)

    invdt = 1 / frame.time.step
    t0 = frame.time.start

    _delay_and_sum_noamp(weighted_scanlines, frame.tx, frame.rx,
                         focal_law.lookup_times_tx, focal_law.lookup_times_rx,
                         invdt, t0, fillvalue, result)
    return result


# todo: add cache=True if it becomes compatible with parallel=True (numba)
@numba.jit(nopython=True, nogil=True, parallel=True)
def _delay_and_sum_noamp(weighted_scanlines, tx, rx, lookup_times_tx,
                         lookup_times_rx, invdt, t0, fillvalue, result):
    numscanlines, numsamples = weighted_scanlines.shape
    numpoints, numelements = lookup_times_tx.shape

    for point in numba.prange(numpoints):
        res_tmp = 0.

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[
                point, rx[scan]]
            lookup_index = round((lookup_time - t0) * invdt)

            if lookup_index < 0 or lookup_index >= numsamples:
                res_tmp += fillvalue
            else:
                res_tmp += weighted_scanlines[scan, lookup_index]
        result[point] = res_tmp


@numba.jit(nopython=True, nogil=True, cache=True)
def _general_delay_and_sum_nearest(weighted_scanlines, tx, rx, lookup_times_tx,
                                   lookup_times_rx, amplitudes,
                                   dt, t0, fillvalue, result):
    """
    Numba implementation of the delay and sum algorithm using nearest
    interpolation for time point.

    Amplitudes are defined per scanline instead of per element. This function is
    therefore more general but more memory-hungry.

    One amplitude per scanline.

    Parameters
    ----------
    weighted_scanlines : ndarray [numscanlines x numsamples]
    lookup_times_tx : ndarray [numpoints x numelements]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numelements]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes : ndarray [numpoints x numscanlines]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numscanlines]
        Mapping between the scanlines and the transmitter/receiver.
        Values: integers in [0, numelements[

    Returns
    -------
    None
    """
    numscanlines, numsamples = weighted_scanlines.shape
    numpoints, numelements = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[
                point, rx[scan]]
            lookup_index = int((lookup_time - t0) / dt + 0.5)

            if lookup_index < 0 or lookup_index >= numsamples:
                result[point] += fillvalue
            else:
                result[point] += (amplitudes[point, scan] *
                                  weighted_scanlines[scan, lookup_index])


@numba.jit(nopython=True, nogil=True, cache=True)
def _general_delay_and_sum_linear(weighted_scanlines, tx, rx, lookup_times_tx,
                                  lookup_times_rx, amplitudes,
                                  dt, t0, fillvalue, result):
    """
    Numba implementation of the delay and sum algorithm, using linear
    interpolation for time point.

    Amplitudes are defined per scanline instead of per element. This function is
    therefore more general but more memory-hungry.

    One amplitude per scanline.

    Parameters
    ----------
    weighted_scanlines : ndarray [numscanlines x numsamples]
    lookup_times_tx : ndarray [numpoints x numelements]
        Times of flight (floats) between the transmitters and the grid points.
    lookup_times_rx : ndarray [numpoints x numelements]
        Times of flight (floats) between the grid points and the receivers.
    amplitudes : ndarray [numpoints x numscanlines]
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numscanlines]
        Mapping between the scanlines and the transmitter/receiver.
        Values: integers in [0, numelements[

    Returns
    -------
    None
    """
    numscanlines, numsamples = weighted_scanlines.shape
    numpoints, numelements = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[
                point, rx[scan]]
            loc1 = (lookup_time - t0) / dt
            lookup_index = int(loc1)
            frac1 = loc1 - lookup_index
            lookup_index1 = lookup_index + 1

            if lookup_index < 0 or lookup_index1 >= numsamples:
                result[point] += fillvalue
            else:
                lscanVal = weighted_scanlines[scan, lookup_index]
                lscanVal1 = weighted_scanlines[scan, lookup_index1]
                lscanUseVal = lscanVal + frac1 * (lscanVal1 - lscanVal)
                result[point] += amplitudes[point, scan] * lscanUseVal


def delay_and_sum_naive(frame, focal_law, fillvalue=0., result=None,
                        interpolate_position='nearest'):
    """
    Pure-Python implementation of delay and sum.

    This is a very slow implementation, use for test only.
    """
    numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape

    from .. import tfm
    assert isinstance(focal_law.amplitudes, tfm.TxRxAmplitudes)

    _check_shapes(frame, focal_law)
    weighted_scanlines = focal_law.weigh_scanlines(frame.scanlines)
    _, _, dtype_data = _infer_datatypes(weighted_scanlines, focal_law, result)

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)
    if interpolate_position != 'nearest':
        raise NotImplementedError

    lookup_times_tx = focal_law.lookup_times_tx
    lookup_times_rx = focal_law.lookup_times_rx
    amplitudes_tx = focal_law.amplitudes.amplitudes_tx
    amplitudes_rx = focal_law.amplitudes.amplitudes_rx
    scanline_weights = focal_law.scanline_weights
    if scanline_weights is None:
        scanline_weights = np.ones(numscanlines)
    scanlines = frame.scanlines
    tx = frame.tx
    rx = frame.rx
    t0 = frame.time.start
    dt = frame.time.step
    numsamples = len(frame.time)

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[
                point, rx[scan]]
            lookup_index = int(round((lookup_time - t0) / dt))

            if lookup_index < 0 or lookup_index >= numsamples:
                result[point] += fillvalue
            else:
                result[point] += scanline_weights[scan] * amplitudes_tx[point, tx[scan]] \
                                 * amplitudes_rx[point, rx[scan]] * scanlines[
                                     scan, lookup_index]
    return result


def delay_and_sum_cpu(frame, focal_law, fillvalue=0., result=None, interpolate_position='nearest'):
    """
    Delay-and-sum function using internally a C++ version.

    Multi-threading with OpenMP.

    Parameters
    ----------
    frame
    focal_law
    fillvalue
    result
    block_size
    numthreads
    interpolate_position

    Returns
    -------

    """
    numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape
    numsamples = len(frame.time)

    _check_shapes(frame, focal_law)
    weighted_scanlines = focal_law.weigh_scanlines(frame.scanlines)
    dtype_float, dtype_amp, dtype_data = _infer_datatypes(weighted_scanlines, focal_law, result)

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)
    if interpolate_position != 'nearest':
        raise NotImplementedError

    # Function of module '_delay_and_sum_cpu' have their signature
    # as names.
    # Example: _delay_and_sum_nearest_float64_complex128_complex128
    sig_types = "{}_{}_{}".format(dtype_float.name, dtype_amp.name, dtype_data.name)
    try:
        func = getattr(_delay_and_sum_cpu, '_delay_and_sum_nearest_{}'.format(sig_types))
    except AttributeError:
        raise ValueError("no matching signature")

    logger.debug("Delay-and-sum function: {}".format(func.__name__))

    invdt = dtype_float.type(1 / frame.time.step)
    tstart = dtype_float.type(frame.time.start)
    fillvalue = dtype_data.type(fillvalue)

    tx = np.array(frame.tx, dtype=np.uint, copy=False)
    rx = np.array(frame.rx, dtype=np.uint, copy=False)

    lookup_times_tx = np.array(focal_law.lookup_times_tx, dtype=dtype_float, copy=False)
    lookup_times_rx = np.array(focal_law.lookup_times_rx, dtype=dtype_float, copy=False)
    weighted_scanlines = np.array(weighted_scanlines, dtype=dtype_data, copy=False)
    amplitudes_tx = np.array(focal_law.amplitudes.amplitudes_tx, dtype=dtype_amp, copy=False)
    amplitudes_rx = np.array(focal_law.amplitudes.amplitudes_rx, dtype=dtype_amp, copy=False)

    func(weighted_scanlines, tx, rx,
         lookup_times_tx,
         lookup_times_rx,
         amplitudes_tx,
         amplitudes_rx,
         invdt, tstart, fillvalue,
         result, numpoints, numsamples, numelements,
         numscanlines)
    return result


def delay_and_sum(frame, focal_law, *args, **kwargs):
    """
    Recommended delay-and-sum function

    Alias of :func:`delay_and_sum_cpu`
    """
    from .. import tfm
    if isinstance(focal_law.amplitudes, tfm.TxRxAmplitudes):
        return delay_and_sum_cpu(frame, focal_law, *args, **kwargs)
    elif focal_law.amplitudes is None:
        return delay_and_sum_numba_noamp(frame, focal_law, *args, **kwargs)
    else:
        raise NotImplementedError
