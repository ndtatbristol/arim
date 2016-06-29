"""
Common functions used in imaging.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

from .. import settings as s
from ..exceptions import InvalidDimension
from ..utils import chunk_array

__all__ = ['delay_and_sum', 'find_minimum_times']


def delay_and_sum(frame, focal_law, fillvalue=np.nan, result=None, block_size=None, numthreads=None,interpolate_position=0):
    """
    Chunk the grid.

    :param frame: Frame
    :param focal_law: FocalLaw
    :param fillvalue: float
    :param result:
    :param block_size:
    :param numthreads:
    :return:
    """
    numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape

    assert focal_law.lookup_times_rx.shape == (numpoints, numelements)
    assert focal_law.amplitudes_tx.shape == (numpoints, numelements)
    assert focal_law.amplitudes_rx.shape == (numpoints, numelements)
    assert focal_law.scanline_weights.shape == (numscanlines,)
    assert frame.tx.shape == (numscanlines,)
    assert frame.rx.shape == (numscanlines,)

    assert focal_law.lookup_times_tx.flags.c_contiguous
    assert focal_law.lookup_times_rx.flags.c_contiguous
    assert focal_law.amplitudes_tx.flags.c_contiguous
    assert focal_law.amplitudes_rx.flags.c_contiguous
    assert frame.scanlines.flags.c_contiguous

    if result is None:
        result = np.full((numpoints,), 0, dtype=frame.scanlines.dtype)
    assert result.shape == (numpoints,)
    
    # Calculate using parallel CPU
    if block_size is None:
        block_size = s.BLOCK_SIZE_DELAY_AND_SUM
    if numthreads is None:
        numthreads = s.NUMTHREADS

    futures = []
    if interpolate_position == 0:
        #Nearest Match
        with ThreadPoolExecutor(max_workers=numthreads) as executor:
            for chunk in chunk_array((numpoints, ...), block_size):
                futures.append(executor.submit(
                    _delay_and_sum_amplitudes_nearest,
                    frame.scanlines, frame.tx, frame.rx,
                    focal_law.lookup_times_tx[chunk],
                    focal_law.lookup_times_rx[chunk],
                    focal_law.amplitudes_tx[chunk],
                    focal_law.amplitudes_rx[chunk],
                    focal_law.scanline_weights,
                    frame.time.step, frame.time.start, fillvalue,
                    result[chunk]))
    elif interpolate_position == 1:
        #Linear Interpolation
        with ThreadPoolExecutor(max_workers=numthreads) as executor:
            for chunk in chunk_array((numpoints, ...), block_size):
                futures.append(executor.submit(
                    _delay_and_sum_amplitudes_linear,
                    frame.scanlines, frame.tx, frame.rx,
                    focal_law.lookup_times_tx[chunk],
                    focal_law.lookup_times_rx[chunk],
                    focal_law.amplitudes_tx[chunk],
                    focal_law.amplitudes_rx[chunk],
                    focal_law.scanline_weights,
                    frame.time.step, frame.time.start, fillvalue,
                    result[chunk]))
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()
    
    return result


@numba.jit(nopython=True, nogil=True)
def _delay_and_sum_amplitudes_nearest(scanlines, tx, rx, lookup_times_tx, lookup_times_rx, amplitudes_tx, amplitudes_rx,
                              scanline_weights, dt, t0, fillvalue, result):
    """
    (CPU Parallel) Delay and Sum Algorithm, using nearest time point match     
    
    Parameters
    ----------
    scanlines : ndarray [numscanlines x numsamples]
    lookup_times_tx : ndarray [numpoints x numelements]
        Corresponds to the time of flight between the transmitters and the grid points.
        Values: integers in [0 and numsamples[
    lookup_times_rx : ndarray [numpoints x numelements]
        Corresponds to the time of flight between the grid points and the receivers.
        Values: integers in [0 and numsamples[
    amplitudes_tx : ndarray [numpoints x numelements]
    amplitudes_rx : ndarray [numpoints x numelements]
    scanline_weights : ndarray [numscanlines]
        Allow to scale specific scanlines. Useful for HMC, where we want a coefficient 2 when tx=rx, and 1 otherwise.
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numscanlines]
        Mapping between the scanlines and the transmitter/receiver.
        Values: integers in [0, numelements[

    Returns
    -------
    None
    """
    numscanlines, numsamples = scanlines.shape
    numpoints, numelements = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            lookup_index = round((lookup_time - t0) / dt)

            if lookup_index < 0 or lookup_index >= numsamples:
                result[point] += fillvalue
            else:
                result[point] += scanline_weights[scan] * amplitudes_tx[point, tx[scan]] * amplitudes_rx[point, rx[scan]] \
                                 * scanlines[scan, lookup_index]

@numba.jit(nopython=True, nogil=True)
def _delay_and_sum_amplitudes_linear(scanlines, tx, rx, lookup_times_tx, lookup_times_rx, amplitudes_tx, amplitudes_rx,
                              scanline_weights, dt, t0, fillvalue, result):
    """
    (CPU Parallel) Delay and Sum Algorithm, using linear interpolation for time point     
    
    Parameters
    ----------
    scanlines : ndarray [numscanlines x numsamples]
    lookup_times_tx : ndarray [numpoints x numelements]
        Corresponds to the time of flight between the transmitters and the grid points.
        Values: integers in [0 and numsamples[
    lookup_times_rx : ndarray [numpoints x numelements]
        Corresponds to the time of flight between the grid points and the receivers.
        Values: integers in [0 and numsamples[
    amplitudes_tx : ndarray [numpoints x numelements]
    amplitudes_rx : ndarray [numpoints x numelements]
    scanline_weights : ndarray [numscanlines]
        Allow to scale specific scanlines. Useful for HMC, where we want a coefficient 2 when tx=rx, and 1 otherwise.
    result : ndarray [numpoints]
        Result.
    tx, rx : ndarray [numscanlines]
        Mapping between the scanlines and the transmitter/receiver.
        Values: integers in [0, numelements[

    Returns 
    -------
    None
    """
    numscanlines, numsamples = scanlines.shape
    numpoints, numelements = lookup_times_tx.shape

    for point in range(numpoints):
        if np.isnan(result[point]):
            continue

        for scan in range(numscanlines):
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            loc1=(lookup_time - t0) / dt
            lookup_index = int(loc1)
            frac1 = loc1 - lookup_index 
            lookup_index1 = lookup_index+1
            #lookup_index = round((lookup_time - t0) / dt)

            if lookup_index < 0 or lookup_index1 >= numsamples:
                result[point] += fillvalue
            else:
                lscanVal=scanlines[scan, lookup_index]
                lscanVal1=scanlines[scan, lookup_index1]
                lscanUseVal=lscanVal+frac1*(lscanVal1-lscanVal)
                result[point] += scanline_weights[scan] * amplitudes_tx[point, tx[scan]] * amplitudes_rx[point, rx[scan]] \
                                 * lscanUseVal



def find_minimum_times(time_1, time_2, dtype=None, dtype_indices=None, block_size=None, numthreads=None):
    """
    Parameters
    ----------
    time_1
    time_2
    dtype
    dtype_indices

    Returns
    -------

    Notes
    -----
    Memory usage:
    - duplicate 'time_1' if it not in C-order.
    - duplicate 'time_2' if it not in Fortran-order.

    """
    assert time_1.ndim == 2
    assert time_2.ndim == 2
    try:
        n, m = time_1.shape
        m_, p = time_2.shape
    except ValueError:
        raise InvalidDimension("time_1 and time_2 must be 2d.")

    if m != m_:
        raise ValueError('Array shapes must be (n, m) and (m, p).')

    if dtype is None:
        dtype = np.result_type(time_1, time_2)
    if dtype_indices is None:
        dtype_indices = s.UINT

    if block_size is None:
        block_size = s.BLOCK_SIZE_FIND_MIN_TIMES
    if numthreads is None:
        numthreads = s.NUMTHREADS

    out_min_times = np.full((n, p), np.inf, dtype=dtype)
    out_best_indices = np.full((n, p), np.inf, dtype=dtype_indices)

    # time_1 will be scanned row per row, time_2 column per column.
    # Force to use the most efficient order (~20 times speed-up between the best and worst case).
    time_1 = np.ascontiguousarray(time_1)
    time_2 = np.asfortranarray(time_2)

    # Chunk time_1 and time_2 such as each chunk contains roughly 'block_size'
    # floats. Chunks for 'time_1' are lines (only complete lines), chunks
    # for 'time_2' are columns (only complete columns).
    block_size_adj = math.ceil(block_size / m)

    futures = []
    with ThreadPoolExecutor(max_workers=numthreads) as executor:
        for chunk1 in chunk_array((n, m), block_size_adj, axis=0):
            for chunk2 in chunk_array((m, p), block_size_adj, axis=1):
                chunk_res = (chunk1[0], chunk2[1])

                futures.append(executor.submit(
                    _find_minimum_times,
                    time_1[chunk1], time_2[chunk2],
                    out_min_times[chunk_res],
                    out_best_indices[chunk_res]))
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()

    return out_min_times, out_best_indices


@numba.jit(nopython=True, nogil=True)
def _find_minimum_times(time_1, time_2, out_min_times, out_best_indices):
    """
    Parameters
    ----------
    time_1
    time_2
    out_min_time
    out_best_indices

    Returns
    -------

    """
    n, m = time_1.shape
    m, p = time_2.shape
    for i in range(n):
        for j in range(p):
            best_time = np.inf
            best_index = m  # invalid index
            for k in range(m):
                new_time = time_1[i, k] + time_2[k, j]
                if new_time < best_time:
                    best_time = new_time
                    best_index = k
            out_min_times[i, j] = best_time
            out_best_indices[i, j] = best_index