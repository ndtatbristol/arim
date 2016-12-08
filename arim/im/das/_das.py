import numpy as np
from collections import OrderedDict
import logging

from ... import settings as s

_DATATYPES = OrderedDict()
_DATATYPES['f'] = (np.float32, np.float32)
_DATATYPES['c'] = (np.float32, np.complex64)
_DATATYPES['d'] = (np.float64, np.float64)
_DATATYPES['z'] = (np.float64, np.complex128)

logger = logging.getLogger(__name__)

__all__ = ['delay_and_sum_numba', 'delay_and_sum_cpu', 'delay_and_sum',
           'delay_and_sum_naive']


def _check_shapes(frame, focal_law):
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
    assert frame.tx.flags.c_contiguous
    assert frame.rx.flags.c_contiguous


def _infer_datatypes(frame, focal_law, result, dtype_float=None, dtype_amp=None,
                     dtype_data=None):
    if dtype_float is None:
        dtype_float = np.result_type(focal_law.lookup_times_tx, focal_law.lookup_times_rx)
    if dtype_amp is None:
        dtype_amp = np.result_type(focal_law.amplitudes_tx, focal_law.amplitudes_rx)
    if dtype_data is None:
        data_arrays = [focal_law.scanline_weights, frame.scanlines]
        if result is not None:
            data_arrays.append(result)
        dtype_data = np.result_type(*data_arrays)
    return dtype_float, dtype_amp, dtype_data


def delay_and_sum_numba(frame, focal_law, fillvalue=0., result=None, block_size=None, numthreads=None,
                        interpolate_position='nearest'):
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
    from ._das_numba import _delay_and_sum_amplitudes_linear, _delay_and_sum_amplitudes_nearest
    from concurrent.futures import ThreadPoolExecutor
    from ...utils import chunk_array

    # numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape

    _check_shapes(frame, focal_law)
    dtype_float, dtype_amp, dtype_data = _infer_datatypes(frame, focal_law, result)

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)

    weighted_scanlines = frame.scanlines * focal_law.scanline_weights[:, np.newaxis]

    # Parameters for multithrading:
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
                focal_law.amplitudes_tx[chunk],
                focal_law.amplitudes_rx[chunk],
                frame.time.step, frame.time.start, fillvalue,
                result[chunk]))
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()

    return result


def delay_and_sum_naive(frame, focal_law, fillvalue=0., result=None, interpolate_position='nearest'):
    """
    Naive (and slow) implementation. Keep for test only.

    Do not use in production code!
    """
    numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape

    _check_shapes(frame, focal_law)
    _, _, dtype_data = _infer_datatypes(frame, focal_law, result)

    if result is None:
        result = np.full((numpoints,), 0, dtype=dtype_data)
    assert result.shape == (numpoints,)
    if interpolate_position != 'nearest':
        raise NotImplementedError

    lookup_times_tx = focal_law.lookup_times_tx
    lookup_times_rx = focal_law.lookup_times_rx
    amplitudes_tx = focal_law.amplitudes_tx
    amplitudes_rx = focal_law.amplitudes_rx
    scanline_weights = focal_law.scanline_weights
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
            lookup_time = lookup_times_tx[point, tx[scan]] + lookup_times_rx[point, rx[scan]]
            lookup_index = int(round((lookup_time - t0) / dt))

            if lookup_index < 0 or lookup_index >= numsamples:
                result[point] += fillvalue
            else:
                result[point] += scanline_weights[scan] * amplitudes_tx[point, tx[scan]] \
                                 * amplitudes_rx[point, rx[scan]] * scanlines[scan, lookup_index]
    return result


def delay_and_sum_cpu(frame, focal_law, fillvalue=0., result=None, block_size=None, numthreads=None,
                      interpolate_position='nearest'):
    from . import _delay_and_sum_cpu as das_cpu
    numscanlines = frame.numscanlines
    numpoints, numelements = focal_law.lookup_times_tx.shape
    numsamples = len(frame.time)

    _check_shapes(frame, focal_law)
    dtype_float, dtype_amp, dtype_data = _infer_datatypes(frame, focal_law, result)

    weighted_scanlines = frame.scanlines * focal_law.scanline_weights[:, np.newaxis]

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
        func = getattr(das_cpu, '_delay_and_sum_nearest_{}'.format(sig_types))
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
    amplitudes_tx = np.array(focal_law.amplitudes_tx, dtype=dtype_amp, copy=False)
    amplitudes_rx = np.array(focal_law.amplitudes_rx, dtype=dtype_amp, copy=False)

    func(weighted_scanlines, tx, rx,
         lookup_times_tx,
         lookup_times_rx,
         amplitudes_tx,
         amplitudes_rx,
         invdt, tstart, fillvalue,
         result, numpoints, numsamples, numelements,
         numscanlines)
    return result


# alias for default delay and sum
delay_and_sum = delay_and_sum_cpu
