"""
Common functions used in imaging.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

from .. import settings as s
from ..exceptions import InvalidDimension
from ..helpers import chunk_array

__all__ = ['find_minimum_times']


def find_minimum_times(time_1, time_2, dtype=None, dtype_indices=None, block_size=None, numthreads=None):
    """
    For i=1:n and j=1:p,

        out_min_times(i, j)   := min_{k=1:m}    time_1[i, k] + time_2[k, j]
        out_min_indices(i, j) := argmin_{k=1:m} time_1[i, k] + time_2[k, j]


    Parameters
    ----------
    time_1
        Shape: (n, m)
    time_2
        Shape: (m, p)
    dtype
    dtype_indices

    Returns
    -------
    out_min_times
        Shape: (n, p)
    out_min_indices
        Shape: (n, p)

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


@numba.jit(nopython=True, nogil=True, cache=True)
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
