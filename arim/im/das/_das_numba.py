import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
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


@numba.jit(nopython=True, nogil=True)
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


@numba.jit(nopython=True, nogil=True)
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


@numba.jit(nopython=True, nogil=True)
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
