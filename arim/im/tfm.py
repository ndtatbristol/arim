"""
Main functions: :func:`contact_tfm`, :func:`tfm_for_view`

.. seealso::

    :ref:`tfm`

"""

import numpy as np
from collections import namedtuple
import numba
from concurrent.futures import ThreadPoolExecutor
import logging

from .. import geometry as g, model, ut
from . import das
from ..exceptions import ArimWarning
from ..helpers import chunk_array

logger = logging.getLogger(__name__)


class IncompleteFrameWarning(ArimWarning):
    pass


class TxRxAmplitudes:
    """
    Tfm amplitudes where A_ij = B_i * B'_j

    Parameters
    ----------
    amplitudes_tx : ndarray
        Shape (numelements, numgridpoints)
    amplitudes_rx : ndarray
        Shape (numelements, numgridpoints)
    force_c_order : bool
        Default True

    """

    __slots__ = ("amplitudes_tx", "amplitudes_rx")

    def __init__(self, amplitudes_tx, amplitudes_rx, force_c_order=True):
        if force_c_order:
            amplitudes_tx = np.ascontiguousarray(amplitudes_tx)
            amplitudes_rx = np.ascontiguousarray(amplitudes_rx)

        assert amplitudes_tx.dtype == amplitudes_rx.dtype

        assert amplitudes_tx.ndim == amplitudes_rx.ndim == 2
        self.amplitudes_tx = amplitudes_tx
        self.amplitudes_rx = amplitudes_rx

    @property
    def shape(self):
        return self.amplitudes_tx.shape

    @property
    def dtype(self):
        return self.amplitudes_tx.dtype

    def __iter__(self):
        # easy unpacking
        yield self.amplitudes_tx
        yield self.amplitudes_rx


class FocalLaw:
    """
    Focal law for TFM.

    Parameters
    ----------
    lookup_times_tx : ndarray
    lookup_times_rx : ndarray
    amplitudes : TxRxAmplitudes or ndarray or None
    scanline_weights : ndarray or None
        Use
    force_c_order

    Attributes
    ----------
    lookup_times_tx : ndarray
    lookup_times_rx : ndarray
    amplitudes : TxRxAmplitudes or ndarray or None
    scanline_weights : ndarray or None
    numscanlines : int
    numelements

    """

    __slots__ = (
        "lookup_times_tx",
        "lookup_times_rx",
        "amplitudes",
        "scanline_weights",
        "_numscanlines",
    )

    def __init__(
        self,
        lookup_times_tx,
        lookup_times_rx,
        amplitudes=None,
        scanline_weights=None,
        force_c_order=True,
    ):
        if force_c_order:
            lookup_times_tx = np.ascontiguousarray(lookup_times_tx)
            lookup_times_rx = np.ascontiguousarray(lookup_times_rx)

        assert lookup_times_tx.ndim == lookup_times_rx.ndim == 2
        assert lookup_times_tx.shape == lookup_times_rx.shape

        if scanline_weights is not None:
            if force_c_order:
                scanline_weights = np.ascontiguousarray(scanline_weights)
            assert scanline_weights.ndim == 1
            numscanlines = scanline_weights.shape[0]
        else:
            numscanlines = None

        if amplitudes is not None:
            # don't force to C order because 'amplitudes' may be a arim.model.ModelAmplitudes or a TxRxAmplitudes
            if isinstance(amplitudes, TxRxAmplitudes):
                # arrays of shape (numgridpoints, numelements)
                assert amplitudes.shape == lookup_times_tx.shape
            else:
                # arrays of shape (numgridpoints, numscanlines)
                assert amplitudes.ndim == 2
                assert amplitudes.shape[1] == lookup_times_tx.shape[1]
                if numscanlines is not None:
                    assert amplitudes.shape[1] == numscanlines
                else:
                    numscanlines = amplitudes.shape[1]

        self.lookup_times_tx = lookup_times_tx
        self.lookup_times_rx = lookup_times_rx
        self.amplitudes = amplitudes
        self.scanline_weights = scanline_weights
        self._numscanlines = numscanlines

    @property
    def numelements(self):
        return self.lookup_times_tx.shape[1]

    @property
    def numgridpoints(self):
        return self.lookup_times_tx.shape[0]

    @property
    def numscanlines(self):
        if self._numscanlines is None:
            raise AttributeError("no data for inferring the number of scanlines")
        else:
            return self._numscanlines

    def weigh_scanlines(self, scanlines):
        """
        Multiply each scanline by a weight

        Canonical usage: use a weight of 2 for the scanlines i != j in the HMC for contact TFM.

        Parameters
        ----------
        scanlines : ndarray

        Returns
        -------
        weighted_scanlines : ndarray
        """
        assert scanlines.ndim == 2
        if self.scanline_weights is None:
            return scanlines
        else:
            out = scanlines * self.scanline_weights[:, np.newaxis]
            assert out.ndim == 2
            return out

    def amp_dtype(self):
        if self.amplitudes is not None:
            return self.amplitudes.dtype
        else:
            return None


class TfmResult:
    """
    Data container for TFM result
    """

    __slots__ = ("res", "grid")

    def __init__(self, res, grid):
        assert res.shape == grid.shape
        self.res = res
        self.grid = grid

    def maximum_intensity_in_rectbox(
        self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None
    ):
        """
        Returns the maximum absolute intensity of the TFM image in the rectangular box
        defined by the parameters. If a parameter is None, the box is unbounded in the
        corresponding direction.

        Intensity is given as it is (no dB conversion).

        Parameters
        ----------
        xmin : float or None
        xmax : float or None
        ymin : float or None
        ymax : float or None
        zmin : float or None
        zmax : float or None

        Returns
        -------
        intensity : float

        """
        assert self.res is not None
        area_of_interest = self.grid.points_in_rectbox(
            xmin, xmax, ymin, ymax, zmin, zmax
        )
        return self.maximum_intensity_in_area(area_of_interest)

    def maximum_intensity_in_area(self, area):
        """
        Returns the maximum absolute intensity of the TFM image in an area.

        Intensity is given as it is (no dB conversion).

        Parameters
        ----------
        area : ndarray or None or slice
            Indices of the grid.

        Returns
        -------
        intensity : float
        """
        if area is None:
            area = slice(None)
        return np.nanmax(np.abs(self.res[area]))


def contact_tfm(
    frame,
    grid,
    velocity,
    amplitudes=None,
    scanline_weights="default",
    **kwargs_delay_and_sum
):
    """
    Contact TFM

    Supports HMC and FMC frame.

    Parameters
    ----------
    frame : Frame
    grid : Points
    velocity : float
    amplitudes : None or ndarray or TxRxAmplitudes
    scanline_weights : None or ndarray or 'default'
        Default: 2 for i!=j, 1 for i==j. None: 1 for all scanlines.
    kwargs_delay_and_sum : dict

    Returns
    -------
    tfm_res : TfmResult

    """
    lookup_times = (
        g.distance_pairwise(grid.to_1d_points(), frame.probe.locations) / velocity
    )
    assert lookup_times.ndim == 2
    assert lookup_times.shape == (grid.numpoints, frame.probe.numelements)

    if amplitudes is not None:
        if __debug__ and not frame.is_complete_assuming_reciprocity():
            logger.warning(
                "Possible erroneous usage of a noncomplete frame in TFM; "
                "use Frame.expand_frame_assuming_reciprocity()",
                IncompleteFrameWarning,
            )

    if scanline_weights == "default":
        scanline_weights = ut.default_scanline_weights(frame.tx, frame.rx)
    focal_law = FocalLaw(lookup_times, lookup_times, amplitudes, scanline_weights)

    res = das.delay_and_sum(frame, focal_law, **kwargs_delay_and_sum)
    res = res.reshape(grid.shape)
    return TfmResult(res, grid)


def tfm_for_view(frame, grid, view, amplitudes=None, **kwargs_delay_and_sum):
    """
    TFM for a view

    Parameters
    ----------
    frame : Frame
    grid : Points
    velocity : float
    amplitudes : None or ndarray or TxRxAmplitudes
    kwargs_delay_and_sum : dict

    Returns
    -------
    tfm_res : TfmResult

    Notes
    -----
    No check is performed to ensure that the calculation is safe for incomplete frames (HMC for example).
    In this case, an :exc:`IncompleteFrameWarning` is emitted.

    """
    # do not use scanline weights, it is likely to be ill-defined here
    if not frame.is_complete_assuming_reciprocity():
        logger.warning(
            "Possible erroneous usage of a noncomplete frame in TFM; "
            "use Frame.expand_frame_assuming_reciprocity()",
            IncompleteFrameWarning,
        )

    lookup_times_tx = view.tx_path.rays.times.T
    lookup_times_rx = view.rx_path.rays.times.T

    focal_law = FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes)

    res = das.delay_and_sum(frame, focal_law, **kwargs_delay_and_sum)
    res = res.reshape(grid.shape)
    return TfmResult(res, grid)


ExtramaLookupTimes = namedtuple(
    "ExtramaLookupTimes",
    "tmin tmax tx_elt_for_tmin rx_elt_for_tmin tx_elt_for_tmax rx_elt_for_tmax",
)


def extrema_lookup_times_in_rectbox(
    grid,
    lookup_times_tx,
    lookup_times_rx,
    tx,
    rx,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    zmin=None,
    zmax=None,
):
    """
    Returns the minimum and maximum of the lookup times in an rectangular box.
    The output is returned as a named tuple for convenience.

    Parameters
    ----------
    xmin : float or None
    xmax : float or None
    ymin : float or None
    ymax : float or None
    zmin : float or None
    zmax : float or None

    Returns
    -------
    out : ExtramaLookupTimes
        Respectively: returns the minimum and maximum times (fields ``tmin`` and ``tmax``) and the elements indices
        corresponding to these values. If several couples of elements are matching, the first couple is returned.

    """
    area_of_interest = grid.points_in_rectbox(
        xmin, xmax, ymin, ymax, zmin, zmax
    ).ravel()

    sub_lookup_times_tx = np.ascontiguousarray(lookup_times_tx[area_of_interest, ...])
    sub_lookup_times_rx = np.ascontiguousarray(lookup_times_rx[area_of_interest, ...])
    tx = np.ascontiguousarray(tx)
    rx = np.ascontiguousarray(rx)
    out = _extrema_lookup_times(sub_lookup_times_tx, sub_lookup_times_rx, tx, rx)
    return ExtramaLookupTimes(*out)


@numba.jit(nopython=True, cache=True)
def _extrema_lookup_times(lookup_times_tx, lookup_times_rx, tx_list, rx_list):
    numpoints, _ = lookup_times_tx.shape
    numscanlines = tx_list.shape[0]
    tmin = np.inf
    tmax = -np.inf
    tx_elt_for_tmin = tx_list[0]
    tx_elt_for_tmax = tx_list[0]
    rx_elt_for_tmin = rx_list[0]
    rx_elt_for_tmax = rx_list[0]

    for point in range(numpoints):
        for scanline in range(numscanlines):
            t = lookup_times_tx[point, scanline] + lookup_times_rx[point, scanline]
            if t > tmax:
                tmax = t
                tx_elt_for_tmax = tx_list[scanline]
                rx_elt_for_tmax = rx_list[scanline]
            if t < tmin:
                tmin = t
                tx_elt_for_tmin = tx_list[scanline]
                rx_elt_for_tmin = rx_list[scanline]
    return (
        tmin,
        tmax,
        tx_elt_for_tmin,
        rx_elt_for_tmin,
        tx_elt_for_tmax,
        rx_elt_for_tmax,
    )


def tfm_with_scattering(
    frame,
    grid,
    view,
    fillvalue,
    scattering_fn,
    tx_ray_weights,
    rx_ray_weights,
    tx_scattering_angles,
    rx_scattering_angles,
    scanline_weights=None,
    divide_by_sensitivity=True,
    numthreads=None,
    block_size=None,
):
    # todo: refactor this
    numscanlines = frame.numscanlines
    numpoints = grid.numpoints
    numelements = frame.probe.numelements

    tx_lookup_times = np.ascontiguousarray(view.tx_path.rays.times.T)
    rx_lookup_times = np.ascontiguousarray(view.rx_path.rays.times.T)
    tx_amplitudes = np.ascontiguousarray(tx_ray_weights.T)
    rx_amplitudes = np.ascontiguousarray(rx_ray_weights.T)

    if scanline_weights is None:
        scanline_weights = ut.default_scanline_weights(frame.tx, frame.rx)

    weighted_scanlines = frame.scanlines * scanline_weights[:, np.newaxis]

    assert weighted_scanlines.shape[0] == numscanlines
    assert rx_scattering_angles.shape == (numelements, numpoints)
    assert tx_lookup_times.shape == (numpoints, numelements)
    assert rx_lookup_times.shape == (numpoints, numelements)
    assert tx_scattering_angles.shape == (numelements, numpoints)
    assert rx_scattering_angles.shape == (numelements, numpoints)
    assert tx_amplitudes.shape == (numpoints, numelements)
    assert rx_amplitudes.shape == (numpoints, numelements)

    dict(dt=frame.time.step, t0=frame.time.start, fillvalue=fillvalue)

    # TODO: improve selection of datatype
    tfm_result = np.zeros(grid.numpoints, np.complex)
    sensitivity_result = np.zeros(grid.numpoints, np.float)

    if block_size is None:
        block_size = 1000

    futures = []
    with ThreadPoolExecutor(max_workers=numthreads) as executor:
        for chunk in chunk_array((grid.numpoints,), block_size, axis=0):
            _tfm_with_scattering(
                weighted_scanlines,
                scanline_weights,
                frame.tx,
                frame.rx,
                tx_lookup_times[chunk],
                rx_lookup_times[chunk],
                tx_amplitudes[chunk],
                rx_amplitudes[chunk],
                scattering_fn,
                tx_scattering_angles[..., chunk[0]],
                rx_scattering_angles[..., chunk[0]],
                frame.time.step,
                frame.time.start,
                fillvalue,
                sensitivity_result[chunk],
                tfm_result[chunk],
            )
    # Raise exceptions that happened, if any:
    for future in futures:
        future.result()

    tfm_result = tfm_result.reshape((grid.numx, grid.numy, grid.numz))
    sensitivity_result = sensitivity_result.reshape((grid.numx, grid.numy, grid.numz))
    scaled_tfm_result = tfm_result / sensitivity_result

    # Dirty hack: return a BaseTFM object.
    # TODO: create a proper TfmResult object
    tfm_obj = BaseTFM(frame, grid)
    tfm_obj.view = view
    tfm_obj.scaled_res = scaled_tfm_result
    tfm_obj.raw_res = tfm_result
    if divide_by_sensitivity:
        tfm_obj.res = tfm_obj.scaled_res
    else:
        tfm_obj.res = tfm_obj.raw_res

    return tfm_obj, tfm_result, sensitivity_result


def _tfm_with_scattering(
    weighted_scanlines,
    scanline_weights,
    tx,
    rx,
    tx_lookup_times,
    rx_lookup_times,
    tx_amplitudes,
    rx_amplitudes,
    scattering_fn,
    tx_scattering_angles,
    rx_scattering_angles,
    dt,
    t0,
    fillvalue,
    sensitivity_result,
    tfm_result,
):
    """
    Forward model::

        F_ij = Q_i Q'_j S_ij exp(+i omega (tau_i + tau'_j))

    Notation::

        P_ij = Q_i Q'_j S_ij

    Sensitivity::

        I_0 = sum_i sum_j |P_ij|^2

    TFM::

        I = sum_i sum_j conjugate(P_ij) / I_0 * F_ij exp(-i omega (tau_i + tau'_j))


    Parameters
    ----------
    weighted_scanlines
    scanline_weights
    tx
    rx
    tx_lookup_times
        tau
    rx_lookup_times
        tau'
    tx_amplitudes
        Q
    rx_amplitudes
        Q'
    scattering_fn
        Returns S
    tx_scattering_angles
    rx_scattering_angles
    dt : float
    t0 : float
    fillvalue : float
    sensitivity_result
        I_0
    tfm_result
        I

    Returns
    -------
    None. Write on sensitivity_result and tfm_result


    """
    numelements, numpoints = tx_scattering_angles.shape
    numscanlines = tx.shape[0]
    assert tx.shape == (numscanlines,)
    assert rx.shape == (numscanlines,)
    assert weighted_scanlines.shape[0] == numscanlines
    assert rx_scattering_angles.shape == (numelements, numpoints)
    assert tx_lookup_times.shape == (numpoints, numelements)
    assert rx_lookup_times.shape == (numpoints, numelements)
    assert tx_amplitudes.shape == (numpoints, numelements)
    assert rx_amplitudes.shape == (numpoints, numelements)
    assert sensitivity_result.shape == (numpoints,)
    assert tfm_result.shape == (numpoints,)

    # This can be big, warning:
    scattering_amplitudes = scattering_fn(
        np.take(tx_scattering_angles, tx, axis=0),
        np.take(rx_scattering_angles, rx, axis=0),
    )
    scattering_amplitudes = np.ascontiguousarray(scattering_amplitudes.T)
    assert scattering_amplitudes.shape == (numpoints, numscanlines)

    # Model amplitudes P_ij
    model_amplitudes = (
        scattering_amplitudes
        * np.take(tx_amplitudes, tx, axis=1)
        * np.take(rx_amplitudes, rx, axis=1)
    )
    del scattering_amplitudes

    # Compute sensitivity image (write result on sensitivity_result)
    model.sensitivity_image(model_amplitudes, scanline_weights, sensitivity_result)

    # Remark: the sensitivity here does not depend on the
    tfm_amplitudes = model_amplitudes.conjugate()
    del model_amplitudes

    das.das._general_delay_and_sum_nearest(
        weighted_scanlines,
        tx,
        rx,
        tx_lookup_times,
        rx_lookup_times,
        tfm_amplitudes,
        dt,
        t0,
        fillvalue,
        tfm_result,
    )
