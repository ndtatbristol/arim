"""
Main functions: :func:`contact_tfm`, :func:`tfm_for_view`

.. seealso::

    :ref:`tfm`

"""

import logging
import warnings
from collections import namedtuple

import numba
import numpy as np

from .. import geometry as g
from .. import ut
from ..exceptions import ArimWarning
from ..ray import RayGeometry
from . import das

logger = logging.getLogger(__name__)


class IncompleteFrameWarning(ArimWarning):
    pass


class TxRxAmplitudes:
    """
    Tfm amplitudes where A_ij = B_i * B'_j

    Parameters
    ----------
    amplitudes_tx : ndarray
        Shape (numtx, numgridpoints)
    amplitudes_rx : ndarray
        Shape (numrx, numgridpoints)
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
        warnings.warn(
            DeprecationWarning("TxRxAmplitudes.shape is deprecated because ambiguous")
        )
        return self.amplitudes_tx.shape

    @property
    def dtype(self):
        return self.amplitudes_tx.dtype

    def __iter__(self):
        # easy unpacking
        yield self.amplitudes_tx
        yield self.amplitudes_rx


def angle_limit(theta, phi, limit, elev=0.0, azim=np.pi/2, window='hanning'):
    """
    Apply an angle limit to the provided angles. Hanning window applied over
    limit centred on elevation. Assumes 2D.

    Parameters
    ----------
    theta : ndarray
        Shape (numtx, numgridpoints)
        Angle made by the ray from the probe.
    limit : float
    elev : float, optional

    Returns
    -------
    TxRxAmplitudes

    """
    lookvec = np.asarray(
        [np.sin(azim) * np.sin(elev), np.cos(azim) * np.sin(elev), np.cos(elev)]
    )
    radial = np.asarray(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(theta)]
    ).transpose(1, 2, 0)
    gamma = np.arccos(np.dot(radial, lookvec))
    amplitudes = np.zeros(gamma.shape)
    if window == 'hanning':
        amplitudes[np.abs(gamma) <= limit] = (
            np.cos(gamma[np.abs(gamma) <= limit] * np.pi / limit) + 1
        ) / 2
    elif window == 'rectangular':
        amplitudes[np.abs(gamma) <= limit] = 1
    else:
        raise NotImplementedError('Invalid angle limit window.')
        
    return amplitudes


def angle_limit_in_contact(grid, probe, limit, elev=0.0, azim=np.pi/2, window='hanning'):
    """
    Calculates the amplitudes required to implement an amplitude limit for the
    focal law when the grid is in contact with the probe (i.e. one leg, no
    reflections).

    Parameters
    ----------
    grid : Points
    probe : Probe
    limit : float
    elev : float, optional
    azim : float, optional

    Returns
    -------
    TxRxAmplitudes

    """
    grid = grid.to_1d_points().reshape((-1, 1))
    probe = probe.locations.reshape((1, -1))
    x = grid.x - probe.x
    y = grid.y - probe.y
    z = grid.z - probe.z

    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi = np.arctan2(y, x)
    amplitudes = angle_limit(theta, phi, limit, elev, azim, window=window)
    return TxRxAmplitudes(amplitudes, amplitudes)


def angle_limit_for_view(view, limit, tx_elev=0.0, tx_azim=np.pi/2, rx_elev=0.0, rx_azim=np.pi/2, window='hanning'):
    """
    Calculates the amplitudes required to implement an amplitude limit for the
    focal law for a provided view (i.e. angle limit applied to the final leg of
    the view).
    
    Note that `elev` and `azim` are defined in terms of the basis of the last
    interface. For view LL-T with transmit path reflecting from the backwall,
    `tx_elev = 0.0` points in the opposite direction to `rx_elev` because the
    basis of the backwall should have `z_hat = [0, 0, -1]`, and the basis of
    the probe should have `z_hat = [0, 0, 1]`.

    Parameters
    ----------
    view : View
    limit : float
    elev : float, optional
    azim : float, optional

    Returns
    -------
    TxRxAmplitudes

    """
    tx_ray = RayGeometry.from_path(view[0])
    rx_ray = RayGeometry.from_path(view[1])

    tx_amps = angle_limit(
        tx_ray.out_leg_polar(-2), tx_ray.out_leg_azimuth(-2), limit, tx_elev, tx_azim, window=window
    ).transpose()
    rx_amps = angle_limit(
        rx_ray.out_leg_polar(-2), rx_ray.out_leg_azimuth(-2), limit, rx_elev, rx_azim, window=window
    ).transpose()

    return TxRxAmplitudes(tx_amps, rx_amps)


class FocalLaw:
    """
    Focal law for TFM.

    Parameters
    ----------
    lookup_times_tx : ndarray
    lookup_times_rx : ndarray
    amplitudes : TxRxAmplitudes or ndarray or None
    timetrace_weights : ndarray or None
        Use
    force_c_order

    Attributes
    ----------
    lookup_times_tx : ndarray
    lookup_times_rx : ndarray
    amplitudes : TxRxAmplitudes or ndarray or None
    timetrace_weights : ndarray or None
    numtimetraces : int
    numelements

    """

    __slots__ = (
        "lookup_times_tx",
        "lookup_times_rx",
        "amplitudes",
        "timetrace_weights",
        "_numtimetraces",
    )

    def __init__(
        self,
        lookup_times_tx,
        lookup_times_rx,
        amplitudes=None,
        timetrace_weights=None,
        force_c_order=True,
    ):
        if force_c_order:
            lookup_times_tx = np.ascontiguousarray(lookup_times_tx)
            lookup_times_rx = np.ascontiguousarray(lookup_times_rx)

        assert lookup_times_tx.ndim == lookup_times_rx.ndim == 2
        assert lookup_times_tx.shape[0] == lookup_times_rx.shape[0]

        if timetrace_weights is not None:
            if force_c_order:
                timetrace_weights = np.ascontiguousarray(timetrace_weights)
            assert timetrace_weights.ndim == 1
            numtimetraces = timetrace_weights.shape[0]
        else:
            numtimetraces = None

        if amplitudes is not None:
            # don't force to C order because 'amplitudes' may be a arim.model.ModelAmplitudes or a TxRxAmplitudes
            if isinstance(amplitudes, TxRxAmplitudes):
                # arrays of shape (numgridpoints, num{tx,rx})
                assert amplitudes.amplitudes_tx.shape == lookup_times_tx.shape
                assert amplitudes.amplitudes_rx.shape == lookup_times_rx.shape
            else:
                # arrays of shape (numgridpoints, numtimetraces)
                assert amplitudes.ndim == 2
                assert amplitudes.shape[1] == lookup_times_tx.shape[1]
                if numtimetraces is not None:
                    assert amplitudes.shape[1] == numtimetraces
                else:
                    numtimetraces = amplitudes.shape[1]

        self.lookup_times_tx = lookup_times_tx
        self.lookup_times_rx = lookup_times_rx
        self.amplitudes = amplitudes
        self.timetrace_weights = timetrace_weights
        self._numtimetraces = numtimetraces

    @property
    def scanline_weights(self):
        warnings.warn(
            DeprecationWarning(
                "FocalLaw.scanline_weights is deprecated. Use FocalLaw.timetrace_weights"
            )
        )
        return self.timetrace_weights

    @property
    def numelements(self):
        warnings.warn(
            DeprecationWarning(
                "FocalLaw.numelements is deprecated. Use FocalLaw.numtx or FocalLaw.numrx"
            )
        )
        return self.lookup_times_tx.shape[1]

    @property
    def numtx(self):
        return self.lookup_times_tx.shape[1]

    @property
    def numrx(self):
        return self.lookup_times_rx.shape[1]

    @property
    def numgridpoints(self):
        return self.lookup_times_tx.shape[0]

    @property
    def numtimetraces(self):
        if self._numtimetraces is None:
            raise AttributeError("no data for inferring the number of timetraces")
        else:
            return self._numtimetraces

    def weigh_timetraces(self, timetraces):
        """
        Multiply each timetrace by a weight

        Canonical usage: use a weight of 2 for the timetraces i != j in the HMC for contact TFM.

        Parameters
        ----------
        timetraces : ndarray

        Returns
        -------
        weighted_timetraces : ndarray
        """
        assert timetraces.ndim == 2
        if self.timetrace_weights is None:
            return timetraces
        else:
            out = timetraces * self.timetrace_weights[:, np.newaxis]
            assert out.ndim == 2
            return out

    def weigh_scanlines(self, timetraces):
        warnings.warn(
            DeprecationWarning(
                "FocalLaw.weigh_scanlines is deprecated. Use FocalLaw.weigh_timetraces"
            )
        )
        return self.weigh_timetraces(self, timetraces)

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
    timetrace_weights="default",
    **kwargs_delay_and_sum,
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
    timetrace_weights : None or ndarray or 'default'
        Default: 2 for i!=j, 1 for i==j. None: 1 for all timetraces.
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
                "use Frame.expand_frame_assuming_reciprocity()"
            )

    if timetrace_weights == "default":
        timetrace_weights = ut.default_timetrace_weights(frame.tx, frame.rx)
    focal_law = FocalLaw(lookup_times, lookup_times, amplitudes, timetrace_weights)

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
    # do not use timetrace weights, it is likely to be ill-defined here
    if not frame.is_complete_assuming_reciprocity():
        logger.warning(
            "Possible erroneous usage of a noncomplete frame in TFM; "
            "use Frame.expand_frame_assuming_reciprocity()"
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
    numtimetraces = tx_list.shape[0]
    tmin = np.inf
    tmax = -np.inf
    tx_elt_for_tmin = tx_list[0]
    tx_elt_for_tmax = tx_list[0]
    rx_elt_for_tmin = rx_list[0]
    rx_elt_for_tmax = rx_list[0]

    for point in range(numpoints):
        for timetrace in range(numtimetraces):
            t = lookup_times_tx[point, timetrace] + lookup_times_rx[point, timetrace]
            if t > tmax:
                tmax = t
                tx_elt_for_tmax = tx_list[timetrace]
                rx_elt_for_tmax = rx_list[timetrace]
            if t < tmin:
                tmin = t
                tx_elt_for_tmin = tx_list[timetrace]
                rx_elt_for_tmin = rx_list[timetrace]
    return (
        tmin,
        tmax,
        tx_elt_for_tmin,
        rx_elt_for_tmin,
        tx_elt_for_tmax,
        rx_elt_for_tmax,
    )
