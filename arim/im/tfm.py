"""
This module defines classes to perform TFM and TFM-like imaging.
"""

import numpy as np
import warnings
from collections import namedtuple
import numba
from concurrent.futures import ThreadPoolExecutor

from .. import geometry as g
from .. import model, ut
from . import amplitudes
from . import das
from .. import settings as s
from .. import core as c
from ..exceptions import ArimWarning
from ..helpers import chunk_array

__all__ = ['BaseTFM', 'ContactTFM', 'SingleViewTFM', 'SimpleTFM', 'tfm_with_scattering']

ExtramaLookupTimes = namedtuple('ExtramaLookupTimes',
                                'tmin tmax tx_elt_for_tmin rx_elt_for_tmin tx_elt_for_tmax rx_elt_for_tmax')


class BaseTFM:
    """
    Base object for TFM-like algorithms. Define the general workflow.

    To implement a TFM-like algorithm: create a new class that inherits this object and implement if necessary the
    computation of scanlines amplitudes, weights and the delay laws.


    Parameters
    ----------
    frame : Frame
    grid : Grid
    amplitudes : Amplitudes or str
        An amplitude object. Accepted keywords: 'uniform'.
    dtype : numpy.dtype
    scanlines_weight : ndarray or None
    geom_probe_to_grid : GeometryHelper

    Attributes
    ----------
    result : ndarray
    frame : Frame
    grid : Grid
    dtype
    amplitudes : Amplitudes
    geom_probe_to_grid

    """

    def __init__(self, frame, grid, amplitudes_tx='uniform', amplitudes_rx='uniform',
                 scanline_weights=None, dtype=None, geom_probe_to_grid=None):
        self.frame = frame
        self.grid = grid

        if dtype is None:
            dtype = s.FLOAT
        self.dtype = dtype

        self.res = None

        if geom_probe_to_grid is None:
            geom_probe_to_grid = g.GeometryHelper(frame.probe.locations, grid.as_points,
                                                  frame.probe.pcs)
        else:
            assert geom_probe_to_grid.is_valid(frame.probe, grid.as_points)
        self._geom_probe_to_grid = geom_probe_to_grid

        if scanline_weights is not None:
            scanline_weights = np.asarray(scanline_weights)
            assert scanline_weights.shape == frame.numscanlines
        self._scanline_weights = scanline_weights

        if amplitudes_tx == 'uniform':
            amplitudes_tx = amplitudes.UniformAmplitudes(frame, grid)
        if amplitudes_rx == 'uniform':
            amplitudes_rx = amplitudes.UniformAmplitudes(frame, grid)
        self.amplitudes_tx = amplitudes_tx
        self.amplitudes_rx = amplitudes_rx

    def run(self, delay_and_sum_func=None, **delay_and_sum_kwargs):
        """
        Compute TFM: get the lookup times, the amplitudes, and delay and sum
        the scanlines.

        Parameters
        ----------
        delay_and_sum_func : delay_and_sum_func to use
        delay_and_sum_kwargs : extra arguments for the delay and sum function

        Returns
        -------
        result

        """
        self.hook_start_run()

        lookup_times_tx = self.get_lookup_times_tx()
        lookup_times_rx = self.get_lookup_times_rx()
        amplitudes_tx = self.get_amplitudes_tx()
        amplitudes_rx = self.get_amplitudes_rx()
        scanline_weights = self.get_scanline_weights()

        focal_law = c.FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes_tx,
                               amplitudes_rx, scanline_weights)

        focal_law = self.hook_focal_law(focal_law)
        self.focal_law = focal_law

        if delay_and_sum_func is None:
            delay_and_sum_func = das.delay_and_sum
        if delay_and_sum_kwargs is None:
            delay_and_sum_kwargs = {}

        res = delay_and_sum_func(self.frame, focal_law, **delay_and_sum_kwargs)

        res = self.hook_result(res)

        self.res = res

        return res

    def get_amplitudes_tx(self):
        return self.amplitudes_tx()

    def get_amplitudes_rx(self):
        return self.amplitudes_rx()

    def get_lookup_times_tx(self):
        raise NotImplementedError('must be implemented by child class')

    def get_lookup_times_rx(self):
        raise NotImplementedError('must be implemented by child class')

    def get_scanline_weights(self):
        """
        Scanline weights (most common case: multiply by 2. the non-pulse-echo scanlines
        in HMC).

        If scanline_weights were to given at the object contruction, return them.
        Otherwise, use arim.ut.default_scanline_weights
        """
        if self._scanline_weights is not None:
            return self._scanline_weights
        else:
            return ut.default_scanline_weights(self.frame.tx, self.frame.rx)

    def hook_start_run(self):
        """Implement this method in child class if necessary."""
        pass

    def hook_focal_law(self, focal_law):
        """Hooked called after creating the focal law.
        Implement this method in child class if necessary."""
        return focal_law

    def hook_result(self, res):
        """Implement this method in child class if necessary.

        Default behaviour: reshape results (initially 1D array) to 3D array with
        same shape as the grid.
        """
        return res.reshape((self.grid.numx, self.grid.numy, self.grid.numz))

    @property
    def geom_probe_to_grid(self):
        return self._geom_probe_to_grid

    def maximum_intensity_in_rectbox(self, xmin=None, xmax=None, ymin=None, ymax=None,
                                     zmin=None, zmax=None):
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
        area_of_interest = self.grid.points_in_rectbox(xmin, xmax, ymin, ymax,
                                                       zmin, zmax)
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

    def extrema_lookup_times_in_rectbox(self, xmin=None, xmax=None, ymin=None, ymax=None,
                                        zmin=None, zmax=None):
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

        Notes
        -----

            $$\min\limits_{i,j}{\(a_i + b_j\)} = \min\limits_{i}{a_i} + \min\limits_{j}{b_j}$$

        """
        area_of_interest = self.grid.points_in_rectbox(xmin, xmax, ymin, ymax,
                                                       zmin, zmax).ravel()

        all_lookup_times_tx = self.get_lookup_times_tx()
        all_lookup_times_rx = self.get_lookup_times_rx()
        lookup_times_tx = np.ascontiguousarray(all_lookup_times_tx[area_of_interest, ...])
        lookup_times_rx = np.ascontiguousarray(all_lookup_times_rx[area_of_interest, ...])
        tx = np.ascontiguousarray(self.frame.tx)
        rx = np.ascontiguousarray(self.frame.rx)
        out = _extrema_lookup_times(lookup_times_tx, lookup_times_rx, tx, rx)
        return ExtramaLookupTimes(*out)


class ContactTFM(BaseTFM):
    """
    Contact TFM. The probe is assumed to lay on the surface of the examination
    object.
    """

    def __init__(self, speed, **kwargs):
        # This attribute is attached to the instance AND the class (double underscore):
        self.__lookup_times = None

        self.speed = speed
        super().__init__(**kwargs)

    def get_lookup_times_tx(self):
        """
        Lookup times obtained by dividing Euclidean distances between elements and
        image points by the speed (``self.speed``).
        """
        if self.__lookup_times is None:
            distance = self._geom_probe_to_grid.distance_pairwise()
            # distance = distance_pairwise(
            #     self.grid.as_points, self.frame.probe.locations, **self.distance_pairwise_kwargs)
            distance /= self.speed
            self.__lookup_times = distance
        return self.__lookup_times

    get_lookup_times_rx = get_lookup_times_tx


class SingleViewTFM(BaseTFM):
    def __init__(self, frame, grid, view, **kwargs):
        if grid.numpoints != len(view.tx_path.interfaces[-1].points):
            raise ValueError("Inconsistent grid")
        if grid.numpoints != len(view.rx_path.interfaces[-1].points):
            raise ValueError("Inconsistent grid")

        tx_rays = view.tx_path.rays
        rx_rays = view.rx_path.rays

        if (not tx_rays.indices.flags.fortran
            or not rx_rays.indices.flags.fortran
            or not tx_rays.times.flags.fortran
            or not rx_rays.times.flags.fortran):
            msg = "Rays will be converted to fortran order. If multiple TFM are performed, "
            msg += "converting the rays before passing them to this object is more computationally "
            msg += "efficient."
            warnings.warn(msg, ArimWarning, stacklevel=2)

        assert tx_rays.path[0] is frame.probe.locations
        assert rx_rays.path[0] is frame.probe.locations
        assert tx_rays.path[-1] is grid.as_points
        assert rx_rays.path[-1] is grid.as_points
        self.tx_rays = tx_rays
        self.rx_rays = rx_rays
        self.view = view

        # used in get_amplitudes
        self.fillvalue_extreme_points = np.nan

        amplitudes_tx = kwargs.get('amplitudes_tx')
        if amplitudes_tx is None:
            amplitudes_tx = amplitudes.AmplitudesRemoveExtreme(frame, grid, tx_rays)
        kwargs['amplitudes_tx'] = amplitudes_tx

        amplitudes_rx = kwargs.get('amplitudes_rx')
        if amplitudes_rx is None:
            amplitudes_rx = amplitudes.AmplitudesRemoveExtreme(frame, grid, rx_rays)
        kwargs['amplitudes_rx'] = amplitudes_rx

        super().__init__(frame, grid, **kwargs)

    def get_lookup_times_tx(self):
        """Lookup times in transmission, obtained with Fermat solver."""
        return np.ascontiguousarray(self.tx_rays.times.T)

    def get_lookup_times_rx(self):
        """Lookup times in reception, obtained with Fermat solver."""
        return np.ascontiguousarray(self.rx_rays.times.T)

    def __repr__(self):
        return "<{}: {} at {}>".format(
            self.__class__.__name__,
            str(self.view),
            hex(id(self)))


class SimpleTFM(BaseTFM):
    """
    A TFM class that takes as input the lookup times array.
    """

    def __init__(self, frame, grid, lookup_times_tx, lookup_times_rx,
                 amplitudes_tx='uniform', amplitudes_rx='uniform',
                 **kwargs):
        lookup_times_tx = np.asarray(lookup_times_tx)
        lookup_times_rx = np.asarray(lookup_times_rx)

        assert lookup_times_tx.shape == (grid.numpoints, frame.probe.numelements)
        assert lookup_times_rx.shape == (grid.numpoints, frame.probe.numelements)

        if not lookup_times_tx.flags.contiguous:
            warnings.warn("Lookup times are converted to contiguous array.", ArimWarning,
                          stacklevel=2)
            lookup_times_rx = np.ascontiguousarray(lookup_times_tx)
        if not lookup_times_tx.flags.contiguous:
            warnings.warn("Lookup times are converted to contiguous array.", ArimWarning,
                          stacklevel=2)
            lookup_times_rx = np.ascontiguousarray(lookup_times_rx)

        self._lookup_times_tx = lookup_times_tx
        self._lookup_times_rx = lookup_times_rx

        super().__init__(frame, grid, amplitudes_tx, amplitudes_rx, **kwargs)

    def get_lookup_times_tx(self):
        return self._lookup_times_tx

    def get_lookup_times_rx(self):
        return self._lookup_times_rx


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
    return tmin, tmax, tx_elt_for_tmin, rx_elt_for_tmin, tx_elt_for_tmax, rx_elt_for_tmax


def tfm_with_scattering(frame, grid, view, fillvalue, scattering_fn,
                        tx_ray_weights, rx_ray_weights,
                        tx_scattering_angles, rx_scattering_angles,
                        scanline_weights=None, divide_by_sensitivity=True,
                        numthreads=None, block_size=None):
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
            _tfm_with_scattering(weighted_scanlines, scanline_weights, frame.tx, frame.rx,
                                 tx_lookup_times[chunk], rx_lookup_times[chunk],
                                 tx_amplitudes[chunk], rx_amplitudes[chunk],
                                 scattering_fn, tx_scattering_angles[..., chunk[0]],
                                 rx_scattering_angles[..., chunk[0]],
                                 frame.time.step, frame.time.start, fillvalue,
                                 sensitivity_result[chunk], tfm_result[chunk])
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
        weighted_scanlines, scanline_weights, tx, rx, tx_lookup_times, rx_lookup_times,
        tx_amplitudes, rx_amplitudes, scattering_fn,
        tx_scattering_angles, rx_scattering_angles,
        dt, t0, fillvalue,
        sensitivity_result,
        tfm_result):
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
    scattering_amplitudes = scattering_fn(np.take(tx_scattering_angles, tx, axis=0),
                                          np.take(rx_scattering_angles, rx, axis=0))
    scattering_amplitudes = np.ascontiguousarray(scattering_amplitudes.T)
    assert scattering_amplitudes.shape == (numpoints, numscanlines)

    # Model amplitudes P_ij
    model_amplitudes = (scattering_amplitudes * np.take(tx_amplitudes, tx, axis=1)
                        * np.take(rx_amplitudes, rx, axis=1))
    del scattering_amplitudes

    # Compute sensitivity image (write result on sensitivity_result)
    model.sensitivity_image(model_amplitudes, scanline_weights,
                            sensitivity_result)

    # Remark: the sensitivity here does not depend on the
    tfm_amplitudes = model_amplitudes.conjugate()
    del model_amplitudes

    das.das._general_delay_and_sum_nearest(weighted_scanlines, tx, rx,
                                                   tx_lookup_times, rx_lookup_times,
                                                   tfm_amplitudes,
                                                   dt, t0, fillvalue, tfm_result)


