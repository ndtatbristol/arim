"""
This module defines classes to perform TFM and TFM-like imaging.
"""

import numpy as np
import warnings
from collections import namedtuple

from .. import geometry as g
from .amplitudes import UniformAmplitudes, AmplitudesRemoveExtreme
from .fermat_solver import FermatPath, Rays
from .. import settings as s
from .. import core
from ..core import Frame, FocalLaw, View
from ..enums import CaptureMethod
from ..path import IMAGING_MODES  # import for backward compatibility
import numba

__all__ = ['BaseTFM', 'ContactTFM', 'SingleViewTFM', 'IMAGING_MODES']

ExtramaLookupTimes = namedtuple('ExtramaLookupTimes', 'tmin tmax tx_elt_for_tmin rx_elt_for_tmin tx_elt_for_tmax rx_elt_for_tmax')


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
    geom_probe_to_grid : GeometryHelper

    Attributes
    ----------
    result
    frame
    grid
    dtype
    amplitudes : Amplitudes
    geom_probe_to_grid

    """

    def __init__(self, frame, grid, amplitudes_tx='uniform', amplitudes_rx='uniform',
                 dtype=None, geom_probe_to_grid=None):
        self.frame = frame
        self.grid = grid

        if dtype is None:
            dtype = s.FLOAT
        self.dtype = dtype

        self.res = None

        if geom_probe_to_grid is None:
            geom_probe_to_grid = g.GeometryHelper(frame.probe.locations, grid.as_points, frame.probe.pcs)
        else:
            assert geom_probe_to_grid.is_valid(frame.probe, grid.as_points)
        self._geom_probe_to_grid = geom_probe_to_grid

        if amplitudes_tx == 'uniform':
            amplitudes_tx = UniformAmplitudes(frame, grid)
        if amplitudes_rx == 'uniform':
            amplitudes_rx = UniformAmplitudes(frame, grid)
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

        focal_law = FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes_tx, amplitudes_rx, scanline_weights)

        focal_law = self.hook_focal_law(focal_law)
        self.focal_law = focal_law

        if delay_and_sum_func is None:
            from .das import delay_and_sum
            delay_and_sum_func = delay_and_sum
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
        Standard scanline weights. Handle FMC and HMC.

        For FMC: weights 1.0 for all scanlines.
        For HMC: weights 2.0 for scanlines where TX and RX elements are different, 1.0 otherwise.

        """
        capture_method = self.frame.metadata.get('capture_method', None)
        if capture_method is None:
            raise NotImplementedError
        elif capture_method is CaptureMethod.fmc:
            weights = np.ones(self.frame.numscanlines, dtype=self.dtype)
            return weights
        elif capture_method is CaptureMethod.hmc:
            weights = np.full(self.frame.numscanlines, 2.0, dtype=self.dtype)
            same_tx_rx = self.frame.tx == self.frame.rx
            weights[same_tx_rx] = 1.0
            return weights
        else:
            raise NotImplementedError

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
        return np.nanmax(np.abs(self.res[area_of_interest]))

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
        # assert grid is view.tx_path[-1]
        # assert grid is view.rx_path[-1]
        if grid.numpoints != len(view.tx_path.interfaces[-1].points):
            raise ValueError("Inconsistent grid")
        if grid.numpoints != len(view.rx_path.interfaces[-1].points):
            raise ValueError("Inconsistent grid")

        tx_rays = view.tx_path.rays
        rx_rays = view.rx_path.rays

        # assert rays_rx.indices.flags.fortran
        # assert rays_tx.indices.flags.fortran
        # assert rays_tx.times.flags.fortran
        # assert rays_rx.times.flags.fortran
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
            amplitudes_tx = AmplitudesRemoveExtreme(frame, grid, tx_rays)
        kwargs['amplitudes_tx'] = amplitudes_tx

        amplitudes_rx = kwargs.get('amplitudes_rx')
        if amplitudes_rx is None:
            amplitudes_rx = AmplitudesRemoveExtreme(frame, grid, rx_rays)
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

    @classmethod
    def make_views(cls, probe, frontwall, backwall, grid, v_couplant, v_longi, v_shear):
        """
        Create direct-direct, skip-direct and skip-skip views for a block an immersion.

        This method is here for backward-compatibility only. It is recommended to use the more generic
        ``arim.make_view_for_block_in_immersion`` instead.

        Parameters
        ----------
        probe : Points
        frontwall : Points
        backwall : Points
        grid : Points
        v_couplant : float
        v_longi : float
        v_shear : float

        Returns
        -------
        views : List[View]
        """
        from .. import paths_for_block_in_immersion, views_for_block_in_immersion

        warnings.warn(DeprecationWarning(
            "Using arim.make_view_for_block_in_immersion is recommended. This method will be removed in future versions."))

        # Create dummy interfaces:
        probe_interface = core.Interface(probe, None)
        frontwall_interface = core.Interface(frontwall, None)
        backwall_interface = core.Interface(backwall, None)
        grid_interface = core.Interface(grid, None)

        # Create Dummy material:
        block = core.Material(v_longi, v_shear, state_of_matter='solid')
        couplant = core.Material(v_couplant, state_of_matter='liquid')

        paths_dict = paths_for_block_in_immersion(block, couplant, probe_interface, frontwall_interface,
                                                  backwall_interface, grid_interface)
        views = views_for_block_in_immersion(paths_dict)

        return views


@numba.jit(nopython=True)
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
