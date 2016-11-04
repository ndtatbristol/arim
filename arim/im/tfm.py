"""
This module defines classes to perform TFM and TFM-like imaging.
"""

import numpy as np
import warnings

from .. import geometry as g
from .amplitudes import UniformAmplitudes, AmplitudesRemoveExtreme
from .base import delay_and_sum
from .fermat_solver import FermatPath, View, Rays
from .. import settings as s
from ..core import Frame, FocalLaw
from ..enums import CaptureMethod

__all__ = ['BaseTFM', 'ContactTFM', 'MultiviewTFM', 'IMAGING_MODES']

# Order by length then by lexicographic order
# Remark: independant views for one array (i.e. consider that view AB-CD is the
# same as view DC-BA).
IMAGING_MODES = ["L-L", "L-T", "T-T",
                 "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L", "TT-T",
                 "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                 "LT-LT", "LT-TL", "LT-TT",
                 "TL-LT", "TL-TT",
                 "TT-TT"]


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
    interpolate_position (Default is 0 = Nearest)
    fillvalue : float
        Value to assign to scanlines outside ``[tmin, tmax]``. Default: nan
    delay_and_sum_kwargs : dict

    """
    def __init__(self, frame, grid, amplitudes_tx='uniform', amplitudes_rx='uniform',
                 dtype=None, fillvalue=np.nan, geom_probe_to_grid=None,interpolate_position=0):
        self.frame = frame
        self.grid = grid

        if dtype is None:
            dtype = s.FLOAT
        self.dtype = dtype

        self.fillvalue = fillvalue
        self.interpolate_position=interpolate_position

        self.delay_and_sum_kwargs = {}
        self.res = None

        if geom_probe_to_grid is None:
            geom_probe_to_grid = g.GeometryHelper(frame.probe.locations, grid.as_points, frame.probe.pcs)
        else:
            assert geom_probe_to_grid.is_valid(frame.probe, grid.as_points)
        self._geom_probe_to_grid = geom_probe_to_grid

        if amplitudes_tx == 'uniform':
            amplitudes_tx = UniformAmplitudes(frame, grid, fillvalue=fillvalue)
        if amplitudes_rx == 'uniform':
            amplitudes_rx = UniformAmplitudes(frame, grid, fillvalue=fillvalue)
        self.amplitudes_tx = amplitudes_tx
        self.amplitudes_rx = amplitudes_rx

    def run(self):
        """
        Compute TFM: get the lookup times, the amplitudes, and delay and sum
        the scanlines.

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

        delay_and_sum_kwargs = dict(fillvalue=self.fillvalue)
        res = delay_and_sum(self.frame, focal_law,interpolate_position=self.interpolate_position,**delay_and_sum_kwargs)

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


class MultiviewTFM(BaseTFM):
    def __init__(self, frame, grid, view, rays_tx, rays_rx, **kwargs):
        # assert grid is view.tx_path[-1]
        # assert grid is view.rx_path[-1]
        if grid.numpoints != len(view.tx_path[-1]):
            raise ValueError("Inconsistent grid")
        if grid.numpoints != len(view.rx_path[-1]):
            raise ValueError("Inconsistent grid")

        assert view.tx_path == rays_tx.path
        assert view.rx_path == rays_rx.path
        assert isinstance(rays_tx, Rays)
        assert isinstance(rays_rx, Rays)
        #assert rays_rx.indices.flags.fortran
        #assert rays_tx.indices.flags.fortran
        #assert rays_tx.times.flags.fortran
        #assert rays_rx.times.flags.fortran
        assert rays_tx.path[0] is frame.probe.locations
        assert rays_rx.path[0] is frame.probe.locations
        assert rays_tx.path[-1] is grid.as_points
        assert rays_rx.path[-1] is grid.as_points
        self.rays_tx = rays_tx
        self.rays_rx = rays_rx
        self.view = view

        # used in get_amplitudes
        self.fillvalue_extreme_points = np.nan

        amplitudes_tx = kwargs.get('amplitudes_tx')
        if amplitudes_tx is None:
            amplitudes_tx = AmplitudesRemoveExtreme(frame, grid, rays_tx)
        kwargs['amplitudes_tx'] = amplitudes_tx

        amplitudes_rx = kwargs.get('amplitudes_rx')
        if amplitudes_rx is None:
            amplitudes_rx = AmplitudesRemoveExtreme(frame, grid, rays_rx)
        kwargs['amplitudes_rx'] = amplitudes_rx

        super().__init__(frame, grid, **kwargs)

    def get_lookup_times_tx(self):
        """Lookup times in transmission, obtained with Fermat solver."""
        return np.ascontiguousarray(self.rays_tx.times.T)

    def get_lookup_times_rx(self):
        """Lookup times in reception, obtained with Fermat solver."""
        return np.ascontiguousarray(self.rays_rx.times.T)


    def __repr__(self):
        return "<{}: {} at {}>".format(
            self.__class__.__name__,
            str(self.view),
            hex(id(self)))

    @classmethod
    def make_views(cls, probe, frontwall, backwall, grid, v_couplant, v_longi, v_shear):
        """
        Create direct-direct, skip-direct and skip-skip views.

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
        warnings.warn(PendingDeprecationWarning("Using 'arim.Path' objects is now recommended. This method will be removed in future versions."))
        views = []
        parse = lambda name: cls._parse_name_view(name, backwall, v_longi, v_shear)
        for name in IMAGING_MODES:
            tx_name, rx_name = name.split('-')
            tx_path = FermatPath((probe, v_couplant, frontwall) + parse(tx_name) + (grid,))
            rx_path = FermatPath((probe, v_couplant, frontwall) + parse(rx_name[::-1]) + (grid,))
            views.append(View(tx_path, rx_path, name))
        return views

    @staticmethod
    def _parse_name_view(name, backwall, v_longi, v_shear):
        name = name.upper()
        if name == 'L':
            return (v_longi,)
        elif name == 'T':
            return (v_shear,)
        elif name == 'LL':
            return (v_longi, backwall, v_longi)
        elif name == 'LT':
            return (v_longi, backwall, v_shear)
        elif name == 'TL':
            return (v_shear, backwall, v_longi)
        elif name == 'TT':
            return (v_shear, backwall, v_shear)
        else:
            raise ValueError("Cannot parse view '{}'".format(name))
