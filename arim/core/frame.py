import numpy as np

from .. import utils as u
from .. import settings as s
from ..enums import CaptureMethod

__all__ = ['Frame']


class Frame:
    """
    A frame contains the data received by a probe at a specific location.

    Scanlines are stored a 2D array of length `numscanlines x numsamples`. Each line of the array  is a scanline, i.e.
    the data received by a specific element when a specific element was transmitting.

    Attributes
    ----------
    scanlines : ndarray
        Filtered scanlines.
    scanlines_raw : ndarray
        Unfiltered scanlines (if available).
    time : Time
        Time vector associated to all scanlines.
    tx : ndarray
        1D array of length `numscanlines`. `tx[i]` is the index of the element transmitting during the acquisition of the
        i-th scanline.
    tx : ndarray
        1D array of length `numscanlines`. `tx[i]` is the index of the element receiving during the acquisition of the
        i-th scanline.
    probe : Probe
        Probe used during acquisition.
    examination_object : ExaminationObject
        Object inspected.
    numscanlines : int
        Number of scanlines in the frame.
    metadata : dict
        Metadata


    """

    def __init__(self, scanlines, time, tx, rx, probe, examination_object, scanlines_raw=None, metadata=None):
        """

        Parameters
        ----------
        scanlines
            Scanlines. Assumed these are raw scanlines, unless ``scanlines_raw`` is given.
        time
        tx
        rx
        probe
        examination_object
        scanlines_raw : array or None
            Raw scanlines, if different from ``scanlines``.
        metadata

        Returns
        -------

        """
        # Check shape and dimensions
        try:
            samples = time.samples
        except AttributeError:
            raise TypeError("'time' should be an object 'Time' (current: {}).".format(type(time)))
        numsamples = len(time)

        if scanlines_raw is None:
            scanlines_raw = scanlines

        (numscanlines, _) = u.get_shape_safely(scanlines, 'scanlines', (None, numsamples))
        _ = u.get_shape_safely(tx, 'tx', (numscanlines,))
        _ = u.get_shape_safely(rx, 'rx', (numscanlines,))
        _ = u.get_shape_safely(scanlines_raw, 'scanlines_raw', (numscanlines, numsamples))

        self.scanlines = scanlines
        self.scanlines_raw = scanlines_raw
        self.tx = tx
        self.rx = rx
        self.time = time
        self.probe = probe
        self.examination_object = examination_object

        self.numscanlines = numscanlines
        self.numsamples = numsamples

        if metadata is None:
            metadata = {}
        if metadata.get('capture_method', None) is None:
            metadata['capture_method'] = u.infer_capture_method(tx, rx)
        self.metadata = metadata

    def apply_filter(self, filt):
        """
        Filter the raw scanlines and save them in the frame.

        Warning: the attribute ``scanlines`` is overwritten during this operation. 

        Parameters
        ----------
        filt: Filter

        Returns
        -------
        filtered scanlines

        """
        self.scanlines = filt(self.scanlines_raw)
        assert self.scanlines.shape == self.scanlines_raw.shape
        self.metadata['filter'] = str(filt)
        
        if self.scanlines.dtype.kind == 'c':
            if self.scanlines.dtype != s.COMPLEX:
                self.scanlines = np.squeeze(self.scanlines.astype(s.COMPLEX))
        elif self.scanlines.dtype.kind == 'f':
            if self.scanlines.dtype != s.FLOAT:
                self.scanlines = np.squeeze(self.scanlines.astype(s.FLOAT))
        
        return self.scanlines

    def get_scanline(self, tx, rx, use_raw=False):
        """
        Return the scanline corresponding to the pair (tx, rx).

        Parameters
        ----------
        tx: int
        rx: int
        raw: bool
            Default: False

        Returns
        -------
        scan: 1d array

        """
        if use_raw:
            scanlines = self.scanlines_raw
        else:
            scanlines = self.scanlines
        match = np.logical_and(self.tx == tx,  self.rx == rx)

        return scanlines[match, ...].reshape((self.numsamples, ))
