"""
I/O module for BRAIN files (Matlab NDT library of Universiy of Bristol).


Implemented as of 20/6/2016:
- dtype of variables is according to settings.py

- get element dimensions from el_x1, el_y1, el_z1, el_x2, el_y2, el_z2: 
  Information calculated is probe orientation dependent.


"""

import numpy as np
from scipy.io import loadmat

from .. import geometry as g
from .. import settings as s
from ..core import Probe, Frame, Time, InfiniteMedium, Material

__all__ = ['NotHandledByScipy', 'InvalidExpData', 'load_expdata']

try:
    import h5py
except ImportError:
    h5py = None


class NotHandledByScipy(Exception):
    pass


class InvalidExpData(IOError):
    pass


def load_expdata(file):
    """
    Load exp_data file.

    Parameters
    ----------
    file: str or file object

    Returns
    -------
    Frame

    Raises
    ------
    InvalidExpData, OSError (HDF5 fail)
    """
    try:
        (exp_data, array, filename) = _load_from_scipy(file)
    except NotHandledByScipy:
        # It seems the file is HDF5 (matlab 7.3)
        if h5py is None:
            raise Exception("Unable to import Matlab file because its file format version is unsupported. "
                            "Try importing the file in Matlab and exporting it with the "
                            "command 'save' and the flag '-v7'. Alternatively, try to install the Python library 'h5py'.")
        (exp_data, array, filename) = _load_from_hdf5(file)

    # As this point exp_data and array are populated either by scipy.io or hdf5:
    try:
        probe = _load_probe(array)
    except Exception as e:
        raise InvalidExpData(e) from e

    try:
        frame = _load_frame(exp_data, probe)
    except Exception as e:
        raise InvalidExpData(e) from e

    frame.metadata['from_brain'] = filename
    frame.probe.metadata['from_brain'] = filename
    frame.examination_object.metadata['from_brain'] = filename
    return frame


def _load_probe(array):
    """
    :param array: dict-like object corresponding to Matlab struct exp_data.array.
    :return: Probe
    """
    frequency = array['centre_freq'][0, 0]

    #dtype = np.result_type(array['el_xc'], array['el_yc'], array['el_zc'])
    dtype=s.FLOAT
    
    # Get locations
    locations_x = np.squeeze(array['el_xc']).astype(dtype)
    locations_y = np.squeeze(array['el_yc']).astype(dtype)
    locations_z = np.squeeze(array['el_zc']).astype(dtype)

    locations = g.Points(locations_x, locations_y, locations_z)
    
    #Calculate Probe Dimensions (using el_x1, el_x2 and el_xc etc for each dimension)
    dimensions_x = 2*np.maximum(np.absolute(np.squeeze(array['el_x1']).astype(dtype) - locations_x),np.absolute(np.squeeze(array['el_x2']).astype(dtype) - locations_x))
    dimensions_y = 2*np.maximum(np.absolute(np.squeeze(array['el_y1']).astype(dtype) - locations_y),np.absolute(np.squeeze(array['el_y2']).astype(dtype) - locations_y))
    dimensions_z = 2*np.maximum(np.absolute(np.squeeze(array['el_z1']).astype(dtype) - locations_z),np.absolute(np.squeeze(array['el_z2']).astype(dtype) - locations_z)) 
    dimensions = g.Points(dimensions_x, dimensions_y, dimensions_z)
    
    return Probe(locations, frequency,dimensions=dimensions)


def _load_frame(exp_data, probe):
    # NB: Matlab is 1-indexed, Python is 0-indexed
    tx = np.squeeze(exp_data['tx'].astype(s.UINT)) - 1
    rx = np.squeeze(exp_data['rx'].astype(s.UINT)) - 1

    # Remark: [...] is required to read in the case of HDF5 file
    # (and does nothing if we have a regular array
    scanlines = np.squeeze(exp_data['time_data'][...].astype(s.FLOAT))

    # exp_data.time_data is such as a two consecutive time samples are stored contiguously, which
    # is what we want. However Matlab saves either in Fortran order (shape: numscanlines x numsamples)
    # or C order (shape: numsamples x numscanlines). We force using the later case.
    if scanlines.flags.f_contiguous:
        scanlines = scanlines.T

    timevect = np.squeeze(exp_data['time'].astype(s.FLOAT))
    time = Time.from_vect(timevect)

    velocity = np.squeeze(exp_data['ph_velocity'].astype(s.FLOAT))
    material = Material(velocity)
    examination_object = InfiniteMedium(material)

    return Frame(scanlines, time, tx, rx, probe, examination_object)


def _load_from_scipy(file):
    """

    :param file:
    :return:
    :raises: NotHandledByScipy
    """
    try:
        data = loadmat(file)
    except NotImplementedError as e:
        raise NotHandledByScipy(e)

    # Get data:
    try:
        exp_data = data['exp_data'][0, 0]
        array = exp_data['array'][0, 0]
    except IndexError as e:
        raise InvalidExpData(e) from e

    # Get filename (works whether 'file' is a file object or a (str) filename)
    try:
        filename = file.name
    except AttributeError:
        filename = str(file)

    return exp_data, array, filename


def _load_from_hdf5(file):
    # This line might raise an OSError:
    f = h5py.File(file, mode='r')

    try:
        # File successfully loaded by HDF5:
        exp_data = f['exp_data']
        array = exp_data['array']
    except IndexError as e:
        raise InvalidExpData(e) from e

    filename = f.filename

    return exp_data, array, filename

