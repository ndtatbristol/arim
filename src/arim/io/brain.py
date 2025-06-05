"""
I/O module for BRAIN files (Matlab NDT library of Universiy of Bristol).


Implemented as of 20/6/2016:
- dtype of variables is according to settings.py

- get element dimensions from el_x1, el_y1, el_z1, el_x2, el_y2, el_z2:
  Information calculated is probe orientation dependent.


"""

import numpy as np

from .. import geometry as g
from .. import settings as s
from ..core import ExaminationObject, Frame, Material, Probe, Time

__all__ = ["load_expdata"]


class NotHandledByScipy(Exception):
    pass


class InvalidExpData(IOError):
    pass


def _import_h5py():
    try:
        import h5py
    except ImportError:
        h5py = None
    return h5py


def load_expdata(file):
    """
    Load exp_data file.

    Parameters
    ----------
    file: str or file object

    Returns
    -------
    arim.core.Frame

    Raises
    ------
    InvalidExpData, OSError (HDF5 fail)

    """
    try:
        (exp_data, array, filename) = _load_from_scipy(file)
    except NotHandledByScipy:
        # It seems the file is HDF5 (matlab 7.3)
        h5py = _import_h5py()
        if h5py is None:
            raise Exception(
                "Unable to import Matlab file because its file format version is unsupported. "
                "Try importing the file in Matlab and exporting it with the "
                "command 'save' and the flag '-v7'. Alternatively, try to install the Python library 'h5py'."
            )
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

    frame.metadata["from_brain"] = filename
    frame.probe.metadata["from_brain"] = filename
    frame.examination_object.metadata["from_brain"] = filename
    return frame


def _load_probe(array):
    """
    Parameters
    ----------
    array : dict
        dict-like object corresponding to Matlab struct exp_data.array.

    Returns
    -------
    Probe

    """
    frequency = array["centre_freq"][0, 0]

    # dtype = np.result_type(array['el_xc'], array['el_yc'], array['el_zc'])
    dtype = s.FLOAT

    # Get locations
    locations_x = np.squeeze(array["el_xc"]).astype(dtype)
    locations_y = np.squeeze(array["el_yc"]).astype(dtype)
    locations_z = np.squeeze(array["el_zc"]).astype(dtype)

    locations = g.Points.from_xyz(locations_x, locations_y, locations_z)

    # Calculate Probe Dimensions (using el_x1, el_x2 and el_xc etc for each dimension)
    dimensions_x = 2 * np.maximum(
        np.absolute(np.squeeze(array["el_x1"]).astype(dtype) - locations_x),
        np.absolute(np.squeeze(array["el_x2"]).astype(dtype) - locations_x),
    )
    dimensions_y = 2 * np.maximum(
        np.absolute(np.squeeze(array["el_y1"]).astype(dtype) - locations_y),
        np.absolute(np.squeeze(array["el_y2"]).astype(dtype) - locations_y),
    )
    dimensions_z = 2 * np.maximum(
        np.absolute(np.squeeze(array["el_z1"]).astype(dtype) - locations_z),
        np.absolute(np.squeeze(array["el_z2"]).astype(dtype) - locations_z),
    )
    dimensions = g.Points.from_xyz(dimensions_x, dimensions_y, dimensions_z)

    return Probe(locations, frequency, dimensions=dimensions)


def _load_frame(exp_data, probe):
    # NB: Matlab is 1-indexed, Python is 0-indexed
    tx = np.squeeze(exp_data["tx"])
    rx = np.squeeze(exp_data["rx"])
    tx = tx.astype(s.UINT) - 1
    rx = rx.astype(s.UINT) - 1
    # Remark: [...] is required to read in the case of HDF5 file
    # (and does nothing if we have a regular array
    timetraces = np.squeeze(exp_data["time_data"][...])
    timetraces = timetraces.astype(s.FLOAT)
    # exp_data.time_data is such as a two consecutive time samples are stored contiguously, which
    # is what we want. However Matlab saves either in Fortran order (shape: numtimetraces x numsamples)
    # or C order (shape: numsamples x numtimetraces). We force using the later case.
    if timetraces.flags.f_contiguous:
        timetraces = timetraces.T

    timevect = np.squeeze(exp_data["time"])
    timevect = timevect.astype(s.FLOAT)
    time = Time.from_vect(timevect)
    try:
        velocity = np.squeeze(exp_data["material"]["vel_spherical_harmonic_coeffs"])
        if isinstance(velocity[()], np.ndarray):
            # Accept nested array
            velocity = velocity[()]
    # Old version of brain saves phase velocity, new version has it saved in material.
    except (ValueError, KeyError):
        velocity = np.squeeze(exp_data["ph_velocity"])

    # Sometimes have location saved in `exp_data`. Useful to have access to this info.
    metadata = None
    try:
        location = {
            k: float(exp_data["location"][k][0, 0])
            for k in exp_data["location"].dtype.names
        }
        metadata = {"location": location}
    except AttributeError:
        location = {k: float(v[0, 0]) for k, v in exp_data["location"].items()}
        metadata = {"location": location}
    except (ValueError, KeyError):
        pass

    velocity = velocity.astype(s.FLOAT)
    material = Material(velocity)
    examination_object = ExaminationObject(material)

    return Frame(timetraces, time, tx, rx, probe, examination_object, metadata)


def _load_from_scipy(file):
    """
    Parameters
    ----------
    file : str or obj
        Path-like string to a file to be loaded, or a file object.

    Returns
    -------
        exp_data : dict
            Dict-like object containing experimental details (tx, rx, time_data, etc.)
        array : dict
            Dict-like object containing array details (locations, frequency, etc.)
        filename : str
            If `file` is a file object, return the filename.

    Raises
    ------
    NotHandledByScipy
        If file cannot be loaded by scipy (i.e. '-v7.3' tag used in MATLAB.)

    """
    import scipy.io as sio

    try:
        data = sio.loadmat(file)
    except NotImplementedError as e:
        raise NotHandledByScipy(e)

    # Get data:
    try:
        exp_data = data["exp_data"][0, 0]
        array = exp_data["array"][0, 0]
    except IndexError as e:
        raise InvalidExpData(e) from e

    # Get filename (works whether 'file' is a file object or a (str) filename)
    try:
        filename = file.name
    except AttributeError:
        filename = str(file)

    return exp_data, array, filename


def _load_from_hdf5(file):
    import h5py

    # This line might raise an OSError:
    f = h5py.File(file, mode="r")

    try:
        # File successfully loaded by HDF5:
        exp_data = f["exp_data"]
        array = exp_data["array"]
    except IndexError as e:
        raise InvalidExpData(e) from e

    filename = f.filename

    return exp_data, array, filename
