__all__ = ['load_scat', 'load_scat_from_matlab']


class InvalidFileFormat(Exception):
    pass


def load_scat(filename, format='auto'):
    """
    Load scattering from any supported source.

    Parameters
    ----------
    filename : str
        Filename
    format : str
        'auto' (default), 'matlab'

    Returns
    -------
    arim.scat.ScatFromData


    """
    formats = ['matlab']

    if format == 'matlab':
        return load_scat_from_matlab(filename)
    elif format == 'auto':
        for format in formats:
            try:
                return load_scat(filename, format=format)
            except InvalidFileFormat:
                pass
        # at this point, everything failed
        raise NotImplementedError('cannot determine the file format')
    else:
        raise ValueError('invalid format')


def load_scat_from_matlab(filename):
    """
    Load scattering from Matlab.

    Parameters
    ----------
    filename

    Returns
    -------
    arim.scat.ScatFromData

    """
    from .. import scat
    import scipy.io as sio

    try:
        data = sio.loadmat(filename)
    except NotImplementedError as e:
        raise InvalidFileFormat() from e

    frequencies = data['frequencies']
    numfreq = frequencies.size
    if frequencies.shape not in {(1, numfreq), (numfreq, 1), (numfreq,)}:
        raise ValueError("invalid shape for 'frequencies'")
    frequencies = frequencies.reshape((numfreq,))

    matrices = dict()

    for scat_key in scat.SCAT_KEYS:
        try:
            matrices[scat_key] = data['scattering_{}'.format(scat_key)]
        except KeyError:
            pass

    return scat.ScatFromData.from_dict(frequencies, matrices)
