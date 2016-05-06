"""
Several helpers related to ultrasonic testing (UT)
"""

import numpy as np

from ..enums import CaptureMethod

__all__ = ['fmc', 'hmc', 'infer_capture_method', 'decibel']


def fmc(numelements):
    """
    Return all pairs of elements for a FMC.
    HMC as performed by Brain.

    Returns
    -------
    tx : ndarray [numelements^2]
        Transmitter for each scanline: 0, 0, ..., 1, 1, ...
    rx : ndarray
        Receiver for each scanline: 1, 2, ..., 1, 2, ...
    """
    numelements = int(numelements)
    elements = np.arange(numelements)

    # 0 0 0    1 1 1    2 2 2
    tx = np.repeat(elements, numelements)

    # 0 1 2    0 1 2    0 1 2
    rx = np.tile(elements, numelements)
    return tx, rx


def hmc(numelements):
    """
    Return all pairs of elements for a HMC.
    HMC as performed by Brain (rx >= tx)

    Returns
    -------
    tx : ndarray [numelements^2]
        Transmitter for each scanline: 0, 0, 0, ..., 1, 1, 1, ...
    rx : ndarray
        Receiver for each scanline: 0, 1, 2, ..., 1, 2, ...
    """
    numelements = int(numelements)
    elements = np.arange(numelements)

    # 0 0 0    1 1    2
    tx = np.repeat(elements, range(numelements, 0, -1))

    # 0 1 2    0 1    2
    rx = np.zeros_like(tx)
    take_n_last = np.arange(numelements, 0, -1)
    start = 0
    for n in take_n_last:
        stop = start + n
        rx[start:stop] = elements[-n:]
        start = stop
    return tx, rx


def infer_capture_method(tx, rx):
    """
    FMC, HMC, or other?
    """
    numelements = max(np.max(tx), np.max(rx)) + 1
    assert len(tx) == len(rx)

    # Get the unique combinations tx/rx of the input.
    # By using set, we ignore the order of the combinations tx/rx.
    combinations = set(zip(tx, rx))

    # Could it be a HMC? Most frequent case, go first.
    # Remark: HMC can be made with tx >= rx or tx <= rx. Check both.
    tx_hmc, rx_hmc = hmc(numelements)
    combinations_hmc1 = set(zip(tx_hmc, rx_hmc))
    combinations_hmc2 = set(zip(rx_hmc, tx_hmc))

    if (len(tx_hmc) == len(tx)) and ((combinations == combinations_hmc1) or (combinations == combinations_hmc2)):
        return CaptureMethod.hmc

    # Could it be a FMC?
    tx_fmc, rx_fmc = fmc(numelements)
    combinations_fmc = set(zip(tx_fmc, rx_fmc))
    if (len(tx_fmc) == len(tx)) and (combinations == combinations_fmc):
        return CaptureMethod.fmc

    # At this point we are hopeless
    return CaptureMethod.unsupported


def decibel(arr, reference=None, neginf_value=-1000., return_reference=False):
    """
    Return 20*log10(abs(arr) / reference)

    If reference is None, use:

        reference := max(abs(arr))

    Parameters
    ----------
    arr : ndarray
        Values to convert in dB.
    reference : float or None
        Reference value for 0 dB. Default: None
    neginf_value : float or None
        If not None, convert -inf dB values to this parameter. If None, -inf
        dB values are not changed.
    return_max : bool
        Default: False.

    Returns
    -------
    arr_db
        Array in decibel.
    arr_max: float
        Return ``max(abs(arr))``. This value is returned only if return_max is true.

    """
    # Disable warnings messages for log10(0.0)
    arr_abs = np.abs(arr)
    if reference is None:
        reference = np.nanmax(arr_abs)
    else:
        assert reference > 0.

    with np.errstate(divide='ignore'):
        arr_db = 20 * np.log10(arr_abs / reference)

    if neginf_value is not None:
        arr_db[np.isneginf(arr_db)] = neginf_value

    if return_reference:
        return arr_db, reference
    else:
        return arr_db


def directivity_finite_width_2d(theta, element_width, wavelength):
    """
    Returns the directivity of an element based on the integration of uniformally radiating sources
    along a straight line in 2D.

    A element is modelled as 'rectangle' of finite width and infinite length out-of-plane.

    This directivity is based only on the element width: each source is assumed to radiate
    uniformally.

    Considering a points1 in the axis Ox in the cartesian basis (O, x, y, z),
    ``theta`` is the inclination angle, ie. the angle in the plane Oxz. Cf. Wooh's paper.

    The directivity is normalised by the its maximum value, obtained for
    theta=0°.

    Returns:

        sinc(pi*a*sin(theta)/lambda)

    where: sinc(x) = sin(x)/x


    Parameters
    ----------
    theta : ndarray
        Angles in radians.
    element_width : float
        In meter.
    wavelength : float
        In meter.

    Returns
    -------
    directivity
        Signed directivity for each angle.

    Notes
    -----

    [1] Wooh, Shi-Chang, and Yijun Shi. 1999. ‘Three-Dimensional Beam Directivity of Phase-Steered Ultrasound’.
    The Journal of the Acoustical Society of America 105 (6): 3275–82. doi:10.1121/1.424655.

    """
    if element_width < 0:
        raise ValueError('Negative width')
    if wavelength < 0:
        raise ValueError('Negative wavelength')

    # /!\ numpy.sinc defines sinc(x) := sin(pi * x)/(pi * x)
    x = element_width * np.sin(theta) / wavelength
    return np.sinc(x)
