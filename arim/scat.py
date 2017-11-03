"""
Scattering functions
"""
import math
import warnings
import functools
from functools import partial
import abc

import numpy as np
from numpy.core.umath import pi, cos, sin
from scipy.special._ufuncs import hankel1, hankel2
from scipy import interpolate

from . import _scat, exceptions

SCAT_KEYS = frozenset(('LL', 'LT', 'TL', 'TT'))


def make_angles(numpoints):
    """Return angles for scattering matrices. Linearly spaced vector in [-pi, pi[."""
    return np.linspace(-np.pi, np.pi, numpoints, endpoint=False)


def make_angles_grid(numpoints):
    """Return angles for scattering matrices as a grid of incident and outgoing angles.
    """
    theta = make_angles(numpoints)
    inc_theta, out_theta = np.meshgrid(theta, theta, indexing='xy')
    return inc_theta, out_theta


def func_to_matrix(scattering_func, numpoints):
    """
    Call the function 'scattering_func' for all incident and scattered angles.
    Returns the result as a matrix.

    Parameters
    ----------
    scattering_func :
        Function that returns the amplitudes of the scattered wave.
         Its first argument must be the angle of the incident wave, its second
         argument must be the angle of the scattered wave.
    numpoints : int
        Number of points to use.

    Returns
    -------
    scattering_matrix : ndarray
        Shape (numpoints, numpoints)
        scattering_matrix[i, j] is the coefficient for the incident angle theta[j] and
        the scattering angle theta[i].
    """
    inc_theta, out_theta = make_angles_grid(numpoints)
    scattering_matrix = scattering_func(inc_theta, out_theta)
    return scattering_matrix


def interpolate_matrix(scattering_matrix):
    """
    Returns a function that takes as input the incident angles and the scattering angles.
    This returned function returns the scattering amplitudes, obtained by bilinear
    interpolation of the scattering matrix.

    Parameters
    ----------
    scattering_matrix

    Returns
    -------
    func

    """
    scattering_matrix = np.asarray(scattering_matrix)
    if scattering_matrix.ndim != 2:
        raise ValueError('scattering matrix must have a shape (n, n)')
    if scattering_matrix.shape[0] != scattering_matrix.shape[1]:
        raise ValueError('scattering matrix must have a shape (n, n)')

    return partial(_scat._interpolate_scattering_matrix_ufunc, scattering_matrix)


def interpolate_matrices(scattering_matrices):
    """
    Convert a dictionary containing scattering matrices to a dictionary containing
    functions that interpolate the values of the scattering matrices.

    Parameters
    ----------
    scattering_matrices : dict[str, ndarray]

    Returns
    -------
    dict[str, function]
    """
    return {key: interpolate_matrix(mat) for key, mat
            in scattering_matrices.items()}


def multi_to_single_freq_scat_matrices(multi_freq_scat_matrices, new_freq,
                                       **interp1d_kwargs):
    """
    Return the single-frequency scattering matrices from multi-frequency scattering
    matrices by interpolating at the desired frequency.

    Parameters
    ----------
    multi_freq_scat_matrices : dict[str]
        Keys: frequencies (1d array), LL, LT, TL, TT
    new_freq : float
        New frequency where to interpolate
    interp1d_kwargs : kwargs
        Arguments for `scipy.interpolate.interp1d`

    Returns
    -------
    single_freq_scat_matrices

    """
    frequencies = multi_freq_scat_matrices['frequencies']
    out = {}

    interpolation_is_needed = len(frequencies) > 1
    if not interpolation_is_needed:
        # only one frequency, return the only scattering matrices
        if new_freq != frequencies[0]:
            warnings.warn("'new_freq' is unused because no interpolation is needed",
                          exceptions.ArimWarning)
        if len(interp1d_kwargs) > 0:
            warnings.warn("interp1d arguments are unused because no interpolation "
                          "is needed", exceptions.ArimWarning)

    for key in SCAT_KEYS:
        try:
            matrix = multi_freq_scat_matrices[key]
        except KeyError:
            continue
        else:
            if interpolation_is_needed:
                out[key] = interpolate.interp1d(frequencies, matrix, axis=0,
                                                **interp1d_kwargs)(new_freq)
            else:
                out[key] = matrix[0]
    return out


def sdh_2d_scat(inc_theta, out_theta, frequency, radius, longitudinal_vel,
                transverse_vel, min_terms=10, term_factor=4, to_compute=SCAT_KEYS):
    """
    Scattering coefficients for a side-drilled hole in 2D

    The scattered field is given by::

        u_scat(r, theta) = u0 * sqrt(1 / r) * exp(-i k r + i omega i ray) *
                           (sqrt(lambda_L) A(theta) e_r +
                            sqrt(lambda_T) B(theta) e_theta)

    where A(theta) and B(theta) are the scattering coefficients for respectively L and
    T scattered waves and where e_r and e_theta are the two vectors of the cylindrical
    coordinate system.

    Keys for the coefficients: 'LL', 'LT', 'TL', 'TT'. The first letter refers to the kind
    of the incident wave, the second letter refers to the kind of the scattered wave.

    The coefficient for LL, LT, TL and TT are obtained from Lopez-Sanchez's paper,
    equations 33, 34, 39, 40. See also Brind's paper. Compared to these papers, the
    complex conjugate coefficients are returned because these papers use the
    convention ``u(ray) = u(omega) exp(-i omega ray)`` whereas we use the convention
    ``u(ray) = u(omega) exp(+i omega ray)``.

    Another difference with these papers is the definition of theta. We use the NDT
    convention where pulse-echo corresponds to theta=0. For Brind, Lopez-Sanchez et al.
    pulse-echo corresponds to theta=pi.

    The number of factor in the sum is::

        maxn = max(min_terms, ceil(term_factor * alpha), ceil(term_factor * beta))


    Parameters
    ----------
    inc_theta : ndarray
        Angle in radians. Pulse echo case corresponds to inc_theta = out_theta
    out_theta : ndarray
        Angle in radians.
    frequency : float
    radius : float
    longitudinal_vel : float
    transverse_vel : float
    min_terms : int
    term_factor : int
    to_compute : set
        Coefficients to compute. Default: compute all.

    Returns
    -------
    result : dict
        Keys corresponds to 'to_compute' argument. Values have the shape of theta.

    References
    ----------
    [Lopez-Sanchez] Lopez-Sanchez, Ana L., Hak-Joon Kim, Lester W. Schmerr, and Alexander
    Sedov. 2005. ‘Measurement Models and Scattering Models for Predicting the Ultrasonic
    Pulse-Echo Response From Side-Drilled Holes’. Journal of Nondestructive Evaluation 24
    3): 83–96. doi:10.1007/s10921-005-7658-4.

    [Brind] Brind, R. J., J. D. Achenbach, and J. E. Gubernatis. 1984. ‘High-Frequency
    Scattering of Elastic Waves from Cylindrical Cavities’. Wave Motion 6 (1):
    41–60. doi:10.1016/0165-2125(84)90022-2.

    [Zhang] Zhang, Jie, B.W. Drinkwater, and P.D. Wilcox. 2008. ‘Defect Characterization
    Using an Ultrasonic Array to Measure the Scattering Coefficient Matrix’. IEEE
    Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 55 (10): 2254–65.
    doi:10.1109/TUFFC.924.


    """
    theta = out_theta - inc_theta

    if not SCAT_KEYS.issuperset(to_compute):
        raise ValueError("Valid 'to_compute' arguments are {} (got {})".format(SCAT_KEYS,
                                                                               to_compute))

    # wavenumber

    kl = 2 * pi * frequency / longitudinal_vel
    kt = 2 * pi * frequency / transverse_vel

    # Brind eq 2.8
    alpha = kl * radius
    beta = kt * radius
    beta2 = beta * beta

    # sum from n=0 to n=maxn (inclusive)
    # The larger maxn, the better the axppromixation
    maxn = max([int(min_terms),
                math.ceil(term_factor * alpha), math.ceil(term_factor * beta)])
    n = np.arange(0, maxn + 1)
    n2 = n * n

    # Brind eq 2.8
    epsilon = np.full(n.shape, 2.)
    epsilon[0] = 1.

    # Definition of C_n^(i)(x) and D_n^(i)(x)
    # Brind, eq 31
    c1 = lambda x: (n2 + n - beta2 / 2) * hankel1(n, x) - x * hankel1(n - 1, x)
    c2 = lambda x: (n2 + n - beta2 / 2) * hankel2(n, x) - x * hankel2(n - 1, x)
    d1 = lambda x: (n2 + n) * hankel1(n, x) - n * x * hankel1(n - 1, x)
    d2 = lambda x: (n2 + n) * hankel2(n, x) - n * x * hankel2(n - 1, x)
    c1_alpha = c1(alpha)
    c2_alpha = c2(alpha)
    d1_alpha = d1(alpha)
    d2_alpha = d2(alpha)
    c1_beta = c1(beta)
    c2_beta = c2(beta)
    d1_beta = d1(beta)
    d2_beta = d2(beta)

    # in angle
    phi = theta + pi

    # n_phi[i1, ..., id, j] := phi[i1, ..., id] * n[j]
    n_phi = np.einsum('...,j->...j', phi, n)
    cos_n_phi = cos(n_phi)
    sin_n_phi = sin(n_phi)
    del n_phi

    result = dict()

    # NB: sqrt(2j/(pi * k)) = sqrt(i) / pi

    if 'LL' in to_compute:
        # Lopez-Sanchez eq (29)
        A_n = 1j / (2 * alpha) * (
            1 + (c2_alpha * c1_beta - d2_alpha * d1_beta) /
            (c1_alpha * c1_beta - d1_alpha * d1_beta))

        # Brind (2.9) without:
        #   - u0, the amplitude of the incident wave,
        #   - 'exp(i k r)'  which in Bristol LTI model is in the propagation term,
        #   - 'lambda/sqrt(r)' which in Bristol LTI model is the 2D beamspread term,
        #
        # This is consistent with Lopez-Sanchez eq (33).
        #
        # NB: exp(i pi /4) = sqrt(i)
        #
        # The line:
        #   out = np.einsum('...j,j->...', n_phi, coeff)
        # gives the result:
        #   out[i1, ..., id] = sum_j (n_phi[i1, ..., id, j] * coeff[j])
        r = (np.sqrt(1j) / pi * alpha) * \
            np.einsum('...j,j->...', cos_n_phi, epsilon * A_n)

        result['LL'] = r.conjugate()

    if 'LT' in to_compute:
        # Lopez-Sanchez eq (30)
        B_n = 2 * n / (pi * alpha) * ((n2 - beta2 / 2 - 1) /
                                      (c1_alpha * c1_beta - d1_alpha * d1_beta))

        # Lopez-Sanchez (34)
        # Warning: there is a minus sign in Brind (2.10). We trust LS here.
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * beta) * \
            np.einsum('...j,j->...', sin_n_phi, epsilon * B_n)
        result['LT'] = r.conjugate()

    if 'TL' in to_compute:
        # Lopez-Sanchez eq (41)
        A_n = 2 * n / (pi * beta) * (n2 - beta2 / 2 - 1) / (
            c1_alpha * c1_beta - d1_alpha * d1_beta)

        # Lopez-Sanchez eq (39)
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * alpha) * \
            np.einsum('...j,j->...', sin_n_phi, epsilon * A_n)
        result['TL'] = r.conjugate()

    if 'TT' in to_compute:
        # Lopez-Sanchez eq (42)
        B_n = 1j / (2 * beta) * (1 + (c2_beta * c1_alpha - d2_beta * d1_alpha) /
                                 (c1_alpha * c1_beta - d1_alpha * d1_beta))

        # Lopez-Sanchez eq (40)
        # See also comments for result['LL']
        r = (np.sqrt(1j) / pi * beta) * \
            np.einsum('...j,j->...', cos_n_phi, epsilon * B_n)
        result['TT'] = r.conjugate()

    return result


def _scat_2d_cylinder(inc_theta, out_theta, scat_key, **scat_params):
    return sdh_2d_scat(inc_theta, out_theta, to_compute={scat_key},
                       **scat_params)[scat_key]


def scat_2d_cylinder_funcs(radius, longitudinal_vel,
                           transverse_vel, **kwargs):
    """
    Returns scattering functions for side-drilled holes.

    Cf. :func:`sdh_2d_scat`

    Parameters
    ----------
    radius
    longitudinal_vel
    transverse_vel
    kwargs

    Returns
    -------
    scat_funcs: dict of func
        Usage: scat_funcs['LT'](inc_theta, out_theta)

    """
    scat_funcs = {}
    scat_params = dict(
        radius=radius,
        longitudinal_vel=longitudinal_vel,
        transverse_vel=transverse_vel,
        **kwargs
    )

    for scat_key in ['LL', 'LT', 'TL', 'TT']:
        scat_funcs[scat_key] = partial(_scat_2d_cylinder, scat_key=scat_key,
                                       **scat_params)
    return scat_funcs


def scat_2d_cylinder_matrices(numpoints, radius, longitudinal_vel,
                              transverse_vel,
                              to_compute={'LL', 'LT', 'TL', 'TT'}, **kwargs):
    """
    Returns scattering matrices for side-drilled holes.

    Cf. :func:`sdh_2d_scat`.

    Parameters
    ----------
    numpoints : int
        Number of points for discretising [-pi, pi[.
    radius
    longitudinal_vel
    transverse_vel
    min_terms
    term_factor
    to_compute

    Returns
    -------
    matrices : dict
    """
    matrices = {}
    scat_funcs = scat_2d_cylinder_funcs(radius, longitudinal_vel,
                                        transverse_vel, **kwargs)
    for scat_key in to_compute:
        matrices[scat_key] = func_to_matrix(scat_funcs[scat_key], numpoints)
    return matrices


def scat_point_source_funcs(longitudinal_vel, transverse_vel):
    """
    Returns scattering functions for (unphysical) point source. For debug only.

    Parameters
    ----------
    longitudinal_vel : float
    transverse_vel : float

    Returns
    -------
    dict
    """
    vl = longitudinal_vel
    vt = transverse_vel
    return {
        'LL': lambda inc, out: np.full_like(inc, 1.),
        'LT': lambda inc, out: np.full_like(inc, vl / vt),
        'TL': lambda inc, out: -np.full_like(inc, vt / vl),
        'TT': lambda inc, out: np.full_like(inc, 1.),
    }


def rotate_matrix(scat_matrix, phi):
    """
    Return the scattering matrix S' of the scatterer rotated by an angle phi,
    knowing the scattering matrix S of the unrotated scatterer.

        S'(theta_1, theta_2) = S(theta_1 - phi, theta_2 - phi)

    Use FFT internally.

    Parameters
    ----------
    scat_matrix : ndarray
        Shape (numangles, numangles)
    phi : float
        Defect's rotation angle in radian.

    Returns
    ------
    roated_scat_matrix: ndarray
        Shape : (numangles, numangles)

    """
    n, _ = scat_matrix.shape

    freq = np.fft.fftfreq(n, 2 * np.pi / n)

    freq_x, freq_y = np.meshgrid(freq, freq, indexing='ij')

    freqshift = np.exp(-2j * np.pi * (freq_x + freq_y) * phi)
    scat_matrix_f = np.fft.fft2(scat_matrix)
    return np.fft.ifft2(freqshift * scat_matrix_f)


def _partial_one_scat_key(scat_func, scat_key, *args, **kwargs):
    # Remark: do not try to replace this by a lambda function, a proper closure is needed
    # here.
    # See https://stackoverflow.com/questions/3252228/python-why-is-functools-partial-necessary
    to_compute = {scat_key}
    return functools.partial(scat_func, *args, to_compute=to_compute, **kwargs)[scat_key]


def scattering_factory(kind, **kwargs):
    pass


class Scattering(abc.ABC):
    @abc.abstractmethod
    def __call__(self, inc_theta, out_theta, frequency, to_compute=SCAT_KEYS):
        """
        Returns the scattering values for the given angles and frequency.

        Parameters
        ----------
        inc_theta : ndarray
        out_theta : ndarray
        frequency : float
        to_compute : set[str]

        Returns
        -------
        scat_values : dict[ndarray]
            Keys: at least the ones given in `to_compute`.

        """

    def as_multi_freq_funcs(self):
        """
        Returns a dict of scattering functions that take as input the incident angle,
        the outgoing angle and the frequency.
        """
        scat_funcs = {}

        for scat_key in SCAT_KEYS:
            scat_funcs[scat_key] = _partial_one_scat_key(self, scat_key)
        return scat_funcs

    def as_single_freq_funcs(self, frequency):
        """
        Returns a dict of scattering functions that take as input the incident angle
        and the outgoing angle.
        """
        scat_funcs = {}

        for scat_key in SCAT_KEYS:
            scat_funcs[scat_key] = _partial_one_scat_key(self, scat_key,
                                                         frequency=frequency)
        return scat_funcs

    def as_multi_freq_matrices(self, frequencies, numangles, to_compute=SCAT_KEYS):
        """
        Returns scattering matrices at different frequencies.

        Parameters
        ----------
        frequency : ndarray
            Shape: (numfreq, )
        numangles : int
        to_compute

        Returns
        -------
        dict[str, ndarray]
            Shape of each matrix: ``(numfreq, numpoints, numpoints)``

        """
        inc_theta, out_theta = make_angles_grid(numangles)

        out = None

        for i, frequency in enumerate(frequencies):
            matrices = self(inc_theta, out_theta, frequency, to_compute)
            if out is None:
                # Late initialisation for getting the datatype of matrices
                out = {scat_key: np.zeros((len(frequencies), numangles, numangles),
                                          matrices[scat_key].dtype)
                       for scat_key in to_compute}
            for scat_key in to_compute:
                out[scat_key][i] = matrices[scat_key]
        return out

    def as_single_freq_matrices(self, frequency, numangles, to_compute=SCAT_KEYS):
        """
        Returns scattering matrices at a given frequency.

        Parameters
        ----------
        frequency : float
        numangles : int
        to_compute : set[str]


        Returns
        -------
        dict[str, ndarray]
            Shape of each matrix: ``(numpoints, numpoints)``

        """
        inc_theta, out_theta = make_angles_grid(numangles)
        return self(inc_theta, out_theta, frequency, to_compute)


class ScatteringFromFunc(Scattering):
    """
    Wrapper for scattering functions that take as three first arguments 'inc_theta',
    'out_theta' and 'frequency', and that accepts an argument 'to_compute'.

    To use:
    - create a class that inherit this class,
    - set the wrapped function as the '_scat_func' attribute,
    - populate the '_scat_kwargs' attribute with the extra arguments to pass to '_scat_func',
    ie any argument but 'inc_theta', 'out_theta', 'frequency' and 'to_compute'.
    """
    _scat_kwargs = None  # placeholder

    @staticmethod
    @abc.abstractmethod
    def _scat_func(*args, **kwargs):
        """Wrapped function."""
        raise NotImplementedError

    def __call__(self, inc_theta, out_theta, frequency, to_compute=SCAT_KEYS):
        return self._scat_func(inc_theta, out_theta, frequency,
                               to_compute=to_compute, **self._scat_kwargs)

    def __repr__(self):
        # Returns something like 'Scattering(x=1, y=2)'
        arg_str = ", ".join(['{}={}'.format(key, val)
                             for key, val in self._scat_kwargs.items()])
        return self.__class__.__qualname__ + '(' + arg_str + ')'


class Sdh2dScat(ScatteringFromFunc):
    '''
    Scattering for side-drilled hole
    '''
    _scat_func = staticmethod(sdh_2d_scat)

    def __init__(self, radius, longitudinal_vel, transverse_vel, min_terms=10,
                 term_factor=4):
        self._scat_kwargs = dict(radius=radius, longitudinal_vel=longitudinal_vel,
                                 transverse_vel=transverse_vel,
                                 min_terms=min_terms, term_factor=term_factor)


class PointSourceScat(ScatteringFromFunc):
    '''
    Scattering an unphysical point source. For debug only.

    For any incident and scattered angles, the scattering is defined as::

        S_LL = 1
        S_LT = v_L / v_T
        S_TL = -v_T / v_L
        S_TT = 1

    Remark: these scattering functions could have been defined as::

        S_LL = a
        S_LT = b * v_L / v_T
        S_TL = -b * v_T / v_L
        S_TT = c

    with any a, b, c. These coefficients were chosen arbitrarily in the present function.
    Therefore drawing quantitative conclusions from a model using this function must be
    done with care.

    '''

    @staticmethod
    def _scat_func(phi_in, phi_out, frequency, longitudinal_vel, transverse_vel,
                   to_compute=SCAT_KEYS):
        shape = np.shape(phi_in)
        assert np.shape(phi_in) == np.shape(phi_out)

        v_L = longitudinal_vel
        v_T = transverse_vel

        out = dict()
        if 'LL' in to_compute:
            out['LL'] = np.full(shape, 1.)
        if 'LT' in to_compute:
            out['LT'] = np.full(shape, v_L / v_T)
        if 'TL' in to_compute:
            out['TL'] = np.full(shape, -v_T / v_L)
        if 'TT' in to_compute:
            out['TT'] = np.full(shape, 1.)
        return out

    def __init__(self, longitudinal_vel, transverse_vel):
        self._scat_kwargs = dict(longitudinal_vel=longitudinal_vel,
                                 transverse_vel=transverse_vel)


class ScatteringFromMatrices(Scattering):
    pass
