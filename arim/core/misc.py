from collections import namedtuple
from warnings import warn

import numpy as np

from ..utils import linspace2, get_shape_safely, get_name
from .. import settings as s

__all__ = ['Time', 'Material', 'ExaminationObject', 'BoundedMedium', 'InfiniteMedium']


class Material(namedtuple('Material', 'longitudinal_vel shear_vel density metadata')):
    """
        >>> alu = Material(6300, {'long_name': 'Aluminium'})
    """

    def __new__(cls, longitudinal_vel, shear_vel=None, density=None, metadata=None):
        longitudinal_vel = longitudinal_vel * 1.

        if shear_vel is not None:
            shear_vel = shear_vel * 1.
        shear_vel = shear_vel

        if density is not None:
            density = density * 1.
        density = density

        if metadata is None:
            metadata = {}
        metadata = metadata

        return super().__new__(cls, longitudinal_vel, shear_vel, density, metadata)

    def __str__(self):
        name = self.metadata.get('long_name', None)
        if name is None:
            name = self.metadata.get('short_name', None)
            if name is None:
                name = 'Unnamed'

        return "{} - longitudinal vel.: {} m/s, transverse vel.: {} m/s)".format(
            name,
            self.longitudinal_vel,
            self.shear_vel,
            hex(id(self)))

    def __repr__(self):
        name = get_name(self.metadata)

        return "<{}: {} at {}>".format(
            self.__class__.__name__,
            str(self),
            hex(id(self)))


class ExaminationObject:
    def __init__(self, metadata=None):
        if metadata is None:
            metadata = {}
        self.metadata = metadata


class InfiniteMedium(ExaminationObject):
    def __init__(self, material, metadata=None):
        self.material = material

        super().__init__(metadata)


class BoundedMedium(ExaminationObject):
    def __init__(self, material, geometry, metadata=None):
        self.material = material
        self.geometry = geometry

        super().__init__(metadata)


class ExaminationObject:
    def __init__(self, material, geometry=None, metadata=None):
        self.material = material
        if geometry is None:
            geometry = {}
        self.geometry = geometry

        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def L(self):
        return self.material

class Time:
    """Linearly spaced time vector.

    Parameters
    ----------


    Attributes
    ----------
    samples : ndarray
    start : float
    end : float

    Notes
    -----
    Use len() to get the number of samples.

    """
    __slots__ = ['_samples', '_step']

    def __init__(self, start, step, num, dtype=None):
        step = step * 1.
        if step < 0:
            raise ValueError("'step' must be positive.")
        samples = linspace2(start, step, num, dtype)
        self._samples = samples
        self._step = step
        
    @property
    def samples(self):
        return self._samples
        
    @property
    def step(self):
        return self._step

    @property
    def start(self):
        return self._samples[0]

    @property
    def end(self):
        return self._samples[-1]

    def __len__(self):
        return len(self._samples)

    def __str__(self):
        r = '{} from {:.1f} to {:.1f} µs ({} samples, step={:.2f} µs)'
        return r.format(self.__class__.__qualname__, self.start * 1e6, self.end * 1e6, len(self), self.step * 1e6)

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))

    @classmethod
    def from_vect(cls, timevect, rtol=1e-2):
        """Construct a time object from a linearly spaced time vector.
        Check that the vector is linearly spaced (tolerance: all steps must be within
        ± 1e-3 times the average step).

        Parameters
        ----------
        timevect: array-like
            Linearly spaced array
        rtol: float
            Relative tolerance for the steps. Cf ``numpy.allclose``

        Raises
        ------
        ValueError

        """
        (num,) = get_shape_safely(timevect, 'timevect', (None,))

        start = timevect[0]
        steps = np.diff(timevect)
        avg_step = np.mean(steps)

        if not np.allclose(steps, avg_step, atol=0, rtol=rtol):
            raise ValueError("The vector seems not linearly spaced.")

        return cls(start, avg_step, num)
        
    def window(self, tmin=None, tmax=None, endpoint_left=True, endpoint_right=True):
        """
        Return a slice which selects the points [tmin, tmax].
        """
        if tmin is None:
            imin = None
        else:
            if endpoint_left:
                side = 'left'
            else:
                side = 'right'
            imin = np.searchsorted(self.samples, tmin, side=side)
        if tmax is None:
            imax = None
        else:
            if endpoint_right:
                side = 'right'
            else:
                side = 'left'
            imax = np.searchsorted(self.samples, tmax, side=side)
        return slice(imin, imax)

    def closest_index(self, time):
        return np.argmin(np.abs(self.samples - time))
