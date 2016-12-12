from collections import namedtuple
from warnings import warn
import enum

import numpy as np

from ..utils import linspace2, get_shape_safely, get_name, parse_enum_constant
from .. import settings as s

__all__ = ['Time', 'Material', 'ExaminationObject', 'BoundedMedium', 'InfiniteMedium',
           'StateMatter', 'TransmissionReflection', 'Mode', 'InterfaceKind', 'Interface',
           'Path', 'View']

StateMatter = enum.Enum('StateMatter', 'liquid solid')
StateMatter.__doc__ = "Enumerated constants for the states of matter."

TransmissionReflection = enum.Enum('TransmissionReflection', 'transmission reflection')
StateMatter.__doc__ = "Enumerated constants: transmission or reflection."


class Mode(enum.Enum):
    """Enumerated constants for the modes: L or T."""
    longitudinal = 0
    transverse = 1
    L = 0
    T = 1


@enum.unique
class InterfaceKind(enum.Enum):
    """Enumerated constants for the interface kinds."""
    fluid_solid = 0
    solid_fluid = 1


class Interface:
    """
    An Interface object contains information about the interface for a given ray path.
    It contains the locations of the interface points but also whether the rays are transmitted or reflected, etc.

    Parameters
    ----------
    points : Points
        Cf. attributes.
    orientations : Points
        Cf. attributes.
    kind : InterfaceKind or None
        Cf. attributes. Remark: accept strings but values are stored as :class:`InterfaceKind` constants.
    transmission_reflection : TransmissionReflection or str or None
        Cf. attributes. Remark: accept strings but values are stored as :class:`TransmissionReflection` constants.
    reflection_against : Material or None
        Cf. attributes.
    are_normals_on_inc_rays_side : bool or None
        Cf. attributes.
    are_normals_on_out_rays_side : bool or None
        Cf. attributes.

    Attributes
    ----------
    points : Points
        Location of the interface points.
    orientations : Points
        Orientations of the interface surface. For each interface point, the orientation is defined as three orthonormal vectors:
        the two first must be tangent to the surface, the third one must be the normal to the surface. All normals must
        point towards the same side of the surface.
    kind : InterfaceKind or str or None
        Kind of the interface. For example: "solid_fluid", "fluid_solid".
        None if not relevant.

        Note: internally, the corresponding constant from :class:`InterfaceKind` is stored.
    transmission_reflection : TransmissionReflection or str or None
        If the rays are transmitted through the interface, use "transmission".
        If the rays are reflected against the interface, use "reflection".
        For any other case, including probe emission and scattering, use None.

        Note: internally, the corresponding constant from :class:`TransmissionReflection` is stored.
    reflection_against : Material or None
        If the rays are reflected against the interface, this parameter is the material on the other side of
        the interface.
        For any other case, must be None.
    are_normals_on_inc_rays_side : bool or None
        Are the normals of the interface pointing towards the incoming rays? True or False.
        If not relevant (no incoming rays): None.
    are_normals_on_out_rays_side : bool or None
        Are the normals of the interface pointing towards the outgoing rays? True or False.
        If not relevant (no outgoing rays): None.
    """

    def __init__(self, points, orientations, kind=None, transmission_reflection=None, reflection_against=None,
                 are_normals_on_inc_rays_side=None, are_normals_on_out_rays_side=None):
        assert are_normals_on_inc_rays_side is None or isinstance(are_normals_on_inc_rays_side, bool)
        assert are_normals_on_out_rays_side is None or isinstance(are_normals_on_out_rays_side, bool)

        if transmission_reflection is not None:
            transmission_reflection = parse_enum_constant(transmission_reflection, TransmissionReflection)
        if kind is not None:
            kind = parse_enum_constant(kind, InterfaceKind)

        if (reflection_against is not None) and (transmission_reflection is not TransmissionReflection.reflection):
            raise ValueError("Parameter 'reflection_against' must be None for anything but a reflection")
        if (reflection_against is None) and (transmission_reflection is TransmissionReflection.reflection):
            raise ValueError("Parameter 'reflection_against' must be defined for a reflection")

        self.points = points
        self.orientations = orientations
        self.are_normals_on_inc_rays_side = are_normals_on_inc_rays_side
        self.are_normals_on_out_rays_side = are_normals_on_out_rays_side
        self.kind = kind
        self.transmission_reflection = transmission_reflection
        self.reflection_against = reflection_against

    def __str__(self):
        infos = []
        infos.append("Interface for points {}".format(self.points))
        infos.append("Number of points: {}".format(self.points.shape))
        if self.transmission_reflection is TransmissionReflection.transmission:
            infos.append("Transmission")
        elif self.transmission_reflection is TransmissionReflection.reflection:
            infos.append("Reflection against {}".format(self.reflection_against))
        if self.kind is not None:
            infos.append("Interface kind: {}".format(self.kind.name))

        infos.append("Orientations: {}".format(self.orientations))
        infos.append("Normals are on INC.. rays side: {}".format(self.are_normals_on_inc_rays_side))
        infos.append("Normals are on OUT. rays side: {}".format(self.are_normals_on_out_rays_side))
        infos_str = "\n".join(["    " + x if i > 0 else x for i, x in enumerate(infos)])
        return infos_str

    def __repr__(self):
        return "<{} for {} at {}>".format(
            self.__class__.__name__,
            str(self.points),
            hex(id(self)))


class Path:
    """
    A Path object specifies the interfaces, the materials and the modes related to a path.

    The Path specifies that rays are starting from a given interface and arriving at another one (via other interfaces).
    However the cordinates of the rays are not contained in path; these cordinates are the output of the ray tracing whereas
    a Path is the input of the ray tracing.

    The material ``materials[i]`` is between ``interfaces[i]`` and ``interfaces[i+1]``.

    Ray tracing can be done with the information contained in a Path object; no extra information is necessary.

    NB::

        numlegs = nummodes = numinterfaces - 1


    Parameters
    ----------
    interfaces : tuple of Interface
        Cf. attributes.
    materials : tuple of Material
        Cf. attributes.
    modes : tuple of Mode
        Cf. attributes.
    name : str or None
        Cf. attributes.

    Attributes
    ----------
    interfaces : tuple of Interface
        Interface where the rays goes through, including the extremities.
    materials : tuple of Material
        Materials where the rays goes through. The i-th leg of the ray is in the i-th material.
        The i-th material is between the i-th and the (i+1)-th interfaces.

        Lenght: numinterfaces - 1
    modes : tuple of Mode
        Mode for each leg.

        Lenght: numinterfaces - 1
    rays : Rays
        Results of ray tracing.
    name : str or None
        Name (optional)
    """

    def __init__(self, interfaces, materials, modes, name=None):
        numinterfaces = len(interfaces)
        numlegs = numinterfaces - 1

        assert numinterfaces >= 2

        assert len(materials) == numlegs
        assert len(modes) == numlegs

        self.interfaces = interfaces
        self.materials = materials
        self.modes = tuple(parse_enum_constant(mode, Mode) for mode in modes)
        self.name = name

    @property
    def numinterfaces(self):
        return len(self.interfaces)

    @property
    def numlegs(self):
        return len(self.interfaces) - 1

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)

    @property
    def velocities(self):
        return tuple(material.velocity(mode) for material, mode in zip(self.materials, self.modes))

    def to_fermat_path(self):
        """
        Create a FermatPath from this object (low-level object for ray tracing).

        Returns
        -------

        """
        from ..im.fermat_solver import FermatPath
        return FermatPath.from_path(self)


class Material(namedtuple('Material', 'longitudinal_vel transverse_vel density state_of_matter metadata')):
    """Material(longitudinal_vel, transverse_vel=None, density=None, state_of_matter=None, metadata=None)

    Parameters
    ----------
    longitudinal_vel : float
    transverse_vel : float or None
    density : float or None
    state_of_matter : StateMatter or None
    metadata : dict or None

    Example
    -------
        >>> alu = Material(6300, 3120, 2700, 'solid', {'long_name': 'Aluminium'})
    """

    def __new__(cls, longitudinal_vel, transverse_vel=None, density=None, state_of_matter=None, metadata=None):
        longitudinal_vel = longitudinal_vel * 1.

        if transverse_vel is not None:
            transverse_vel = transverse_vel * 1.

        if density is not None:
            density = density * 1.

        if state_of_matter is not None:
            state_of_matter = parse_enum_constant(state_of_matter, StateMatter)

        if metadata is None:
            metadata = {}

        return super().__new__(cls, longitudinal_vel, transverse_vel, density, state_of_matter, metadata)

    def __str__(self):
        name = get_name(self.metadata)

        return "{} (v_l: {} m/s, v_t: {} m/s)".format(
            name,
            self.longitudinal_vel,
            self.transverse_vel,
            hex(id(self)))

    def __repr__(self):
        name = get_name(self.metadata)

        return "<{}: {} at {}>".format(
            self.__class__.__name__,
            name,
            hex(id(self)))

    def velocity(self, mode):
        """
        Returns the velocity of the material for the mode 'mode'.

        Parameters
        ----------
        mode : Mode or string

        Returns
        -------
        velocitity: float

        """
        mode = parse_enum_constant(mode, Mode)
        if mode is Mode.longitudinal:
            return self.longitudinal_vel
        elif mode is Mode.transverse:
            return self.transverse_vel
        else:
            raise ValueError("Don't know what to do with mode '{}'".format(mode))


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


class View(namedtuple('View', ['tx_path', 'rx_path', 'name'])):
    """
    View(tx_path, rx_path, name)
    """
    __slots__ = []

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)