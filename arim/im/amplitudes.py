from _warnings import warn

import numpy as np

from .. import geometry as g
from .. import settings as s
from ..core import NoCache, ElementShape
from ..exceptions import ArimWarning
from .. import model

__all__ = ["Amplitudes", "UniformAmplitudes", "FixedAmplitudes", "AmplitudesRays",
           "MultiAmplitudes",
           "DirectivityCosine", "DirectivityFiniteWidth2D",
           "DirectivityFiniteWidth2D_Rays", "AmplitudesRemoveExtreme",
           "SensitivityConjugateAmplitudes"]


class Amplitudes:
    """
    Abstract class for computing lookup amplitudes in TFM.

    Usage: instantiate an object, then call it to obtain the amplitudes.

    Parameters
    ==========
    frame
    grid
    dtype
    fillvalue
    cache_res
    """

    def __init__(self, frame, grid, dtype=None, fillvalue=np.nan, cache_res=True):
        if dtype is None:
            dtype = s.FLOAT
        self.dtype = dtype

        self.frame = frame
        self.grid = grid

        self.fillvalue = fillvalue

        self.cache_res = cache_res
        self._cached_res = None

    @property
    def shape(self):
        return (self.grid.numpoints, self.frame.probe.numelements)

    def __call__(self):
        """
        Returns the amplitudes (possibly cached).
        """
        if self._cached_res is not None:
            return self._cached_res
        res = self._compute_amplitudes()
        if self.cache_res:
            self._cached_res = res
        return res

    def clear(self):
        self._cached_res = None

    def _compute_amplitudes(self):
        raise NotImplementedError('must be implemented by child class')


class UniformAmplitudes(Amplitudes):
    """
    Uniform amplitudes: for any grid point, any transmitter and any receiver,
    the amplitude is one (1.).
    """

    def _compute_amplitudes(self):
        return np.ones(self.shape, dtype=self.dtype)


class FixedAmplitudes(Amplitudes):
    """
    Specifiy an array as amplitudes.
    """

    def __init__(self, frame, grid, amps):
        dtype = amps.dtype
        super().__init__(frame, grid, dtype, fillvalue=None, cache_res=True)

        if amps.shape != self.shape:
            raise ValueError(
                "invalid amplitudes shape: expected {}, got {}".format(self.shape,
                                                                       amps.shape))
        self._amps = amps

    def _compute_amplitudes(self):
        return self._amps


class AmplitudesRays(Amplitudes):
    def __init__(self, frame, grid, rays, **kwargs):
        """
        Abstract class for amplitudes computation on the first leg of rays.

        Parameters
        ----------
        frame : Frame
        grid : Grid
        rays : Rays
        kwargs : dict
            Arguments for Amplitudes.__init__

        Returns
        -------

        """
        super().__init__(frame=frame, grid=grid, **kwargs)
        self.rays = rays

        # TODO : use global cache here
        first_interface = rays.path.points[1]
        geom_first_leg = g.GeometryHelper(frame.probe.locations, first_interface,
                                          frame.probe.pcs)
        self.geom_first_leg = geom_first_leg

    def first_leg_spherical(self):
        """
        Returns
        -------
        r, theta, phi : ndarray
            Size: (numpoints, numelements)
            Spherical coordinates of the first leg of the ray.
            Considering a ray starting from the j-th element going to the i-th grid point, ``theta[i, j]`` is the
            polar angle between the normal of the j-th element and the first leg of the ray.

        """
        numelements = self.frame.probe.numelements

        # Spherical coordinates of the points of the first leg
        spher = self.geom_first_leg.points2_to_pcs_pairwise_spherical()

        # This array gives the indices of the points of the first interface depending on the probe element
        # and the grid point:
        first_interfaces_indices = self.rays.indices[..., 1].T
        assert first_interfaces_indices.shape == (self.grid.numpoints, numelements)

        r = spher.r[first_interfaces_indices, np.arange(numelements, dtype=int)]
        theta = spher.theta[first_interfaces_indices, np.arange(numelements, dtype=int)]
        phi = spher.phi[first_interfaces_indices, np.arange(numelements, dtype=int)]
        return g.SphericalCoordinates(r, theta, phi)


class MultiAmplitudes(Amplitudes):
    def __init__(self, amplitudes_list, **kwargs):
        """
        Amplitudes obtained by multiplying several amplitudes.

        Use this class if you want to combine several Amplitudes objects.

        Parameters
        ----------
        amplitudes_list : list of Amplitudes
        cache_res : boolean
        kwargs : dict
            Arguments for Amplitudes()

        Returns
        -------

        """
        self.amplitudes_list = amplitudes_list
        if len(set(amp.frame for amp in amplitudes_list)) > 1:
            warn('Different Frame objects are used in amplitudes. Use first.',
                 ArimWarning)
        if len(set(amp.grid for amp in amplitudes_list)) > 1:
            warn('Different Grid objects are used in amplitudes. Use first.', ArimWarning)
        super().__init__(frame=amplitudes_list[0].frame, grid=amplitudes_list[0].grid)

    def _compute_amplitudes(self):
        """
        Multiply all amplitudes in the container.
        """
        res = None
        for amps in self.amplitudes_list:
            if res is None:
                res = amps()
            else:
                res *= amps()  # inplace multiplication
        return res

    def __str__(self):
        return "\n".join(str(amps) for amps in self)

    def clear(self):
        self._cached_res = None
        for amps in self.amplitudes_list:
            amps.clear()


class DirectivityFiniteWidth2D(Amplitudes):
    """
    Amplitudes using a model of directivity of the element (``directivity_finite_width_2d``).

    Rays are assumed to go straight from the elements to the grid points.
    For multiview TFM, use ``DirectivityFiniteWidth2D_Rays``.

    Warning: ``probe.dimensions[0]`` is assumed to be the width of the elements. 
    
    """

    def __init__(self, frame, grid, speed, **kwargs):
        self.speed = speed

        # Validation:
        numelements = frame.probe.numelements

        probe = frame.probe
        if probe.dimensions is None:
            raise ValueError('Dimensions of the elements must be specified.')

        if probe.orientations is None:
            warn(
                'The orientations of elements are not provided. Assume they are all [0., 0., 1.] in the PCS',
                ArimWarning)
        else:
            expected_orientations = np.tile([0, 0., 1.], numelements).reshape(numelements,
                                                                              3)
            assert np.allclose(probe.orientations_pcs, expected_orientations), \
                'This function works only if element orientations in the PCS are all [0., 0., 1.].'

        if probe.shapes is None:
            warn('Elements shapes are not provided. Assume rectangular.', ArimWarning)
        else:
            assert np.all(probe.shapes == ElementShape.rectangular), \
                'Elements shapes must be rectangular.'

        # TODO : use global cache here
        self.geom_probe_to_grid = g.GeometryHelper(frame.probe.locations, grid.as_points,
                                                   frame.probe.pcs)

        super().__init__(frame=frame, grid=grid, **kwargs)

    def _compute_amplitudes(self):
        numelements = self.frame.probe.numelements
        amplitudes = np.zeros((self.grid.numpoints, numelements), dtype=self.dtype)
        probe = self.frame.probe

        wavelength = self.speed / probe.frequency

        spher = self.geom_probe_to_grid.points2_to_pcs_pairwise_spherical()
        for (element, (elt_dim, elt_loc)) in enumerate(
                zip(probe.dimensions, probe.locations)):
            elt_width = elt_dim[0]
            amplitudes[..., element] = model.directivity_finite_width_2d(
                spher.theta[..., element], elt_width,
                wavelength)
        return amplitudes


class DirectivityFiniteWidth2D_Rays(AmplitudesRays):
    """
    Amplitudes using a model of directivity of the element (``directivity_finite_width_2d``).

    The first leg of the ray is used.

    Warning: ``probe.dimensions[0]`` is assumed to be the width of the elements. 

    Parameters
    ----------
    frame : Frame
    grid : Grid
    rays : Rays
    speed : float
    kwargs : dict
        Extra arguments for parent object.

    """

    def __init__(self, frame, grid, rays, speed, **kwargs):
        super().__init__(frame=frame, grid=grid, rays=rays, **kwargs)
        self.speed = speed

    def _compute_amplitudes(self):
        numelements = self.frame.probe.numelements
        amplitudes = np.zeros((self.grid.numpoints, numelements), dtype=self.dtype)
        probe = self.frame.probe

        spher = self.first_leg_spherical()

        for (element, (elt_dim, elt_loc)) in enumerate(
                zip(probe.dimensions, probe.locations)):
            elt_width = elt_dim[0]
            amplitudes[..., element] = model.directivity_finite_width_2d(
                spher.theta[..., element], elt_width,
                self.wavelength)
        return amplitudes

    # def get_theta_first_interface(self):
    #     numelements = self.frame.probe.numelements
    #     spher = self.geom_probe_to_first_interface.points2_to_pcs_pairwise_spherical()
    #     all_theta = spher.theta
    #     first_interfaces_indices = self.rays.indices[..., 1].T
    #     assert first_interfaces_indices.shape == (self.grid.numpoints, numelements)
    #     theta = all_theta[first_interfaces_indices, np.arange(numelements, dtype=int)]
    #     assert first_interfaces_indices.shape == (self.grid.numpoints, numelements)
    #     return theta

    @property
    def wavelength(self):
        return self.speed / self.frame.probe.frequency


class DirectivityCosine(AmplitudesRays):
    """
    Amplitudes of cosine theta where theta is the angle between the normal
    of the element and the first leg of the rays.
    """

    def _compute_amplitudes(self):
        spher = self.first_leg_spherical()
        return np.cos(spher.theta)


class AmplitudesRemoveExtreme(Amplitudes):
    """
    Assign the fillvalue to rays that are passing through the first/last points
    of the intermediary interfaces. Such rays may be non-physical.

    """

    def __init__(self, frame, grid, rays, cache_extreme=None, **kwargs):
        if cache_extreme is None:
            cache_extreme = NoCache()
        self._cache_extreme = cache_extreme
        # TODO : use cache
        self.rays = rays
        super().__init__(frame, grid, **kwargs)

    def _compute_amplitudes(self):
        amps = np.ones((self.shape), dtype=self.dtype)

        # get rays that goes through extreme points (non-physical rays in general)
        extreme_points = self.rays.gone_through_extreme_points().T

        amps[extreme_points] = self.fillvalue
        return amps


class SensitivityConjugateAmplitudes(Amplitudes):
    """


    Parameters
    ----------
    frame : Frame
    grid : Grid
    ray_weights : ndarray
        Shape:
    sensitivity
    divide_by_sensitivity
    kwargs
    """

    def __init__(self, frame, grid, ray_weights, sensitivity=None,
                 divide_by_sensitivity=False, **kwargs):
        ray_weights = np.asarray(ray_weights)
        assert ray_weights.shape == (frame.probe.numelements, grid.numpoints)

        if sensitivity is not None:
            sensitivity = np.asarray(sensitivity)
            assert sensitivity.shape == (grid.numpoints,)

        self.ray_weights = ray_weights
        self.sensitivity = sensitivity
        self.divide_by_sensitivity = divide_by_sensitivity
        super().__init__(frame, grid, **kwargs)

    def _compute_amplitudes(self):
        amps = np.conjugate(self.ray_weights)
        if self.divide_by_sensitivity:
            sensitivity = self.sensitivity
            if sensitivity is None:
                sensitivity = model.sensitivity_conjugate_for_path(self.ray_weights)
            assert np.can_cast(sensitivity, amps.dtype, 'safe')
            amps /= sensitivity[np.newaxis, ...]
        return np.ascontiguousarray(amps.T)
