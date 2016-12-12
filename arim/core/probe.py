from enum import IntEnum

import numpy as np

from .. import geometry as g
from .. import utils as u

__all__ = ['ElementShape', 'Probe']


class ElementShape(IntEnum):
    '''Enumeration which describes the shape of an element.
    The values are compliant with the specifications of MFMC format
    (field ``ELEMENT_SHAPE``).
    '''
    ellipse = 0
    rectangular = 1
    other = 2


class Probe:
    """A 'Probe' contains general information about its elements: locations in the PCS and the GCS, frequency, dimensions,
    etc.

    Parameters
    ----------
    locations :
        Cf. corresponding attribute.
    frequency :
        Cf. corresponding attribute.
    dimensions :
        Cf. corresponding attribute. Default: None
    orientations :
        Cf. corresponding attribute. Default: None
    shapes :
        Cf. corresponding attribute. Default: None
    dead_elements :
        Cf. corresponding attribute. Default: all elements are working.
    bandwidth :
        Cf. corresponding attribute. Default: None
    pcs :
        Cf. corresponding attribute. Default: PCS = GCS
    metadata :
        Cf. corresponding attribute. Default: empty dictionary


    Attributes
    ----------
    numelements : int
    locations : Points
        Locations of the elements centre in the GCS.
    locations_pcs : Points
        Locations of the elements centre in the PCS (read-only).
    frequency : float
    dimensions : Points
        Dimensions of elements in the PCS.
    orientations : Points or None
        Normal vector of elements surfaces in the GCS, towards the front of the probe. Norm: 1. 'None' if unknown.
    orientations_pcs : Points or None
        Normal vector of elements surfaces in the PCS, towards the front of the probe. Norm: 1. 'None' if unknown.
    dead_elements : ndarray
        1D array of size `numelements`. For each element, ``True`` if the element is dead (not working), ``False`` if
        the element is working.
    shapes : ndarray of ElementShape
    bandwidth : float or None
    pcs : CoordinateSystem
        Probe coordinate system.
    metadata : dict


    """

    __slots__ = ['locations', 'frequency', 'dimensions', 'orientations', 'dead_elements', 'shapes', 'bandwidth',
                 'metadata', 'numelements', 'pcs']

    def __init__(
            self,
            locations,
            frequency,
            dimensions=None,
            orientations=None,
            shapes=None,
            dead_elements=None,
            bandwidth=None,
            pcs=None,
            metadata=None,
    ):
        # Check shape and dimensions
        assert isinstance(locations, g.Points)
        numelements = len(locations)
        if dimensions is not None:
            assert isinstance(dimensions, g.Points)
            assert len(dimensions) == numelements
        if orientations is not None:
            assert isinstance(orientations, g.Points)
            assert len(orientations) == numelements
        if shapes is not None:
            _ = u.get_shape_safely(shapes, 'shapes', (numelements,))
        if dead_elements is not None:
            _ = u.get_shape_safely(dead_elements, 'dead_elements', (numelements,))

        # Populate optional parameters
        if dead_elements is None:
            dead_elements = np.full((numelements,), False, dtype=np.bool)
        if metadata is None:
            metadata = {}
        if pcs is None:
            pcs = g.GCS.copy()

        self.locations = locations
        self.dimensions = dimensions
        self.orientations = orientations
        self.shapes = shapes
        self.dead_elements = dead_elements
        self.pcs = pcs
        self.metadata = metadata
        self.numelements = numelements
        self.frequency = frequency
        self.bandwidth = bandwidth

        # Try to infer probe_type:
        if self.metadata.get('probe_type', None) is None:
            if numelements == 1:
                self.metadata['probe_type'] = 'single'
            elif g.are_points_aligned(locations):
                self.metadata['probe_type'] = 'linear'

    def __str__(self):
        return "{} - {} elements, {:.1f} MHz".format(
            u.get_name(self.metadata),
            self.numelements,
            self.frequency / 1e6)

    def __repr__(self):
        return "<{}: {} at {}>".format(
            self.__class__.__name__,
            str(self),
            hex(id(self)))

    @classmethod
    def make_matrix_probe(cls, numx, pitch_x, numy, pitch_y, frequency, *args, **kwargs):
        """
        Construct a matrix probe with ``numx × numy`` elements.
        Elements are indexed as follows: (X1, Y1), (X2, Y1), ..., (Xnumx, Y1), (X1, Y2), ...

        This class method is an alternative constructor for ``Probe``.

        The elements are centered around the point (0, 0, 0). All elements are in the plane z=0.

        Populate the following keys in internal dictionary `metadata` if they not exist: numx, pitch_x, numy,
        pitch_y, probe_type ('single', 'linear' or 'matrix').


        :param numx: number of elements along x axis
        :param pitch_x: difference between two consecutive elements along x axis (either positive or negative)
        :param numy: number of elements along y axis
        :param pitch_y: difference between two consecutive elements along y axis (either positive or negative)
        :param args: positional arguments passed to Probe.__init__
        :param kwargs: keywords arguments passed to Probe.__init__
        :return: Probe
        """
        numx = int(numx)
        numy = int(numy)

        if (numx < 1) or (numy < 1):
            raise ValueError("Number of elements along x or y must be strictly positive.")

        # the pitch in one row (or one column) is meaning
        if numx == 1:
            pitch_x = np.nan
        if numy == 1:
            pitch_y = np.nan

        # Force to float:
        pitch_x *= 1.
        pitch_y *= 1.

        # Get result datatype
        dtype = np.result_type(pitch_x, pitch_y)

        x = np.arange(numx, dtype=dtype)
        if numx > 1:
            # Remark: without this, we get a bug if pitch_x is NaN
            x *= pitch_x
        x -= x.mean()
        xx = np.tile(x, numy)

        y = np.arange(numy, dtype=dtype)
        if numy > 1:
            y *= pitch_y
        y -= y.mean()
        yy = np.repeat(y, numx)

        locations = g.Points.from_xyz(xx, yy, np.zeros(numx * numy, dtype=dtype))

        probe = cls(locations, frequency, *args, **kwargs)

        # Populate metadata dict:
        if probe.metadata.get('probe_type', None) is None:
            if (numx == 1) and (numy == 1):
                probe.metadata['probe_type'] = 'single'
            elif (numx == 1) or (numy == 1):
                probe.metadata['probe_type'] = 'linear'
            else:
                probe.metadata['probe_type'] = 'matrix'
        if probe.metadata.get('numx', None) is None:
            probe.metadata['numx'] = numx
        if probe.metadata.get('numy', None) is None:
            probe.metadata['numy'] = numy
        if probe.metadata.get('pitch_x', None) is None:
            probe.metadata['pitch_x'] = pitch_x
        if probe.metadata.get('pitch_y', None) is None:
            probe.metadata['pitch_y'] = pitch_y

        return probe

    @property
    def locations_pcs(self):
        return self.pcs.convert_from_gcs(self.locations)

    @property
    def orientations_pcs(self):
        if self.orientations is None:
            return None
        # use (O, i_hat, j_hat, k_hat) because the orientation are defined from point O.
        cs = g.CoordinateSystem((0., 0., 0.), self.pcs.i_hat, self.pcs.j_hat)
        return cs.convert_from_gcs(self.orientations)

    def rotate(self, rotation_matrix, centre=None):
        """
        Rotate the probe relatively in the GCS.

        The following attributes are updated: ``locations``, ``locations_pcs``, ``orientations`` (if provided),
        ``orientations_pcs`` (if orientions are provided), ``pcs``.

        Parameters
        ----------
        rotation_matrix : ndarray
            3x3 matrix
        centre : Points or None
            Centre of the rotation. If None: O(0, 0, 0) of GCS

        Returns
        -------
        self : returns the modified probe

        """
        locations = self.locations.rotate(rotation_matrix, centre)
        if self.orientations is not None:
            orientations = self.orientations.rotate(rotation_matrix, None)
        else:
            orientations = None
        pcs = self.pcs.rotate(rotation_matrix, centre)
        self.locations = locations
        self.orientations = orientations
        self.pcs = pcs
        return self

    def translate(self, vector):
        """
        Translate the probe relatively in the GCS.

        The following attributes are updated: ``locations``, ``locations_pcs``, ``pcs``.

        Parameters
        ----------
        vector : Point
            Translation vector

        Returns
        -------
        self : returns the modified probe

        """
        locations = self.locations.translate(vector)
        # do not translate the orientations!
        pcs = self.pcs.translate(vector)
        self.locations = locations
        self.pcs = pcs
        return self

    def set_element_dimensions(self, size_x, size_y, size_z):
        """
        Set the dimensions of all elements. This populates/overwrites the attribute ``dimensions``.

        Remark: in the case where elements does not have all the same dimension, update the attribute ``dimensions``
        manually.

        Parameters
        ----------
        size_x : float
        size_y : float
        size_z : float

        Returns
        -------
        self : returns the modified probe

        """
        size_x = 1. * size_x
        size_y = 1. * size_y
        size_z = 1. * size_z
        x = np.repeat(size_x, self.numelements)
        y = np.repeat(size_y, self.numelements)
        z = np.repeat(size_z, self.numelements)
        self.dimensions = g.Points.from_xyz(x, y, z)
        return self

    def flip_probe_around_axis_Oz(self):
        """
        Flip probe around axis Oz by 180°.
        """
        rot = g.rotation_matrix_z(np.pi)
        self.rotate(rot)

    def set_reference_element(self, reference_element='first'):
        """
        Change the origin of the PCS to a given element.

        Some prefer to have the reference point of the probe defined as the first element, some prefer
        the centre of the probe, other change their mind over time. This function should please all of us.

        Parameters
        ----------
        reference_element : float or str
            Can be: an element integer (between 0 and ``numelements-1``) or: 'first' (alias to 0), 'last' (alias to -1),
             'mean' (arithmetic mean of the centres of all elements).

        """

        if reference_element == 'first':
            new_origin = self.locations[0]
        elif reference_element == 'last':
            new_origin = self.locations[-1]
        elif reference_element == 'mean':
            new_origin = self.locations.coords.mean(axis=0)
        else:
            new_origin = self.locations[reference_element]

        self.pcs.origin = new_origin

    def translate_to_point_O(self):
        """
        Move the probe such as its PCS is in point O(0, 0, 0).
        """
        self.translate(-self.pcs.origin)
