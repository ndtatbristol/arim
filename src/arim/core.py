"""
Defines core objects of arim.
"""
import copy
import enum
import warnings
from collections import namedtuple

import numpy as np

from . import geometry as g
from . import helpers, ut

StateMatter = enum.Enum("StateMatter", "liquid solid")
StateMatter.__doc__ = "Enumerated constants for the states of matter."
TransmissionReflection = enum.Enum("TransmissionReflection", "transmission reflection")
TransmissionReflection.__doc__ = "Enumerated constants: transmission or reflection."


def _constant_mat_att(value):
    def attenuation(frequency):
        return np.full_like(frequency, value)

    return attenuation


def _polynomial_mat_att(coeffs):
    p = np.polynomial.Polynomial(coeffs)

    def attenuation(frequency):
        return p(frequency / 1e6)  # scale to avoid numerical issues

    return attenuation


def material_attenuation_factory(kind, *args, **kwargs):
    """
    Material attenuation coefficient in Np/m

    When called, this object returns a frequency-dependent attenuation coefficient
    in Nepers per meter (Np/m).

    Warning: in ``kind=polynomial``, the frequency vector in Hz is rescaled to MHz to avoid numerical issues.

    Parameters
    ----------
    kind : str
        Available methods: 'constant', 'polynomial'.
    args, kwargs
        Extra arguments. See examples

    Returns
    -------
    mat_att_func : function
        Function that takes the frequency array as an input and returns the attenuation in Np/m as an output.

    Examples
    --------
    Constant material attenuation:
    >>> mat_att_func = material_attenuation_factory("constant", 15.)

    Polynomial material attenuation ``(1 + 2*(frequency/1e6) + 3*(frequency/1e6)**2)``
    >>> mat_att_func = material_attenuation_factory("polynomial", (1, 2, 3))

    To use:
    >>> frequency = 5e6
    >>> mat_att_func(frequency)

    """
    if kind == "constant":
        return _constant_mat_att(*args, **kwargs)
    elif kind == "polynomial":
        return _polynomial_mat_att(*args, **kwargs)
    else:
        raise ValueError(f"unknown material attenuation method '{kind}'")


class CaptureMethod(enum.Enum):
    """
    Capture method: unsupported, fmc, hmc
    """

    unsupported = 0
    fmc = 1
    hmc = 2


class Frame:
    """
    A frame contains the data received by a probe at a specific location.

    timetraces are stored a 2D array of length `numtimetraces x numsamples`. Each line of the array  is a timetrace, i.e.
    the data received by a specific element when a specific element was transmitting.

    Parameters
    ----------
    timetraces
    tx
    rx
    probe
    examination_object
    metadata

    Attributes
    ----------
    timetraces : ndarray
        Time-traces as a 2d array of shape `(numtimetraces, numsamples)`.
    time : Time
        Time vector associated to all timetraces.
    tx : ndarray
        1D array of length `numtimetraces`. `tx[i]` is the index of the element
        transmitting during the acquisition of the i-th timetrace.
    rx : ndarray
        1D array of length `numtimetraces`. `rx[i]` is the index of the element
        receiving during the acquisition of the i-th timetrace.
    probe : Probe
        Probe used during acquisition.
    examination_object : ExaminationObject
        Object inspected.
    numtimetraces : int
        Number of timetraces in the frame.
    metadata : dict
        Metadata


    """

    def __init__(
        self, timetraces, time, tx, rx, probe, examination_object, metadata=None
    ):
        # Check shape and dimensions
        try:
            time.samples
        except AttributeError:
            raise TypeError(
                f"'time' should be an object 'Time' (current: {type(time)})."
            )
        numsamples = len(time)

        timetraces = np.asarray(timetraces)
        tx = np.asarray(tx)
        rx = np.asarray(rx)
        if tx.dtype.kind not in ("i", "u"):
            raise TypeError(f"transmitters must be integer indices (got {tx.dtype})")
        if rx.dtype.kind not in ("i", "u"):
            raise TypeError(f"receivers must be integer indices (got {rx.dtype})")

        (numtimetraces, _) = helpers.get_shape_safely(
            timetraces, "timetraces", (None, numsamples)
        )
        _ = helpers.get_shape_safely(tx, "tx", (numtimetraces,))
        _ = helpers.get_shape_safely(rx, "rx", (numtimetraces,))

        unique_tx_rx_pairs = {(tx_i, rx_i) for tx_i, rx_i in zip(tx, rx)}
        if len(unique_tx_rx_pairs) < numtimetraces:
            raise ValueError("The frame contains duplicate timetraces")

        self.timetraces = timetraces
        self.tx = tx
        self.rx = rx
        self.time = time
        self.probe = probe
        self.examination_object = examination_object

        self.numtimetraces = numtimetraces
        self.numsamples = numsamples

        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def numscanlines(self):
        warnings.warn(
            DeprecationWarning(
                "Frame.numscanlines is deprecated, use Frame.numtimetraces instead"
            )
        )
        return self.numtimetraces

    @property
    def scanlines(self):
        warnings.warn(
            DeprecationWarning(
                "Frame.scanlines is deprecated, use Frame.timetraces instead"
            )
        )
        return self.timetraces

    @property
    def capture_method(self):
        return CaptureMethod[ut.infer_capture_method(self.tx, self.rx)]

    def apply_filter(self, filt):
        """
        Filter the timetraces and save them in the frame.

        Warning: the attribute ``timetraces`` is overwritten during this operation.

        Parameters
        ----------
        filt: Filter

        Returns
        -------
        frame: Frame
            New Frame object where the timetraces are filtered. All objects are passed
            by reference.

        """
        new_timetraces = filt(self.timetraces)
        return self.__class__(
            new_timetraces,
            self.time,
            self.tx,
            self.rx,
            self.probe,
            self.examination_object,
            self.metadata,
        )

    def get_timetrace(self, tx, rx):
        """
        Return the timetrace corresponding to the pair (tx, rx).

        Parameters
        ----------
        tx: int
        rx: int

        Returns
        -------
        scan: 1d array

        """
        match = np.logical_and(self.tx == tx, self.rx == rx)

        match_timetrace = self.timetraces[match]
        if match_timetrace.shape[0] == 1:
            return match_timetrace[0]
        else:
            raise IndexError("no timetrace")

    def get_scanline(self, tx, rx):
        warnings.warn(
            DeprecationWarning(
                "Frame.get_scanline is deprecated, use Frame.get_timetrace instead"
            )
        )
        return self.get_timetrace(tx, rx)

    def expand_frame_assuming_reciprocity(self):
        """
        Return a new Frame where new timetraces are inferred assuming reciprocity of
        transmitters and receivers.

        Assumes that the timetrace obtained with the transmitter i and the receiver j
        is the same as the one obtained with the transmitter j and the receiver i.
        If the original frame is a FMC, this function returns a copy of the original
        frame.

        Canonical example: expand a HMC frame to a FMC frame.

        Returns
        -------
        Frame

        """
        orig_pairs = {(tx, rx) for tx, rx in zip(self.tx, self.rx)}
        reciprocal_pairs = {(rx, tx) for tx, rx in zip(self.tx, self.rx)}

        if orig_pairs == reciprocal_pairs:
            # No expansion needed
            return self

        all_pairs = sorted(orig_pairs | reciprocal_pairs)

        pair_to_scan_idx = {
            (tx, rx): i for i, (tx, rx) in enumerate(zip(self.tx, self.rx))
        }

        new_timetraces = np.empty(
            (len(all_pairs), len(self.time)), self.timetraces.dtype
        )

        new_tx, new_rx = zip(*all_pairs)
        for new_scan_idx, (tx, rx) in enumerate(all_pairs):
            try:
                # use the available information if present
                old_scan_idx = pair_to_scan_idx[tx, rx]
            except KeyError:
                # this is the expansion
                old_scan_idx = pair_to_scan_idx[rx, tx]

            new_timetraces[new_scan_idx] = self.timetraces[old_scan_idx]

        return self.__class__(
            new_timetraces,
            self.time,
            new_tx,
            new_rx,
            self.probe,
            self.examination_object,
            self.metadata,
        )

    def is_complete_assuming_reciprocity(self):
        """
        Returns: "for all pair of elements (i, j), the timetrace (i, j) is in the frame
        if and only if the timetrace (j, i) exists."

        Returns
        -------
        bool
        """
        orig_pairs = {(tx, rx) for tx, rx in zip(self.tx, self.rx)}
        reciprocal_pairs = {(rx, tx) for tx, rx in zip(self.tx, self.rx)}
        return orig_pairs == reciprocal_pairs

    def drop_timetraces(self, timetraces_idx):
        pass

    def subframe(self, timetraces_idx):
        """Return a new Frame containing only a subset of the original timetraces.

        Parameters
        ----------
        timetraces_idx : slice or tuple or list or array
            Index of the elements of the original probe to retain.
            Any valid numpy index is accepted.

        Returns
        -------
        subframe : Frame
            Subframe

        Notes
        -----
        The new Frame shares the time, probe, examination object and metadata as the original
        one, and may or may not share the same timetraces, tx and rx depending on the kind of
        indexing.
        """
        timetraces = self.timetraces[timetraces_idx]
        tx = self.tx[timetraces_idx]
        rx = self.rx[timetraces_idx]
        return self.__class__(
            timetraces,
            self.time,
            tx,
            rx,
            self.probe,
            self.examination_object,
            self.metadata,
        )

    def subframe_from_probe_elements(self, elements_idx, make_subprobe=True):
        """Returns a new Frame which only contains specified probe elements

        Parameters
        ----------
        elements_idx : slice or tuple or list or array
            Probe elements to retain. Any valid numpy index is accepted.
        make_subprobe : bool, optional
            If True (default), the subframe contains a new Probe, containing only the
            specified probe elements.
            If False, the subframe contains the original Probe .

        Returns
        -------
        Frame
            Frame

        Notes
        -----
        In the current implementation, ray tracing and forward modelling is
        done using all elements of a probe, independently on their actual use
        in a Frame.
        This may lead to unnecessary computation if using ``make_subprobe=False``.
        However, using ``make_subprobe=True`` allows to reuse ray tracing
        or forward model results if multiple subframes are created.

        If using ``make_subprobe=True``, see also :meth:`Probe.subprobe`.

        """
        retained_elements = np.arange(self.probe.numelements)[elements_idx]
        retained_timetraces_idx = np.logical_and(
            np.isin(self.tx, retained_elements), np.isin(self.rx, retained_elements)
        )
        if not make_subprobe:
            # We just need to retain the timetraces that use the retained elements
            return self.subframe(retained_timetraces_idx)
        else:
            subprobe = self.probe.subprobe(elements_idx)

            # Construct an array so that mapper[old_elt_idx]==new_elt_idx
            mapper = np.zeros(self.probe.numelements, dtype=np.int_)
            mapper[elements_idx] = np.arange(subprobe.numelements)

            tx = mapper[self.tx[retained_timetraces_idx]]
            rx = mapper[self.rx[retained_timetraces_idx]]
            timetraces = self.timetraces[retained_timetraces_idx]
            return self.__class__(
                timetraces,
                self.time,
                tx,
                rx,
                subprobe,
                self.examination_object,
                self.metadata,
            )


class ElementShape(enum.IntEnum):
    """Enumeration which describes the shape of an element.
    The values are compliant with the specifications of MFMC format
    (field ``ELEMENT_SHAPE``).
    """

    ellipse = 0
    rectangular = 1
    other = 2


class Probe:
    """A 'Probe' contains general information about its elements: locations in the PCS and the GCS, frequency, dimensions,
    etc.

    Parameters
    ----------
    locations : Points or array-like
        Accepted array shape: ``(numelements, 3)``
        Cf. corresponding attribute.
    frequency : float
        Cf. corresponding attribute.
    dimensions : Points or array-like
        Accepted array shapes: ``(numelements, 3)`` or ``(3, )`` (if same for all).
        Cf. corresponding attribute. Default: None
    orientations : Points or array-like
        Accepted array shapes: ``(numelements, 3)`` or ``(3, )`` (if same for all).
        Cf. corresponding attribute. Default: None
    shapes : ElementShape or List[ElementShape] or array-like
        Accepted values: one value or ``numelements`` values.
        Cf. corresponding attribute. Default: None
    dead_elements : bool or List[bool]
        Accepted values: one value or ``numelements`` values.
        Cf. corresponding attribute. Default: all elements are working.
    bandwidth : float
        Cf. corresponding attribute. Default: None
    pcs : CoordinateSystem
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
    dimensions : Points or None
        Dimensions of elements in the PCS.
    orientations : Points or None
        Normal vector of elements surfaces in the GCS, towards the front of the probe. Norm: 1. 'None' if unknown.
    orientations_pcs : Points or None
        Normal vector of elements surfaces in the PCS, towards the front of the probe. Norm: 1. 'None' if unknown.
    dead_elements : ndarray of bool
        1D array of size `numelements`. For each element, ``True`` if the element is dead (not working), ``False`` if
        the element is working.
    shapes : ndarray of ElementShape or None
    bandwidth : float or None
    pcs : CoordinateSystem
        Probe coordinate system.
    metadata : dict


    """

    __slots__ = [
        "locations",
        "frequency",
        "dimensions",
        "orientations",
        "dead_elements",
        "shapes",
        "bandwidth",
        "metadata",
        "numelements",
        "pcs",
    ]

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
        locations = g.aspoints(locations)
        numelements = len(locations)
        assert locations.ndim == 1
        if dimensions is not None:
            dimensions = g.aspoints(dimensions)
            if dimensions.shape == ():
                dimensions = g.Points(
                    np.resize(dimensions.coords, (numelements, 3)), name=dimensions.name
                )
            assert dimensions.shape == (numelements,)
        if orientations is not None:
            orientations = g.aspoints(orientations)
            if orientations.shape == ():
                orientations = g.Points(
                    np.resize(orientations.coords, (numelements, 3)),
                    name=orientations.name,
                )
            assert orientations.shape == (numelements,)
        if shapes is not None:
            # force dtype=object in the case we got a IntEnum (which would be convert to int otherwise)
            shapes = np.asarray(shapes, dtype=object)
            if shapes.shape == ():
                shapes = np.resize(shapes, (numelements,))
            assert shapes.shape == (numelements,)

        if dead_elements is None:
            dead_elements = np.full((numelements,), False, dtype=bool)
        else:
            dead_elements = np.asarray(dead_elements, dtype=bool)
            if dead_elements.shape == ():
                dead_elements = np.resize(dead_elements, (numelements,))
        assert dead_elements.shape == (numelements,)

        # Populate optional parameters
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
        self.frequency = None if frequency is None else float(frequency)
        self.bandwidth = None if bandwidth is None else float(bandwidth)

    def __str__(self):
        return "{} - {} elements, {:.1f} MHz".format(
            helpers.get_name(self.metadata), self.numelements, self.frequency / 1e6
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}: {str(self)} at {hex(id(self))}>"

    @classmethod
    def make_matrix_probe(
        cls, numx, pitch_x, numy, pitch_y, frequency, *args, **kwargs
    ):
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
            raise ValueError(
                "Number of elements along x or y must be strictly positive."
            )

        # the pitch in one row (or one column) is meaning
        if numx == 1:
            pitch_x = np.nan
        if numy == 1:
            pitch_y = np.nan

        # Force to float:
        pitch_x *= 1.0
        pitch_y *= 1.0

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
        if probe.metadata.get("probe_type", None) is None:
            if (numx == 1) and (numy == 1):
                probe.metadata["probe_type"] = "single"
            elif (numx == 1) or (numy == 1):
                probe.metadata["probe_type"] = "linear"
            else:
                probe.metadata["probe_type"] = "matrix"
        if probe.metadata.get("numx", None) is None:
            probe.metadata["numx"] = numx
        if probe.metadata.get("numy", None) is None:
            probe.metadata["numy"] = numy
        if probe.metadata.get("pitch_x", None) is None:
            probe.metadata["pitch_x"] = pitch_x
        if probe.metadata.get("pitch_y", None) is None:
            probe.metadata["pitch_y"] = pitch_y

        return probe

    def subprobe(self, elements_idx, save_metadata=False):
        """Return a new Probe with only a subset of the original elements.

        Parameters
        ----------
        elements_idx : slice or tuple or list or array
            Index of the elements of the original probe to retain.
            Any valid numpy index is accepted.
        save_metadata : bool, optional
            Whether to retain the original metadata  (the default is False)

        Returns
        -------
        subprobe : Probe

        Notes
        ----
        The original PCS is retained and may no longer be at a desirable location
        in the subprobe; consider using :meth:`Probe.set_reference_element``.

        The subprobe elements will be in the order specified in ``elements_idx``.
        The subprobe will contain duplicate elements if ``elements_idx`` has
        duplicate entries.

        Examples
        --------

        >>> probe.subprobe([0, 1, 3])
        # Returns a subprobe with the first three elements

        >>> probe.subprobe(np.s_[0::2])
        # Returns a subprobe with every other element, starting from element 0.

        """

        def _index(x):
            if x is None:
                return None
            else:
                return x[elements_idx]

        # subprobe parameters:
        locations = _index(self.locations)
        frequency = self.frequency
        dimensions = _index(self.dimensions)
        orientations = _index(self.orientations)
        shapes = _index(self.shapes)
        shapes = _index(self.shapes)
        dead_elements = _index(self.dead_elements)
        bandwidth = self.bandwidth
        pcs = copy.deepcopy(self.pcs)
        if save_metadata:
            metadata = copy.deepcopy(self.metadata)
        else:
            metadata = None

        return self.__class__(
            locations,
            frequency,
            dimensions,
            orientations,
            shapes,
            dead_elements,
            bandwidth,
            pcs,
            metadata,
        )

    @property
    def locations_pcs(self):
        return self.pcs.convert_from_gcs(self.locations)

    @property
    def orientations_pcs(self):
        if self.orientations is None:
            return None
        # use (O, i_hat, j_hat, k_hat) because the orientation are defined from point O.
        cs = g.CoordinateSystem((0.0, 0.0, 0.0), self.pcs.i_hat, self.pcs.j_hat)
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
        size_x = 1.0 * size_x
        size_y = 1.0 * size_y
        size_z = 1.0 * size_z
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

    def set_reference_element(self, reference_element="first"):
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

        if reference_element == "first":
            new_origin = self.locations[0]
        elif reference_element == "last":
            new_origin = self.locations[-1]
        elif reference_element == "mean":
            new_origin = self.locations.coords.mean(axis=0)
        else:
            new_origin = self.locations[reference_element]

        self.pcs.origin = new_origin

    def translate_to_point_O(self):
        """
        Move the probe such as its PCS is in point O(0, 0, 0).
        """
        self.translate(-self.pcs.origin)

    def reset_position(self):
        """
        Translate and rotate the probe such as its PCS coincides with the GCS. Returns None.
        """
        self.translate_to_point_O()

        # inverse rotation:
        rotation_matrix = np.stack(
            (self.pcs.i_hat, self.pcs.j_hat, self.pcs.k_hat), axis=0
        )

        self.rotate(rotation_matrix)

    def to_oriented_points(self):
        """
        Returns the locations and the orientations of the probe elements in the GCS
        as a OrientedPoints object.

        Returns
        -------
        OrientedPoints

        """
        return g.points_from_probe(self)


class Mode(enum.Enum):
    """Enumerated constants for the modes: L or T."""

    longitudinal = 0
    transverse = 1
    L = 0
    T = 1

    def reverse(self):
        if self is self.longitudinal:
            return self.transverse
        elif self is self.transverse:
            return self.longitudinal
        else:
            raise RuntimeError

    def key(self):
        """
        Returns a one-letter key string: 'L' for longitudinal, 'T' for transverse.

        Returns
        -------
        str

        """
        if self is self.longitudinal:
            return "L"
        elif self is self.transverse:
            return "T"
        else:
            raise RuntimeError


@enum.unique
class InterfaceKind(enum.Enum):
    """Enumerated constants for the interface kinds."""

    fluid_solid = 0
    solid_fluid = 1

    def reverse(self):
        """

        Returns
        -------
        InterfaceKind

        """
        if self is self.fluid_solid:
            return self.solid_fluid
        elif self is self.solid_fluid:
            return self.fluid_solid
        else:
            raise RuntimeError("invalid case")


class Interface:
    """
    An Interface object contains information about the interface for a given ray path.
    It contains the locations of the interface points but also whether the rays are transmitted or reflected, etc.

    Parameters
    ----------
    points : Points
        Cf. attributes.
    orientations : Points
        Cf. attributes. Accepted shape: ``(3, )`` or  ``(*points.shape, 3)``. In the first case,
        the orientation is assumed to be the same for all points.
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

    def __init__(
        self,
        points,
        orientations,
        kind=None,
        transmission_reflection=None,
        reflection_against=None,
        are_normals_on_inc_rays_side=None,
        are_normals_on_out_rays_side=None,
    ):
        assert are_normals_on_inc_rays_side is None or isinstance(
            are_normals_on_inc_rays_side, bool
        )
        assert are_normals_on_out_rays_side is None or isinstance(
            are_normals_on_out_rays_side, bool
        )

        if transmission_reflection is not None:
            transmission_reflection = helpers.parse_enum_constant(
                transmission_reflection, TransmissionReflection
            )
        if kind is not None:
            kind = helpers.parse_enum_constant(kind, InterfaceKind)

        if (reflection_against is not None) and (
            transmission_reflection is not TransmissionReflection.reflection
        ):
            raise ValueError(
                "Parameter 'reflection_against' must be None for anything but a reflection"
            )
        if (reflection_against is None) and (
            transmission_reflection is TransmissionReflection.reflection
        ):
            raise ValueError(
                "Parameter 'reflection_against' must be defined for a reflection"
            )

        points = g.aspoints(points)
        orientations = g.aspoints(orientations)
        if orientations.shape == (3,):
            # only one value has been given, assume it is the same for every point:
            orientations = g.Points(
                np.resize(orientations.coords, (*points.shape, 3, 3)), orientations.name
            )
        if orientations.shape != (*points.shape, 3):
            raise ValueError("inconsistent shapes for points and orientations")

        self.points = points
        self.orientations = orientations
        self.are_normals_on_inc_rays_side = are_normals_on_inc_rays_side
        self.are_normals_on_out_rays_side = are_normals_on_out_rays_side
        self.kind = kind
        self.transmission_reflection = transmission_reflection
        self.reflection_against = reflection_against

    def __str__(self):
        infos = []
        infos.append(f"Interface for points {self.points}")
        infos.append(f"Number of points: {self.points.shape}")
        if self.transmission_reflection is TransmissionReflection.transmission:
            infos.append("Transmission")
        elif self.transmission_reflection is TransmissionReflection.reflection:
            infos.append(f"Reflection against {self.reflection_against}")
        if self.kind is not None:
            infos.append(f"Interface kind: {self.kind.name}")

        infos.append(f"Orientations: {self.orientations}")
        infos.append(
            f"Normals are on INC.. rays side: {self.are_normals_on_inc_rays_side}"
        )
        infos.append(
            f"Normals are on OUT. rays side: {self.are_normals_on_out_rays_side}"
        )
        infos_str = "\n".join(["    " + x if i > 0 else x for i, x in enumerate(infos)])
        return infos_str

    def __repr__(self):
        return f"<{self.__class__.__name__} for {str(self.points)} at {hex(id(self))}>"

    def reverse(self):
        """
        Returns a new Interface object where the incoming and outgoing ways are reversed.

        Returns
        -------

        """
        cls = self.__class__
        # , kind = None, transmission_reflection = None,
        # reflection_against = None,
        # are_normals_on_inc_rays_side = None, are_normals_on_out_rays_side = None
        if self.kind is None:
            rev_kind = None
        else:
            if self.transmission_reflection is None:
                raise ValueError("reverse path is ambiguous")
            elif self.transmission_reflection is TransmissionReflection.transmission:
                rev_kind = self.kind.reverse()
            elif self.transmission_reflection is TransmissionReflection.reflection:
                rev_kind = self.kind
            else:
                raise RuntimeError

        return cls(
            self.points,
            self.orientations,
            kind=rev_kind,
            transmission_reflection=self.transmission_reflection,
            reflection_against=self.reflection_against,
            are_normals_on_inc_rays_side=self.are_normals_on_out_rays_side,
            are_normals_on_out_rays_side=self.are_normals_on_inc_rays_side,
        )


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
        self.modes = tuple(helpers.parse_enum_constant(mode, Mode) for mode in modes)
        self.name = name
        self.rays = None

    @property
    def numinterfaces(self):
        return len(self.interfaces)

    @property
    def numlegs(self):
        return len(self.interfaces) - 1

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @property
    def velocities(self):
        return tuple(
            material.velocity(mode)
            for material, mode in zip(self.materials, self.modes)
        )

    def to_fermat_path(self):
        """
        Create a FermatPath from this object (low-level object for ray tracing).

        Returns
        -------

        """
        # lazy import
        from .ray import FermatPath

        return FermatPath.from_path(self)

    def reverse(self):
        """
        Path in the way round.

        Returns
        -------

        """
        cls = self.__class__
        rev_interfaces = tuple(reversed([i.reverse() for i in self.interfaces]))
        rev_materials = tuple(reversed(self.materials))
        rev_modes = tuple(reversed(self.modes))
        rev_path = cls(rev_interfaces, rev_materials, rev_modes, name=self.name)
        if self.rays is not None:
            rev_path.rays = self.rays.reverse()
        return rev_path


class Material:
    """
    Material contains the material properties.

    Attributes
    ----------
    longitudinal_vel : float
    transverse_vel : float or None
    density : float or None
    state_of_matter : StateMatter or None
    longitudinal_att : MaterialAttenuation or None
    transverse_att : MaterialAttenuation or None
    metadata : dict or None

    Examples
    --------
        >>> alu = Material(6300., 3120., 2700., 'solid', metadata={'long_name': 'Aluminium'})
        >>> water = Material(1480., None, 1000., 'liquid', metadata={'long_name': 'Water'})
    """

    def __init__(
        self,
        longitudinal_vel,
        transverse_vel=None,
        density=None,
        state_of_matter=None,
        longitudinal_att=None,
        transverse_att=None,
        metadata=None,
    ):
        longitudinal_vel = longitudinal_vel * 1.0

        if transverse_vel is not None:
            transverse_vel = transverse_vel * 1.0

        if density is not None:
            density = density * 1.0

        if state_of_matter is not None:
            state_of_matter = helpers.parse_enum_constant(state_of_matter, StateMatter)

        if metadata is None:
            metadata = {}

        self.longitudinal_vel = longitudinal_vel
        self.transverse_vel = transverse_vel
        self.density = density
        self.state_of_matter = state_of_matter
        self.longitudinal_att = longitudinal_att
        self.transverse_att = transverse_att
        self.metadata = metadata

    def __str__(self):
        name = helpers.get_name(self.metadata)

        return "{} (v_l: {} m/s, v_t: {} m/s)".format(
            name, self.longitudinal_vel, self.transverse_vel
        )

    def __repr__(self):
        name = helpers.get_name(self.metadata)

        return f"<{self.__class__.__name__}: {name} at {hex(id(self))}>"

    def velocity(self, mode):
        """
        Returns the velocity of the material for the mode 'mode'.

        Parameters
        ----------
        mode : Mode or str

        Returns
        -------
        velocitity: float

        Examples
        --------
        >>> material.velocity('L')
        # this return material.longitudinal_vel

        """
        mode = helpers.parse_enum_constant(mode, Mode)
        if mode is Mode.longitudinal:
            return self.longitudinal_vel
        elif mode is Mode.transverse:
            return self.transverse_vel
        else:
            raise ValueError(f"Don'ray know what to do with mode '{mode}'")

    def attenuation(self, mode):
        """
        Returns the material attenuation function for the mode 'mode'.

        Parameters
        ----------
        mode : Mode or str

        Returns
        -------
        attenuation: MaterialAttenuation or None

        Examples
        --------
        >>> material.attenuation('L')

        """
        mode = helpers.parse_enum_constant(mode, Mode)
        if mode is Mode.longitudinal:
            return self.longitudinal_att
        elif mode is Mode.transverse:
            return self.transverse_att
        else:
            raise ValueError(f"Don'ray know what to do with mode '{mode}'")


class ExaminationObject:
    """
    Data container for the material and the geometry of the inspected object.
    """

    def __init__(self, material, metadata=None):
        self.material = material

        if metadata is None:
            metadata = {}
        self.metadata = metadata


class BlockInImmersion(ExaminationObject):
    """
    Solid block immersed in a fluid

    Data container

    Attributes
    ----------
    block_material : Material
    material : Material
        Alias for block_material
    couplant_material : Material
    walls : list[OrientedPoints]
    wall_idxs_for_imaging : list[int]
        List of indices in `walls` which are used for simulation or imaging.
        Frontwall not required.

    """

    def __init__(
        self, block_material, couplant_material, walls, wall_idxs_for_imaging, metadata=None
    ):
        self.block_material = block_material
        self.material = block_material  # alias
        self.couplant_material = couplant_material
        self.walls = walls
        self.wall_idxs_for_imaging = wall_idxs_for_imaging

        if metadata is None:
            metadata = {}
        self.metadata = metadata


class BlockInContact(ExaminationObject):
    """
    Solid block on which the probe is in direct contact

    Parameters
    ----------
    block_material : Material
    walls : list[OrientedPoints]
    wall_idxs_for_imaging : list[int]
        List of indices in `walls` which are used for simulation or imaging.
        Frontwall not required.
    under_material : Material
    metadata : dict or None

    Attributes
    ----------
    block_material : Material
    walls : list[OrientedPoints]
    wall_idxs_for_imaging : list[int]
    under_material : Material
    metadata : dict

    """

    def __init__(
        self,
        block_material,
        walls=None,
        wall_idxs_for_imaging=None,
        under_material=None,
        metadata=None,
    ):
        self.material = block_material
        self.walls = walls
        self.wall_idxs_for_imaging = wall_idxs_for_imaging
        self.under_material = under_material
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def block_material(self):
        # alias
        return self.material


class Time:
    """Linearly spaced time vector.

    Parameters
    ----------
    samples : ndarray
    start : float
        Start time (second)
    step : float
        Time step (second)
    num : int
        Number of samples
    dtype : numpy.dtype

    Attributes
    ----------
    samples : ndarray
        Time vector
    start : float
    end : float
    step : float

    Notes
    -----
    Use len() to get the number of samples.

    """

    __slots__ = ["_samples", "_step"]

    def __init__(self, start, step, num, dtype=None):
        step = step * 1.0
        if step < 0:
            raise ValueError("'step' must be positive.")
        samples = ut.make_timevect(num, step, start, dtype)
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
        r = "{} from {:.1f} to {:.1f} µs ({} samples, step={:.2f} µs)"
        return r.format(
            self.__class__.__qualname__,
            self.start * 1e6,
            self.end * 1e6,
            len(self),
            self.step * 1e6,
        )

    def __repr__(self):
        return f"<{str(self)} at {hex(id(self))}>"

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
        (num,) = helpers.get_shape_safely(timevect, "timevect", (None,))

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
                side = "left"
            else:
                side = "right"
            imin = np.searchsorted(self.samples, tmin, side=side)
        if tmax is None:
            imax = None
        else:
            if endpoint_right:
                side = "right"
            else:
                side = "left"
            imax = np.searchsorted(self.samples, tmax, side=side)
        return slice(imin, imax)

    def closest_index(self, time):
        return np.argmin(np.abs(self.samples - time))


class View(namedtuple("View", ["tx_path", "rx_path", "name"])):
    """
    View(tx_path, rx_path, name)
    """

    __slots__ = []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def scat_key(self):
        """
        Return the scattering key (LL, LT, TL or TT) corresponding to the view.

        Returns
        -------
        str

        """
        return self.tx_path.modes[-1].key() + self.rx_path.modes[-1].key()
