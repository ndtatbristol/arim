import math
from unittest import mock

import numpy as np
import pytest

import arim.core as c
from arim import geometry as g
from arim import ut
from arim.exceptions import InvalidDimension


def test_time():
    """Test object Time"""
    start = 10e-6
    num = 1300
    step = 1e-7
    end = start + (num - 1) * step

    time = c.Time(start, step, num)
    assert len(time) == num
    assert time.start == start
    assert time.end == end
    assert time.samples[0] == start
    assert time.samples[-1] == end
    assert np.mean(np.diff(time.samples)) == step
    assert np.allclose(np.diff(time.samples), step, atol=0)

    t = start + 10 * step
    assert np.isclose(time.samples[time.closest_index(t)], t)


def test_time_window():
    start = 10.0
    num = 10
    step = 1.0
    time = c.Time(start, step, num)

    # both sides
    tmin = -666
    tmax = 666
    valid_times = time.samples[time.window(tmin, tmax)]
    ind = np.logical_and(tmin <= time.samples, time.samples <= tmax)
    valid_times_mock = time.samples[ind]
    assert np.all(valid_times == valid_times_mock)

    # both sides
    tmin = 13
    tmax = 16
    valid_times = time.samples[time.window(tmin, tmax)]
    ind = np.logical_and(tmin <= time.samples, time.samples <= tmax)
    valid_times_mock = time.samples[ind]
    assert np.all(valid_times == valid_times_mock)

    # left with endpoint
    tmin = time.samples[2]
    valid_times = time.samples[time.window(tmin)]
    valid_times_mock = time.samples[tmin <= time.samples]
    assert np.all(valid_times == valid_times_mock)

    # left without endpoint
    tmin = time.samples[2]
    valid_times = time.samples[time.window(tmin, endpoint_left=False)]
    valid_times_mock = time.samples[tmin < time.samples]
    assert np.all(valid_times == valid_times_mock)

    # right with endpoint
    tmax = time.samples[-2]
    valid_times = time.samples[time.window(tmax=tmax, endpoint_right=True)]
    valid_times_mock = time.samples[time.samples <= tmax]
    assert np.all(valid_times == valid_times_mock)

    # right with endpoint
    tmax = time.samples[-2]
    valid_times = time.samples[time.window(tmax=tmax, endpoint_right=False)]
    valid_times_mock = time.samples[time.samples < tmax]
    assert np.all(valid_times == valid_times_mock)


def test_time_from_vect():
    """Test object Time.from_vect (alternative constructor)"""
    # =========================================================================
    # Standard case
    start = 10e-6
    num = 1300
    step = 1e-7
    end = start + (num - 1) * step
    _timevect = np.linspace(start, end, num)
    timevect = _timevect.copy()

    time = c.Time.from_vect(timevect)
    assert len(time) == num
    assert time.start == start
    assert time.end == end
    assert time.samples[0] == start
    assert time.samples[-1] == end
    assert np.mean(np.diff(time.samples)) == step
    assert np.allclose(np.diff(time.samples), step, atol=0)

    # =========================================================================
    # With Nan
    timevect = _timevect.copy()
    timevect[12] = np.nan

    with pytest.raises(ValueError):
        time = c.Time.from_vect(timevect)

    # =========================================================================
    # With weird shape
    timevect = np.vstack([_timevect, _timevect])
    assert timevect.ndim == 2

    with pytest.raises(InvalidDimension):
        time = c.Time.from_vect(timevect)

    # =========================================================================
    # With jumps
    timevect = _timevect.copy()
    timevect[12] = timevect[13]

    with pytest.raises(ValueError):
        time = c.Time.from_vect(timevect)


def test_make_matrix_probe():
    # =========================================================================
    # Linear points1 along x
    numx = 5
    numy = 1
    pitch_x = 1
    pitch_y = np.nan
    frequency = 1e6

    probe = c.Probe.make_matrix_probe(numx, pitch_x, numy, pitch_y, frequency)

    locations = np.zeros((numx * numy, 3))
    locations[:, 0] = [-2, -1, 0, 1, 2]
    assert probe.metadata["probe_type"] == "linear"
    assert probe.metadata["numx"] == numx
    assert probe.metadata["numy"] == numy
    assert probe.metadata["pitch_x"] == pitch_x
    assert np.isnan(probe.metadata["pitch_y"])
    assert np.allclose(probe.locations, locations)
    assert probe.frequency == frequency

    # =========================================================================
    # Linear points1 along y (and negative pitch)
    numx = 1
    numy = 5
    pitch_x = 666  # unused
    pitch_y = -1
    frequency = 1e6

    probe = c.Probe.make_matrix_probe(numx, pitch_x, numy, pitch_y, frequency)

    locations = np.zeros((numx * numy, 3))
    locations[:, 1] = [2, 1, 0, -1, -2]
    assert probe.metadata["probe_type"] == "linear"
    assert probe.metadata["numx"] == numx
    assert probe.metadata["numy"] == numy
    assert np.isnan(probe.metadata["pitch_x"])
    assert probe.metadata["pitch_y"] == pitch_y
    assert np.allclose(probe.locations, locations)
    assert probe.frequency == frequency

    # =========================================================================
    # Single element
    numx = 1
    numy = 1
    pitch_x = 666
    pitch_y = 1234
    frequency = 1e6

    probe = c.Probe.make_matrix_probe(numx, pitch_x, numy, pitch_y, frequency)

    locations = np.zeros((numx * numy, 3))

    assert probe.metadata["probe_type"] == "single"
    assert probe.metadata["numx"] == numx
    assert probe.metadata["numy"] == numy
    assert np.isnan(probe.metadata["pitch_x"])
    assert np.isnan(probe.metadata["pitch_y"])
    assert np.allclose(probe.locations, locations)
    assert probe.frequency == frequency

    # =========================================================================
    # Matrix points1
    numx = 2
    numy = 3
    pitch_x = -1
    pitch_y = 2
    frequency = 1e6

    probe = c.Probe.make_matrix_probe(numx, pitch_x, numy, pitch_y, frequency)

    locations = np.zeros((numx * numy, 3))
    locations[:, 0] = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
    locations[:, 1] = [-2, -2, 0, 0, 2, 2]

    assert probe.metadata["probe_type"] == "matrix"
    assert probe.metadata["numx"] == numx
    assert probe.metadata["numy"] == numy
    assert probe.metadata["pitch_x"] == pitch_x
    assert probe.metadata["pitch_y"] == pitch_y
    assert np.allclose(probe.locations, locations)
    assert probe.frequency == frequency


@pytest.fixture(scope="module")
def probe():
    numelements = 4
    locx = np.arange(numelements, dtype=np.float) * 0.1e-3
    locx -= locx.mean()
    locy = np.zeros(numelements, dtype=np.float)
    locz = np.zeros(numelements, dtype=np.float)
    locations = g.Points.from_xyz(locx, locy, locz)
    x = np.arange(numelements) * 0.1e-3

    frequency = 1e6

    metadata = dict(short_name="test_linear16", version=0)
    return c.Probe(locations, frequency, metadata=metadata)


@pytest.fixture(scope="module")
def examination_object():
    material = c.Material(6300, 3100, 2700, metadata=dict(long_name="Aluminium"))
    return c.ExaminationObject(material)


@pytest.fixture(scope="module")
def water():
    return c.Material(
        1400.0,
        density=1000.0,
        state_of_matter="liquid",
        metadata={"long_name": "Water"},
    )


@pytest.fixture(scope="module")
def aluminium():
    v_longi = 6320.0
    v_transverse = 3130.0

    return c.Material(
        v_longi,
        v_transverse,
        density=2700.0,
        state_of_matter="solid",
        metadata={"long_name": "Aluminium"},
    )


@pytest.fixture(scope="module")
def frame(probe, examination_object):
    tx, rx = ut.hmc(probe.numelements)
    metadata = dict(capture_method=c.CaptureMethod.fmc)

    time = c.Time(start=5e-6, step=1 / 25e6, num=1000)

    timetraces = np.zeros((len(tx), len(time)))
    timetraces[...] = (tx * 1000 + rx)[..., np.newaxis]

    return c.Frame(
        timetraces, time, tx, rx, probe, examination_object, metadata=metadata
    )


class TestFrame:
    def test_get_timetrace(self, frame: c.Frame):
        tx = 1
        rx = 2
        scan = frame.get_timetrace(tx, rx)
        assert scan.shape == (frame.numsamples,)
        assert scan[0] == tx * 1000 + rx

        with pytest.raises(IndexError):
            frame.get_timetrace(1000, 0)

    def test_expand_frame_assuming_reciprocity_hmc(self, frame):
        assert not frame.is_complete_assuming_reciprocity()
        fmc_frame = frame.expand_frame_assuming_reciprocity()
        assert fmc_frame.is_complete_assuming_reciprocity()

        expected_tx, expected_rx = ut.fmc(frame.probe.numelements)
        np.testing.assert_array_equal(fmc_frame.tx, expected_tx)
        np.testing.assert_array_equal(fmc_frame.rx, expected_rx)

        for tx, rx in zip(expected_tx, expected_rx):
            if tx <= rx:
                orig_scan = fmc_frame.get_timetrace(tx, rx)
            else:
                orig_scan = fmc_frame.get_timetrace(rx, tx)
            expanded_scan = fmc_frame.get_timetrace(tx, rx)
            np.testing.assert_allclose(orig_scan, expanded_scan)

        fmc_frame2 = fmc_frame.expand_frame_assuming_reciprocity()
        np.testing.assert_array_equal(fmc_frame2.tx, fmc_frame.tx)
        np.testing.assert_array_equal(fmc_frame2.rx, fmc_frame.rx)
        np.testing.assert_array_equal(fmc_frame2.timetraces, fmc_frame.timetraces)

    def test_apply_filter(self, frame):
        filt = lambda x: -x
        frame2 = frame.apply_filter(filt)
        np.testing.assert_array_equal(frame2.tx, frame.tx)
        np.testing.assert_array_equal(frame2.rx, frame.rx)
        np.testing.assert_allclose(frame2.timetraces, -frame.timetraces)

    def test_subframe(self, frame):
        # start with a HMC with 4 elements
        # Retain only element 0, 1 and 3
        frame_a = frame.subframe(timetraces_idx=[0, 1, 3, 4, 6, 9])
        frame_b = frame.subframe_from_probe_elements(
            elements_idx=[0, 1, 3], make_subprobe=False
        )

        # frame_a and frame_b should be the same
        assert frame_a.numtimetraces == 6
        assert frame_b.numtimetraces == 6
        np.testing.assert_allclose(frame_a.timetraces, frame_b.timetraces)
        np.testing.assert_allclose(frame_a.tx, frame_b.tx)
        np.testing.assert_allclose(frame_a.rx, frame_b.rx)
        assert frame_a.probe is frame.probe
        assert frame_b.probe is frame.probe

        frame_c = frame.subframe_from_probe_elements(
            elements_idx=[0, 1, 3], make_subprobe=True
        )
        assert frame_c.numtimetraces == 6
        np.testing.assert_allclose(frame_c.timetraces, frame_a.timetraces)
        assert frame_c.probe.numelements == 3


class TestProbe:
    def test_probe(self, probe):
        str(probe)
        repr(probe)

        assert probe.frequency is not None
        assert isinstance(probe.locations, g.Points)
        assert isinstance(probe.orientations, g.Points) or probe.orientations is None
        assert isinstance(probe.dimensions, g.Points) or probe.dimensions is None
        assert isinstance(probe.shapes, np.ndarray) or probe.shapes is None
        assert (
            isinstance(probe.dead_elements, np.ndarray) or probe.dead_elements is None
        )

    def test_linear_probe(self):
        linear_probe = self.linear_probe()
        self.test_probe(linear_probe)

    def test_tolerant_probe(self):
        probe_bak = self.linear_probe()

        # that's a ndarray:
        locations = probe_bak.locations.coords.copy()

        # that's lists:
        orientations = [
            probe_bak.orientations.x[0],
            probe_bak.orientations.y[0],
            probe_bak.orientations.z[0],
        ]
        dimensions = [
            probe_bak.dimensions.x[0],
            probe_bak.dimensions.y[0],
            probe_bak.dimensions.z[0],
        ]

        # that's just one value:
        shapes = c.ElementShape.rectangular
        dead_elements = False

        bandwidth = probe_bak.bandwidth
        frequency = probe_bak.frequency

        pcs = probe_bak.pcs.copy()

        # I love my new API:
        probe = c.Probe(
            locations,
            frequency,
            dimensions,
            orientations,
            shapes,
            dead_elements,
            bandwidth,
            pcs=pcs,
        )

        self.test_probe(probe)
        self.assert_probe_equal_in_gcs(probe, probe_bak)

    def test_tolerant_linear_probe(self):
        probe_bak = self.linear_probe()

        # that's lists:
        orientations = [
            probe_bak.orientations.x[0],
            probe_bak.orientations.y[0],
            probe_bak.orientations.z[0],
        ]
        dimensions = [
            probe_bak.dimensions.x[0],
            probe_bak.dimensions.y[0],
            probe_bak.dimensions.z[0],
        ]

        # that's just one value:
        shapes = c.ElementShape.rectangular
        dead_elements = False

        probe = c.Probe.make_matrix_probe(
            pitch_x=1e-3,
            numx=10,
            pitch_y=np.nan,
            numy=1,
            frequency=1e6,
            shapes=shapes,
            orientations=orientations,
            dimensions=dimensions,
            bandwidth=0.5e6,
            dead_elements=dead_elements,
        )
        self.test_probe(probe)
        self.assert_probe_equal_in_gcs(probe, probe_bak)

    def linear_probe(self):
        numelements = 10

        dimensions = g.Points.from_xyz(
            np.full(numelements, 0.8e-3),
            np.full(numelements, 30e-3),
            np.zeros(numelements, dtype=np.float),
        )
        orientations = g.Points.from_xyz(
            np.zeros(numelements, dtype=np.float),
            np.zeros(numelements, dtype=np.float),
            np.ones(numelements, dtype=np.float),
        )
        shapes = np.array(numelements * [c.ElementShape.rectangular], dtype="O")
        dead_elements = np.zeros((numelements,), dtype=np.bool)
        probe = c.Probe.make_matrix_probe(
            pitch_x=1e-3,
            numx=numelements,
            pitch_y=np.nan,
            numy=1,
            frequency=1e6,
            shapes=shapes,
            orientations=orientations,
            dimensions=dimensions,
            bandwidth=0.5e6,
            dead_elements=dead_elements,
        )
        return probe

    def test_subprobe(self):
        # 10 elements:
        probe = self.linear_probe()

        # elements 0, 2, 4, 6, 8:
        subprobe = probe.subprobe(np.s_[::2])
        assert subprobe.numelements == 5
        np.testing.assert_allclose(subprobe.locations[1], probe.locations[2])

        subprobe = probe.subprobe([2])
        assert subprobe.numelements == 1

        # duplicate elements and change order
        subprobe = probe.subprobe([2, 2, 1])
        assert subprobe.numelements == 3
        np.testing.assert_allclose(subprobe.locations[0], probe.locations[2])
        np.testing.assert_allclose(subprobe.locations[1], probe.locations[2])
        np.testing.assert_allclose(subprobe.locations[2], probe.locations[1])

    def test_move_probe(self):
        probe = self.linear_probe()
        probe_bak = self.linear_probe()

        # define a nasty isometry:
        centre = np.array((1.1, 1.2, 1.3))
        rotation = g.rotation_matrix_ypr(0.5, -0.6, 0.7)
        translation = np.array((66.0, -77.0, 0.0))

        # rotate!
        probe = probe.rotate(rotation, centre)
        self.assert_probe_equal_in_pcs(probe, probe_bak)

        # translate!
        probe = probe.translate(translation)
        self.assert_probe_equal_in_pcs(probe, probe_bak)

    def test_reset_location(self):
        probe = self.linear_probe()
        probe2 = self.linear_probe()

        # define a nasty isometry:
        centre = np.array((1.1, 1.2, 1.3))
        rotation = g.rotation_matrix_ypr(0.5, -0.6, 0.7)
        translation = np.array((66.0, -77.0, 0.0))
        probe = probe.rotate(rotation, centre)
        probe = probe.translate(translation)

        # define a second isometry:
        centre = np.array((4.0, 5.0, 6.0))
        rotation = g.rotation_matrix_ypr(0.1, -0.1, 0.3)
        translation = np.array((8.0, 9.0, -10.0))
        probe2.rotate(rotation, centre)
        probe2.translate(translation)

        # check they are now not equal:
        with pytest.raises(AssertionError):
            assert not np.testing.assert_allclose(probe.locations, probe2.locations)
        assert not probe.pcs.isclose(probe2.pcs)

        probe.reset_position()
        probe2.reset_position()
        assert g.GCS.isclose(probe.pcs)
        assert g.GCS.isclose(probe2.pcs)
        self.assert_probe_equal_in_gcs(probe, probe2)
        self.assert_probe_equal_in_pcs(probe, probe2)

    def test_reference_element(self):
        """
        Test Probe.set_reference_element, Probe.flip_probe_around_axis_Oz, Probe.translate_to_point_0
        """
        allclose_kwargs = dict(rtol=0, atol=1e-9)
        probe = self.linear_probe()
        probe.set_reference_element("first")
        probe.translate_to_point_O()
        np.testing.assert_allclose(probe.locations.y, 0, **allclose_kwargs)
        np.testing.assert_allclose(probe.locations.z, 0, **allclose_kwargs)
        np.testing.assert_allclose(
            probe.locations[0], (0.0, 0.0, 0.0), **allclose_kwargs
        )
        np.testing.assert_allclose(probe.locations[-1], (9e-3, 0, 0), **allclose_kwargs)

        probe = self.linear_probe()
        probe.set_reference_element("last")
        probe.translate_to_point_O()
        np.testing.assert_allclose(probe.locations.y, 0.0, **allclose_kwargs)
        np.testing.assert_allclose(probe.locations.z, 0.0, **allclose_kwargs)
        np.testing.assert_allclose(probe.locations[0], (-9e-3, 0, 0), **allclose_kwargs)
        np.testing.assert_allclose(
            probe.locations[-1], (0.0, 0.0, 0.0), **allclose_kwargs
        )

        probe = self.linear_probe()
        probe.set_reference_element("last")
        probe.flip_probe_around_axis_Oz()
        probe.translate_to_point_O()
        np.testing.assert_allclose(probe.locations.y, 0.0, **allclose_kwargs)
        np.testing.assert_allclose(probe.locations.z, 0.0, **allclose_kwargs)
        np.testing.assert_allclose(probe.locations[0], (9e-3, 0, 0), **allclose_kwargs)
        np.testing.assert_allclose(
            probe.locations[-1], (0.0, 0.0, 0.0), **allclose_kwargs
        )

        probe = self.linear_probe()
        probe.set_reference_element("mean")
        probe.translate_to_point_O()
        np.testing.assert_allclose(probe.locations.z, 0.0, **allclose_kwargs)
        np.testing.assert_allclose(probe.locations.y, 0.0, **allclose_kwargs)
        np.testing.assert_allclose(
            probe.locations[0], (-4.5e-3, 0.0, 0.0), **allclose_kwargs
        )
        np.testing.assert_allclose(
            probe.locations[-1], (+4.5e-3, 0, 0), **allclose_kwargs
        )

        probe.flip_probe_around_axis_Oz()
        np.testing.assert_allclose(
            probe.locations[0], (+4.5e-3, 0.0, 0.0), **allclose_kwargs
        )
        np.testing.assert_allclose(
            probe.locations[-1], (-4.5e-3, 0, 0), **allclose_kwargs
        )

    @staticmethod
    def assert_probe_equal_in_pcs(probe1, probe2):
        assert probe1.locations_pcs.allclose(probe2.locations_pcs)
        assert probe1.dimensions.allclose(probe2.dimensions)
        assert probe1.orientations_pcs.allclose(probe2.orientations_pcs)
        assert np.all(probe1.dead_elements == probe2.dead_elements)
        assert np.all(probe1.shapes == probe2.shapes)

    @staticmethod
    def assert_probe_equal_in_gcs(probe1, probe2):
        assert probe1.pcs.isclose(probe2.pcs)
        assert probe1.locations.allclose(probe2.locations)
        assert probe1.dimensions.allclose(probe2.dimensions)
        assert probe1.orientations.allclose(probe2.orientations)
        assert np.all(probe1.dead_elements == probe2.dead_elements)
        assert np.all(probe1.shapes == probe2.shapes)


def test_material():
    mat = c.Material(1)
    str(mat)

    mat = c.Material(1, 2, 3, metadata={"short_name": "test_material"})
    str(mat)
    assert math.isclose(mat.longitudinal_vel, 1)
    assert math.isclose(mat.transverse_vel, 2)
    assert math.isclose(mat.density, 3)
    assert mat.state_of_matter is None
    assert mat.metadata["short_name"] == "test_material"

    mat = c.Material(
        1,
        2,
        state_of_matter=c.StateMatter.liquid,
        metadata={"short_name": "test_material"},
    )
    assert mat.state_of_matter is c.StateMatter.liquid
    mat = c.Material(
        1, 2, state_of_matter="liquid", metadata={"short_name": "test_material"}
    )
    assert mat.state_of_matter is c.StateMatter.liquid

    # test method 'velocity':
    assert math.isclose(mat.velocity("longitudinal"), mat.longitudinal_vel)
    assert math.isclose(mat.velocity(c.Mode.L), mat.longitudinal_vel)
    assert math.isclose(mat.velocity("transverse"), mat.transverse_vel)
    assert math.isclose(mat.velocity(c.Mode.T), mat.transverse_vel)


def test_mode():
    assert c.Mode.L is c.Mode.longitudinal
    assert c.Mode.T is c.Mode.transverse
    assert c.Mode.L.reverse() is c.Mode.T
    assert c.Mode.T.reverse() is c.Mode.L

    assert c.Mode.L.key() == "L"
    assert c.Mode.T.key() == "T"


class TestInterface:
    def test_interface_probe(self):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), "coordinate system")

        interface = c.Interface(
            points,
            orientations,
            are_normals_on_inc_rays_side=None,
            are_normals_on_out_rays_side=True,
        )

        assert interface.points is points
        assert np.allclose(
            interface.orientations[np.newaxis, ...], interface.orientations[...]
        )
        str(interface)
        repr(interface)

    def test_interface_grid(self):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), "coordinate system")

        interface = c.Interface(
            points,
            orientations,
            are_normals_on_inc_rays_side=True,
            are_normals_on_out_rays_side=None,
        )

        with pytest.raises(ValueError):
            c.Interface(
                points,
                orientations,
                are_normals_on_inc_rays_side=True,
                are_normals_on_out_rays_side=None,
                reflection_against=water,
            )

        assert interface.points is points
        assert np.allclose(
            interface.orientations[np.newaxis, ...], interface.orientations[...]
        )
        assert interface.transmission_reflection is None
        str(interface)
        repr(interface)

    def test_interface_transmission(self, water):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), "coordinate system")

        interface = c.Interface(
            points,
            orientations,
            transmission_reflection="transmission",
            kind="fluid_solid",
            are_normals_on_inc_rays_side=True,
            are_normals_on_out_rays_side=None,
        )

        with pytest.raises(ValueError):
            c.Interface(
                points,
                orientations,
                transmission_reflection="transmission",
                kind="fluid_solid",
                are_normals_on_inc_rays_side=True,
                are_normals_on_out_rays_side=None,
                reflection_against=water,
            )

        assert interface.points is points
        assert np.allclose(
            interface.orientations[np.newaxis, ...], interface.orientations[...]
        )
        assert (
            interface.transmission_reflection is c.TransmissionReflection.transmission
        )
        str(interface)
        repr(interface)

    def test_interface_reflection(self):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), "coordinate system")

        interface = c.Interface(
            points,
            orientations,
            transmission_reflection="reflection",
            kind="fluid_solid",
            are_normals_on_inc_rays_side=True,
            are_normals_on_out_rays_side=None,
            reflection_against=water,
        )

        with pytest.raises(ValueError):
            c.Interface(
                points,
                orientations,
                transmission_reflection="reflection",
                kind="fluid_solid",
                are_normals_on_inc_rays_side=True,
                are_normals_on_out_rays_side=None,
                reflection_against=None,
            )

        assert interface.points is points
        assert np.allclose(
            interface.orientations[np.newaxis, ...], interface.orientations[...]
        )
        assert interface.transmission_reflection is c.TransmissionReflection.reflection
        assert interface.reflection_against is water
        str(interface)
        repr(interface)


class TestPath:
    def test_path_velocities(self, water, aluminium):
        probe_interface = mock.Mock()
        frontwall_interface = mock.Mock()
        backwall_interface = mock.Mock()
        grid_interface = mock.Mock()
        path = c.Path(
            interfaces=(
                probe_interface,
                frontwall_interface,
                backwall_interface,
                grid_interface,
            ),
            materials=(water, aluminium, aluminium),
            modes=("L", "L", "T"),
            name="LT",
        )

        assert path.velocities == (
            water.longitudinal_vel,
            aluminium.longitudinal_vel,
            aluminium.transverse_vel,
        )


def test_interface_kind():
    a = c.InterfaceKind.fluid_solid
    assert a.reverse() is c.InterfaceKind.solid_fluid


@pytest.mark.parametrize(
    "mat_att_args,expected",
    [(("constant", 777.0), 777.0), (("polynomial", (777.0, 0.0, 0.02)), 779.0)],
)
def test_material_attenuation_factory(mat_att_args, expected):
    mat_att_func = c.material_attenuation_factory(*mat_att_args)
    frequency = 10e6
    fval = mat_att_func(frequency)
    np.testing.assert_allclose(fval, expected)

    # test 1d arrays are accepted:
    frequencies = np.linspace(0.0, 10e6)
    fval = mat_att_func(frequencies)
    assert fval.shape == frequencies.shape
    np.testing.assert_allclose(fval[-1], expected)
