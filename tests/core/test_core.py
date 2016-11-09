import math

import numpy as np
import pytest
from unittest import mock

import arim
from arim import CaptureMethod
from arim import core as c
from arim import utils
from arim import geometry as g
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
    assert probe.metadata['probe_type'] == 'linear'
    assert probe.metadata['numx'] == numx
    assert probe.metadata['numy'] == numy
    assert probe.metadata['pitch_x'] == pitch_x
    assert np.isnan(probe.metadata['pitch_y'])
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
    assert probe.metadata['probe_type'] == 'linear'
    assert probe.metadata['numx'] == numx
    assert probe.metadata['numy'] == numy
    assert np.isnan(probe.metadata['pitch_x'])
    assert probe.metadata['pitch_y'] == pitch_y
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

    assert probe.metadata['probe_type'] == 'single'
    assert probe.metadata['numx'] == numx
    assert probe.metadata['numy'] == numy
    assert np.isnan(probe.metadata['pitch_x'])
    assert np.isnan(probe.metadata['pitch_y'])
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

    assert probe.metadata['probe_type'] == 'matrix'
    assert probe.metadata['numx'] == numx
    assert probe.metadata['numy'] == numy
    assert probe.metadata['pitch_x'] == pitch_x
    assert probe.metadata['pitch_y'] == pitch_y
    assert np.allclose(probe.locations, locations)
    assert probe.frequency == frequency


@pytest.fixture(scope='module')
def probe():
    numelements = 16
    locx = np.arange(numelements, dtype=np.float) * 0.1e-3
    locx -= locx.mean()
    locy = np.zeros(numelements, dtype=np.float)
    locz = np.zeros(numelements, dtype=np.float)
    locations = g.Points.from_xyz(locx, locy, locz)
    x = np.arange(numelements) * 0.1e-3

    frequency = 1e6

    metadata = dict(short_name='test_linear16', version=0)
    return c.Probe(locations, frequency, metadata=metadata)


@pytest.fixture(scope='module')
def examination_object():
    material = c.Material(6300, 3100, 2700, metadata=dict(long_name='Aluminium'))
    return c.InfiniteMedium(material)


@pytest.fixture(scope="module")
def water():
    return c.Material(1400., density=1000., state_of_matter='liquid', metadata={'long_name': 'Water'})


@pytest.fixture(scope="module")
def aluminium():
    v_longi = 6320.
    v_transverse = 3130.

    return c.Material(v_longi, v_transverse, density=2700., state_of_matter='solid',
                      metadata={'long_name': 'Aluminium'})


@pytest.fixture(scope='module')
def frame(probe, examination_object):
    tx, rx = utils.hmc(probe.numelements)
    metadata = dict(capture_method=CaptureMethod.fmc)

    time = c.Time(start=5e-6, step=1 / 25e6, num=1000)

    # generate some fake signals
    f1 = probe.frequency
    f2 = probe.frequency * 1.2
    f3 = probe.frequency * 8
    scan = np.sin(2 * np.pi * time.samples * f1) + np.sin(2 * np.pi * time.samples * f2) + np.sin(
        2 * np.pi * time.samples * f3)

    scanlines = np.zeros((len(tx), len(time)))
    scanlines[...] = scan

    return c.Frame(scanlines, time, tx, rx, probe, examination_object, metadata=metadata)


class TestFrame:
    def test_get_scanline(self, frame: c.Frame):
        tx = 1
        rx = 2
        scan = frame.get_scanline(tx, rx)
        assert scan.shape == (frame.numsamples,)

        scan = frame.get_scanline(tx, rx, use_raw=True)
        assert scan.shape == (frame.numsamples,)


class TestProbe:
    def test_probe(self, probe):
        str(probe)
        repr(probe)

    def linear_probe(self):
        numelements = 10

        dimensions = g.Points.from_xyz(np.full(numelements, 0.8e-3),
                                       np.full(numelements, 30e-3),
                                       np.zeros(numelements, dtype=np.float))
        orientations = g.Points.from_xyz(np.zeros(numelements, dtype=np.float),
                                         np.zeros(numelements, dtype=np.float),
                                         np.ones(numelements, dtype=np.float))
        shapes = np.array(numelements * [c.ElementShape.rectangular], dtype='O')
        dead_elements = np.zeros((numelements,), dtype=np.bool)
        probe = c.Probe.make_matrix_probe(pitch_x=1e-3, numx=numelements, pitch_y=np.nan, numy=1,
                                          frequency=1e6, shapes=shapes,
                                          orientations=orientations, dimensions=dimensions, bandwidth=0.5e6,
                                          dead_elements=dead_elements)
        return probe

    def test_move_probe(self):
        probe = self.linear_probe()
        probe_bak = self.linear_probe()

        # define a nasty isometry:
        centre = np.array((1.1, 1.2, 1.3))
        rotation = g.rotation_matrix_ypr(0.5, -0.6, 0.7)
        translation = np.array((66., -77., 0.))

        # rotate!
        probe = probe.rotate(rotation, centre)
        self.assert_probe_equal_in_pcs(probe, probe_bak)

        # translate!
        probe = probe.translate(translation)
        self.assert_probe_equal_in_pcs(probe, probe_bak)

    @staticmethod
    def assert_probe_equal_in_pcs(probe1, probe2):
        assert probe1.locations_pcs.allclose(probe2.locations_pcs)
        assert probe1.dimensions.allclose(probe2.dimensions)
        assert probe1.orientations_pcs.allclose(probe2.orientations_pcs)
        assert np.all(probe1.dead_elements == probe2.dead_elements)
        assert np.all(probe1.shapes == probe2.shapes)


def test_material():
    mat = c.Material(1)
    str(mat)

    mat = c.Material(1, 2, 3, metadata={'short_name': 'test_material'})
    str(mat)
    assert math.isclose(mat.longitudinal_vel, 1)
    assert math.isclose(mat.transverse_vel, 2)
    assert math.isclose(mat.density, 3)
    assert mat.state_of_matter is None
    assert mat.metadata['short_name'] == 'test_material'

    mat = c.Material(1, 2, state_of_matter=arim.StateMatter.liquid, metadata={'short_name': 'test_material'})
    assert mat.state_of_matter is arim.StateMatter.liquid
    mat = c.Material(1, 2, state_of_matter='liquid', metadata={'short_name': 'test_material'})
    assert mat.state_of_matter is arim.StateMatter.liquid

    # test method 'velocity':
    assert math.isclose(mat.velocity('longitudinal'), mat.longitudinal_vel)
    assert math.isclose(mat.velocity(arim.Mode.L), mat.longitudinal_vel)
    assert math.isclose(mat.velocity('transverse'), mat.transverse_vel)
    assert math.isclose(mat.velocity(arim.Mode.T), mat.transverse_vel)


def test_mode():
    assert c.Mode.L is c.Mode.longitudinal
    assert c.Mode.T is c.Mode.transverse


class TestInterface:
    def test_interface_probe(self):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), 'coordinate system')

        interface = c.Interface(points, orientations, are_normals_on_inc_rays_side=None,
                                are_normals_on_out_rays_side=True)

        assert interface.points is points
        assert interface.orientations is orientations
        str(interface)
        repr(interface)

    def test_interface_grid(self):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), 'coordinate system')

        interface = c.Interface(points, orientations, are_normals_on_inc_rays_side=True,
                                are_normals_on_out_rays_side=None)

        with pytest.raises(ValueError):
            c.Interface(points, orientations, are_normals_on_inc_rays_side=True, are_normals_on_out_rays_side=None,
                        reflection_against=water)

        assert interface.points is points
        assert interface.orientations is orientations
        assert interface.transmission_reflection is None
        str(interface)
        repr(interface)

    def test_interface_transmission(self, water):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), 'coordinate system')

        interface = c.Interface(points, orientations, transmission_reflection='transmission', kind='fluid_solid',
                                are_normals_on_inc_rays_side=True, are_normals_on_out_rays_side=None)

        with pytest.raises(ValueError):
            c.Interface(points, orientations, transmission_reflection='transmission', kind='fluid_solid',
                        are_normals_on_inc_rays_side=True, are_normals_on_out_rays_side=None,
                        reflection_against=water)

        assert interface.points is points
        assert interface.orientations is orientations
        assert interface.transmission_reflection is c.TransmissionReflection.transmission
        str(interface)
        repr(interface)

    def test_interface_reflection(self):
        n = 10
        points = g.Points(np.random.uniform(size=(n, 3)))
        orientations = g.Points(np.eye(3), 'coordinate system')

        interface = c.Interface(points, orientations, transmission_reflection='reflection', kind='fluid_solid',
                                are_normals_on_inc_rays_side=True, are_normals_on_out_rays_side=None,
                                reflection_against=water)

        with pytest.raises(ValueError):
            c.Interface(points, orientations, transmission_reflection='reflection', kind='fluid_solid',
                        are_normals_on_inc_rays_side=True, are_normals_on_out_rays_side=None,
                        reflection_against=None)

        assert interface.points is points
        assert interface.orientations is orientations
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
            interfaces=(probe_interface, frontwall_interface, backwall_interface, grid_interface),
            materials=(water, aluminium, aluminium),
            modes=('L', 'L', 'T'),
            name='LT')

        assert path.velocities == (water.longitudinal_vel, aluminium.longitudinal_vel, aluminium.transverse_vel)
