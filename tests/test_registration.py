import numpy as np
import pytest
import arim.registration as reg
from unittest.mock import Mock

import arim
import arim.geometry as g
from arim import Time, ExaminationObject, Material, Probe, Frame

_MOVE_PROBE_ON_OXY_DATA = [
    ((0., 0., 0.), (5., 0., 0.), 6., 10.),
    ((0., 0., 0.), (5., 0., 0.), 10., 6.),
    ((0., 0., 0.), (-5., 0., 0.), 6., 10.),
    ((0., 0., 0.), (-5., 0., 0.), 10., 6.),
]


def test_manual_registration():
    from tests.test_core import TestProbe
    probe = TestProbe().linear_probe()
    frame = Mock(spec=['probe'])
    frame.probe = probe

    theta = np.deg2rad(10)
    standoff = 15e-3
    reg.manual_registration(frame, theta, standoff)
    np.testing.assert_allclose(frame.probe.pcs.origin, (0, 0, standoff))
    np.testing.assert_allclose(frame.probe.pcs.i_hat, (np.cos(theta), 0, -np.sin(theta)))
    np.testing.assert_allclose(frame.probe.pcs.k_hat, (np.sin(theta), 0, np.cos(theta)))


@pytest.mark.parametrize("A,B,dA,dB", _MOVE_PROBE_ON_OXY_DATA)
def test_move_probe_over_flat_surface_ideal(A, B, dA, dB):
    """Test move_probe_over_flat_surface() with 2 elements"""
    A = np.asarray(A, dtype=np.float)
    B = np.asarray(B, dtype=np.float)

    # Setup: a 2-element points1
    locations = g.Points(np.array([A, B]))  # locations in PCS
    probe = Probe(locations, 1e6)  # NB: assume PCS=GCS at this point

    tx = np.array([0, 0, 1])
    rx = np.array([0, 1, 1])

    distance_to_surface = np.array([dA, np.nan, dB])

    time = Time(0., 1.0, 50)
    scanlines = np.zeros((len(tx), len(time)))

    frame = Frame(scanlines, time, tx, rx, probe, ExaminationObject(Material(1.0)))

    frame, iso = reg.move_probe_over_flat_surface(frame, distance_to_surface, full_output=True)
    new_locations = frame.probe.locations

    # Are the locations in PCS unchanged?
    assert np.allclose(frame.probe.locations_pcs, locations)

    # Right distance to the plane Oxy (NB: z<0):
    assert np.isclose(new_locations.z[0], -dA)
    assert np.isclose(new_locations.z[1], -dB)

    # Elements are in plane y=0
    assert np.isclose(new_locations.y[0], 0.)
    assert np.isclose(new_locations.y[1], 0.)

    # Is element A(0., 0., 0.) now in (0., 0., z)?
    assert np.allclose(new_locations[0], (0., 0., iso.z_o))

    # the elements are still in the right distance
    dAB = g.norm2(*(locations[0] - locations[1]))
    dApBp = g.norm2(*(new_locations[0] - new_locations[1]))
    assert np.isclose(dAB, dApBp)

    # Is the returned theta right?
    # Remark: (dB-dB)/(xB-xA) has the sign of theta, then
    # (zA-zB)/(xB-xA) has the sign of -theta because zA=-dA and zB=-dB
    theta = -np.arctan((new_locations.z[1] - new_locations.z[0]) /
                       (new_locations.x[1] - new_locations.x[0]))
    assert np.isclose(theta, iso.theta)


@pytest.mark.parametrize("theta_deg", [30, -30])
def test_move_probe_over_flat_surface_real(theta_deg):
    """Test move_probe_on_Oxy() with a 10 element linear points1"""
    standoff = -10.

    numelements = 10

    # Setup: a 2-element points1
    probe = Probe.make_matrix_probe(numelements, 0.1, 1, 0., 1e6)

    locations_pcs = probe.locations

    # The point O(0., 0., 0.) is the 'centre' of the points1 (geometrical centre)
    assert np.allclose(np.mean(locations_pcs, axis=0), 0.)

    # rotate and shift points1:
    locations_gcs = locations_pcs @ g.rotation_matrix_y(np.deg2rad(theta_deg)).T
    locations_gcs[:, 2] += standoff

    # empty fmc data
    tx, rx = arim.ut.fmc(numelements)
    time = Time(0., 1.0, 50)
    scanlines = np.zeros((len(tx), len(time)))
    frame = Frame(scanlines, time, tx, rx, probe, ExaminationObject(Material(1.0)))

    # Distance to surface: orthogonal projection on Oxy
    distance_to_surface = np.full(len(tx), np.nan)
    distance_to_surface[tx == rx] = -locations_gcs[tx[tx == rx], 2]

    frame = reg.move_probe_over_flat_surface(frame, distance_to_surface)
    out_locations = frame.probe.locations

    assert np.allclose(out_locations, locations_gcs)


def test_detect_surface_from_extrema():
    probe = Probe.make_matrix_probe(3, 1.0, 1.0, 0.0, 1e6)

    time = Time(10., 1.0, 50)

    tx = np.array([0, 0, 1, 1, 2, 2])
    rx = np.array([0, 1, 1, 0, 2, 0])

    scanlines = np.random.uniform(high=10.0, size=(len(tx), len(time)))

    times_to_surface_expected = np.array([25.0, 26.0, 27.0, 28.0, 29.0, 30.0])

    for (i, t) in enumerate(times_to_surface_expected):
        scanlines[i, time.closest_index(t)] = t

    frame = Frame(scanlines, time, tx, rx, probe, ExaminationObject(Material(1.0)))

    times_to_surface = reg.detect_surface_from_extrema(frame)

    assert np.allclose(times_to_surface, times_to_surface_expected)
