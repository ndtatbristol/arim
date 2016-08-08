import math

import numpy as np

import arim.geometry as g
from arim import Probe, InfiniteMedium, Material, Time, Frame, FocalLaw
from arim import im


def make_delay_and_sum_case1():
    """
    2 elements (x=0, y=0, z=0) and (x=1, y=0, z=0)
    1 reflector at (x=0, y=0, z=2)
    Speed: 10


    Parameters
    ----------
    self

    Returns
    -------

    """
    # very basic points1:
    locations = g.Points(np.array([(0, 0, 0), (1., 0, 0)], dtype=np.float))
    frequency = 1e6
    probe = Probe(locations, frequency)

    # examination object:
    vel = 10.
    material = Material(vel)
    examination_object = InfiniteMedium(material)

    # scanlines
    time = Time(start=0.35, step=0.001, num=100)

    numscanlines = 3
    tx = np.array([0, 0, 1], dtype=np.int)
    rx = np.array([0, 1, 1], dtype=np.int)

    # Model a reflector at distance 2 from the first element, and sqrt(5) from the second
    rt5 = math.sqrt(5)
    times_of_flights = np.array([4.0 / vel, (rt5 + 2) / vel, (2 * rt5) / vel])
    scanlines = np.zeros((numscanlines, len(time)), dtype=np.float)

    for (i, val) in enumerate(times_of_flights):
        closest = np.abs(time.samples - val).argmin()
        scanlines[i, closest] = 5

    lookup_times_tx = (times_of_flights[tx == rx] / 2).reshape((1, 2))
    lookup_times_rx = lookup_times_tx.copy()
    scanline_weights = np.array([1.0, 2.0, 1.0])  # HMC
    amplitudes_tx = np.array([[1.0, 1.0]])
    amplitudes_rx = np.array([[1.0, 1.0]])
    focal_law = FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes_tx, amplitudes_rx, scanline_weights)

    frame = Frame(scanlines, time, tx, rx, probe, examination_object)

    return frame, focal_law


def test_delay_and_sum():
    frame, focal_law = make_delay_and_sum_case1()

    res = im.delay_and_sum(frame, focal_law)
    assert res == 20.0


def test_find_minimum_times():
    '''

                A1      A2

        B1      B2        B3

                C1



    Returns
    -------

    '''
    rt2 = math.sqrt(2.0)
    rt5 = math.sqrt(5.0)

    # Remark: B3 is a bit more on the right to have only one global minimum.

    # distance1[i, k] = distance between Ai and Bk
    distance1 = np.array([[rt2, 1.0, rt2 + .1], [rt5, rt2, 1.1]])

    # distance2[k, j] = distance between Ak and Cj
    distance2 = np.array([[rt2], [1.0], [rt2 + 0.1]])

    # Case 1
    # Best times: A1->B2->C1, A2->B2->C1.
    speed1 = 1.0
    speed2 = 1.0
    time1 = distance1 / speed1
    time2 = distance2 / speed2

    best_times, best_indices = im.find_minimum_times(time1, time2)
    expected_indices = np.array([[1], [1]])

    assert np.allclose(best_times, np.array([[2.0], [1.0 + rt2]]))
    assert np.all(best_indices == expected_indices)

    # Case 2: medium 2 is very fast so spend the shortest possible distance in medium 1
    # Best times: A1->B2->C1, A2->B2->C1.
    speed1 = 1.0
    speed2 = 50.0
    time1 = distance1 / speed1
    time2 = distance2 / speed2
    best_times, best_indices = im.find_minimum_times(time1, time2)
    expected_indices = np.array([[1], [2]])

    assert np.all(best_indices == expected_indices)

def test_find_minimum_times2():
    n = 300
    m = 301
    p = 302

    # The unique minimum of the i-th row of time_1 is on the i-th column and is 0.
    time_1 = np.fromfunction(lambda i, j: (j - i) % m, (n, m), dtype=np.float)

    # Each column of time_2 is constant
    time_2 = np.fromfunction(lambda i,j: j * m, (m, p), dtype=np.float)

    # Run the tested function:
    best_times, best_indices = im.find_minimum_times(time_1, time_2)

    # Expected results:
    best_times_expected = np.fromfunction(lambda i,j: m*j, (n, p), dtype=np.float)
    best_indices_expected = np.fromfunction(lambda i,j: i, (n, p), dtype=np.int)

    assert np.allclose(best_times_expected, best_times)
    assert np.all(best_indices_expected == best_indices)
