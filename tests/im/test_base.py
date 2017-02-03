import math

import numpy as np

from arim import im


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
    time_2 = np.fromfunction(lambda i, j: j * m, (m, p), dtype=np.float)

    # Run the tested function:
    best_times, best_indices = im.find_minimum_times(time_1, time_2)

    # Expected results:
    best_times_expected = np.fromfunction(lambda i, j: m*j, (n, p), dtype=np.float)
    best_indices_expected = np.fromfunction(lambda i, j: i, (n, p), dtype=np.int)

    assert np.allclose(best_times_expected, best_times)
    assert np.all(best_indices_expected == best_indices)
