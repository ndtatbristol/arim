import numpy as np
import pytest

import arim.io.brain as mat
from arim.core import CaptureMethod
from tests.helpers import get_data_filename


@pytest.fixture(scope="module", params=["mat7", "mat73"])
def expdata(request):
    """
    Fixture for test_load_expdata
    """
    resource = f"brain/exp_data.{request.param}.mat"
    return get_data_filename(resource)


def test_load_expdata(expdata):
    frame = mat.load_expdata(expdata)

    assert frame.examination_object.material.longitudinal_vel == 6300
    assert np.isclose(frame.time.step, 4.0e-8)
    assert np.isclose(frame.time.start, 5.0e-6)
    assert frame.timetraces.shape == (2080, 300)
    assert np.allclose(
        frame.timetraces[0, :4], [0.05468750, 0.05468750, 0.05468750, 0.04687500]
    )  # first timetrace
    assert np.allclose(
        frame.timetraces[-1, :4], [0.07031250, 0.06250000, 0.06250000, 0.06250000]
    )  # last timetrace
    assert np.allclose(frame.tx[:4], [0, 0, 0, 0])
    assert np.allclose(frame.rx[:4], [0, 1, 2, 3])
    assert frame.probe.numelements == 64
    assert np.allclose(frame.probe.locations[0], [-0.0198450, 0, 0])  # first element
    assert np.allclose(frame.probe.locations[-1], [+0.0198450, 0, 0])  # last element

    assert frame.capture_method == CaptureMethod.hmc

if __name__ == '__main__':
    test_load_expdata("brain/exp_data.mat73.mat")