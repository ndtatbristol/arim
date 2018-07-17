"""
"""
import numpy as np

from .. import core
from .. import geometry as g
from .registry import ProbeMaker

makers = []


def _make_ima_50_MHz_128_1d():
    metadata = dict(
        probe_type="linear",
        short_name="ima_50_MHz_128_1d",
        long_name="Imasonic 5.0 MHz 128 elts linear array probe",
        version=0,
        serial="7186 A101",
    )
    numelements = 128
    shapes = np.full(numelements, core.ElementShape.rectangular, dtype=np.object)
    orientations = g.Points.from_xyz(
        np.zeros((numelements,), dtype=np.float),
        np.zeros((numelements,), dtype=np.float),
        np.ones((numelements,), dtype=np.float),
    )
    dimensions = g.Points.from_xyz(
        np.full(numelements, 0.2e-3),
        np.full(numelements, 15e-3),
        np.full(numelements, 0.),
    )
    dead_elements = np.full(numelements, False, dtype=np.bool)

    probe = core.Probe.make_matrix_probe(
        numx=numelements,
        pitch_x=0.3e-3,
        numy=1,
        pitch_y=np.nan,
        frequency=5e6,
        shapes=shapes,
        orientations=orientations,
        dimensions=dimensions,
        dead_elements=dead_elements,
        metadata=metadata,
    )
    return probe


makers.append(
    ProbeMaker(
        _make_ima_50_MHz_128_1d,
        "ima_50_MHz_128_1d",
        "Imasonic 5.0 MHz 128 elts linear array probe",
    )
)


def _make_ima_50_MHz_64_1d():
    metadata = dict(
        probe_type="linear",
        short_name="ima_50_MHz_64_1d",
        long_name="Imasonic 5.0 MHz 64 elts linear array probe",
        version=0,
        serial="12157 1001",
    )
    numelements = 64
    shapes = np.full(numelements, core.ElementShape.rectangular, dtype=np.object)
    orientations = g.Points.from_xyz(
        np.zeros((numelements,), dtype=np.float),
        np.zeros((numelements,), dtype=np.float),
        np.ones((numelements,), dtype=np.float),
    )
    dimensions = g.Points.from_xyz(
        np.full(numelements, 0.53e-3),
        np.full(numelements, 15e-3),
        np.full(numelements, 0.),
    )
    dead_elements = np.full(numelements, False, dtype=np.bool)

    probe = core.Probe.make_matrix_probe(
        numx=numelements,
        pitch_x=0.63e-3,
        numy=1,
        pitch_y=np.nan,
        frequency=5e6,
        shapes=shapes,
        orientations=orientations,
        dimensions=dimensions,
        dead_elements=dead_elements,
        metadata=metadata,
    )
    return probe


makers.append(
    ProbeMaker(
        _make_ima_50_MHz_64_1d,
        "ima_50_MHz_64_1d",
        "Imasonic 5.0 MHz 64 elts linear array probe",
    )
)


def _make_ima_25_MHz_64_1d():
    metadata = dict(
        probe_type="linear",
        short_name="ima_25_MHz_64_1d",
        long_name="Imasonic 2.5 MHz 64 elts linear array probe",
        version=0,
        serial="6065 A 101",
    )
    numelements = 64
    shapes = np.full(numelements, core.ElementShape.rectangular, dtype=np.object)
    orientations = g.Points.from_xyz(
        np.zeros((numelements,), dtype=np.float),
        np.zeros((numelements,), dtype=np.float),
        np.ones((numelements,), dtype=np.float),
    )
    dimensions = g.Points.from_xyz(
        np.full(numelements, 0.35e-3),
        np.full(numelements, 15e-3),
        np.full(numelements, 0.),
    )
    dead_elements = np.full(numelements, False, dtype=np.bool)

    probe = core.Probe.make_matrix_probe(
        numx=numelements,
        pitch_x=0.5e-3,
        numy=1,
        pitch_y=np.nan,
        frequency=2.5e6,
        shapes=shapes,
        orientations=orientations,
        dimensions=dimensions,
        dead_elements=dead_elements,
        metadata=metadata,
    )
    return probe


makers.append(
    ProbeMaker(
        _make_ima_25_MHz_64_1d,
        "ima_25_MHz_64_1d",
        "Imasonic 2.5 MHz 64 elts linear array probe",
    )
)


def _make_sonaxis_150_MHz_110_1d():
    metadata = dict(
        probe_type="linear",
        short_name="sonaxis_150_MHz_110_1d",
        long_name="Sonaxis 15.0 MHz 110 elts linear array probe",
        version=0,
        serial="unknown",
    )
    numelements = 110
    shapes = np.full(numelements, core.ElementShape.rectangular, dtype=np.object)
    orientations = g.Points.from_xyz(
        np.zeros((numelements,), dtype=np.float),
        np.zeros((numelements,), dtype=np.float),
        np.ones((numelements,), dtype=np.float),
    )
    dimensions = g.Points.from_xyz(
        np.full(numelements, 0.14e-3),
        np.full(numelements, 3.0e-3),
        np.full(numelements, 0.),
    )
    dead_elements = np.full(numelements, False, dtype=np.bool)

    probe = core.Probe.make_matrix_probe(
        numx=numelements,
        pitch_x=0.17e-3,
        numy=1,
        pitch_y=np.nan,
        frequency=15e6,
        shapes=shapes,
        orientations=orientations,
        dimensions=dimensions,
        dead_elements=dead_elements,
        metadata=metadata,
    )
    return probe


makers.append(
    ProbeMaker(
        _make_sonaxis_150_MHz_110_1d,
        "sonaxis_150_MHz_110_1d",
        "Sonaxis 15.0 MHz 110 elts linear array probe",
    )
)
