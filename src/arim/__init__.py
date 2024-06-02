"""
Python library for modelling and imaging immersion inspections in ultrasonic testing (nondestructive testing)
"""

from . import (
    _probes,
    config,
    exceptions,
    geometry,
    helpers,
    io,
    measurement,
    settings,
    ut,
)
from .core import (
    BlockInContact,
    BlockInImmersion,
    CaptureMethod,
    ElementShape,
    ExaminationObject,
    Frame,
    Interface,
    InterfaceKind,
    Material,
    Mode,
    Path,
    Probe,
    StateMatter,
    Time,
    TransmissionReflection,
    View,
    material_attenuation_factory,
)
from .geometry import Grid, Points

# probe database
probes = _probes.probes

# Aliases
L = Mode.L
T = Mode.T

__all__ = [
    "config",
    "exceptions",
    "geometry",
    "helpers",
    "io",
    "probes",
    "measurement",
    "settings",
    "ut",
    "L",
    "T",
    "Grid",
    "Points",
    "Probe",
    "BlockInContact",
    "BlockInImmersion",
    "CaptureMethod",
    "ElementShape",
    "ExaminationObject",
    "Frame",
    "Interface",
    "InterfaceKind",
    "Material",
    "Mode",
    "Path",
    "StateMatter",
    "Time",
    "TransmissionReflection",
    "View",
    "material_attenuation_factory",
]

# Must respect PEP 440 and SemVer
#  https://www.python.org/dev/peps/pep-0440/
#  https://semver.org/
# Must be bumped at each release
__version__ = "0.10.0.a0"
