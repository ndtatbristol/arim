"""
Python library for modelling and imaging immersion inspections in ultrasonic testing (nondestructive testing)
"""

from . import settings
from . import exceptions
from .core import *

from . import _probes, geometry, measurement, config, helpers, ut, io
from .geometry import Points, Grid


# probe database
probes = _probes.probes

# Aliases
L = Mode.L
T = Mode.T

# Must respect PEP 440 and SemVer
#  https://www.python.org/dev/peps/pep-0440/
#  https://semver.org/
# Must be bumped at each release
__version__ = "0.9.0.a0"
