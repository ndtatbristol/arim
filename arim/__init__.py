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

#
L = Mode.L
T = Mode.T


__author__ = "Nicolas Budyn, Rhodri Bevan"
__credits__ = ["Nicolas Budyn", "Rhodri Bevan"]
__email__ = "n.budyn@bristol.ac.uk"
__copyright__ = f"{__author__} 2016-2020"
__license__ = "MIT"

# Must respect PEP 440: https://www.python.org/dev/peps/pep-0440/
# Must be bumped at each release
__version__ = "0.9.rc2"
