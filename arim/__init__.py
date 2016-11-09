from . import settings
from . import exceptions
from .enums import *
from .core import *
from .misc import get_git_version

from . import _probes, signal, geometry, io, im, registration, model

probes = _probes.probes

__author__ = "Nicolas Budyn"
__credits__ = []
__license__ = "All rights reserved"
__copyright__ = "2016, Nicolas Budyn"

# Must respect PEP 440: https://www.python.org/dev/peps/pep-0440/
# Must be bumbed at each release
__version__ = '0.5.dev2'

