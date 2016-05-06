"""
This module is called by other core modules and utils modules. We put enums here to avoid circular dependencies.

"""

from enum import Enum

__all__ = ['CaptureMethod']


class CaptureMethod(Enum):
    unsupported = 0
    fmc = 1
    hmc = 2
