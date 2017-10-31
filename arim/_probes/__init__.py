"""
Database of probes

Usage::

    import arim

    # Print available probes:
    print(arim.probes.keys())

    # Load a probe:
    probe = arim.probes['ima_50_MHz_128_1d']

"""

from .registry import ProbeRegistry
from . import bristol_ndt

__all__ = ['probes']
probes = ProbeRegistry()

for maker in bristol_ndt.makers:
    probes.register(maker)
