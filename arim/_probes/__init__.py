from .registry import ProbeRegistry
from . import bristol_ndt

__all__ = ['probes']
probes = ProbeRegistry()

for maker in bristol_ndt.makers:
    probes.register(maker)
