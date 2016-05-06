from collections import namedtuple

from .. import utils as u
from .. import settings as s
from ..enums import CaptureMethod

__all__ = ['FocalLaw']

FocalLaw = namedtuple('FocalLaw', ['lookup_times_tx', 'lookup_times_rx', 'amplitudes_tx', 'amplitudes_rx', 'scanline_weights'])


# class FocalLaw:
#     def __init__(self, lookup_times_tx, lookup_times_rx, amplitudes, scanline_weights):
#         self.lookup_times_tx = lookup_times_tx
#         self.lookup_times_rx = lookup_times_rx
#         self.amplitudes = amplitudes
#         self.scanline_weights = scanline_weights
