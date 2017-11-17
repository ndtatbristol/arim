"""
Delay-and-sum functions

Contains different implementation of delay-and-sum algorithm.

.. currentmodule:: arim.im.das

Data structures
---------------

- ``lookup_times_tx``: ndarray of shape (numgridpoints, numelements)
- ``lookup_times_rx``: ndarray of shape (numgridpoints, numelements)
- ``amplitudes_tx``: ndarray of shape (numgridpoints, numelements)
- ``amplitudes_tx``: ndarray of shape (numgridpoints, numelements)
- ``amplitudes``: TxRxAmplitudes or ndarray (numgridpoints, numscanlines) or None



.. autosummary::
    :toctree: _autosummary

    delay_and_sum
    delay_and_sum_numba
    delay_and_sum_cpu
    delay_and_sum_naive

"""

from .das import *
