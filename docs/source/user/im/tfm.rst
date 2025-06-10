.. _tfm:

=====================
Total focusing method
=====================

.. py:currentmodule:: arim.im.tfm

Original paper on TFM: [Holmes]_

Related modules:

- :mod:`arim.im.tfm`: high-level functions for TFM
- :mod:`arim.im.das`: actual computation of delay-and-sum


Definition
----------

The most generic definition available in arim is:

.. math::

  I(r) = \sum_{i,j \in S}
    A_{ij}(r)
    g_{ij}(
      \tau_{i}(r) + \tau'_{j}(r)
      )

:math:`g_{ij}(t)` is the timetrace of the transmitter `i` and the receiver `j`, obtained from
:attr:`arim.core.Frame.timetraces`. The set `S` of the transmitters and receivers is obtained from
:attr:`arim.core.Frame.tx` and :attr:`arim.core.Frame.rx`. Its size is denoted `numtimetraces`.
TFM is always applied on all timetraces in the frame; to apply TFM on a subset of the timetraces, create a new Frame
with only the selected timetraces and pass it to TFM.

:math:`\tau_{i}(r)` is the time of flight for the transmission path between the element `i` and the grid point `r`.
These times are stored in ``lookup_times_tx``, a 2d array of shape `(numgridpoints, numelements)`. Similarly, the
times of fight for the reception path are stored in ``lookup_times_rx``.

The weights :math:`A_{ij}(r)` are stored in a `(numgridpoints, numtimetraces)` array named ``amplitudes``. Because this
array may be too big for the memory, chunking it may be necessary (example: :class:`arim.model.ModelAmplitudes`).

If the following decomposition is possible

.. math::

    A_{ij}(r) = B_{i}(r) B'_{j}(r)

then ``amplitudes`` can be a :class:`arim.model.TxRxAmplitudes` object where ``amplitudes_tx`` and ``amplitudes_rx``,
which contain respectively :math:`B_{i}(r)` and  :math:`B'_{j}(r)`, are arrays of shape `(numelements, numtimetraces)`.

Finally if for all `i`, `j` and `r`

.. math::

  A_{ij}(r) = 1,

``amplitudes`` may be set to ``None``. Internally a faster implementation of delay-and-sum will be used.

Perfoming TFM
-------------

See :mod:`arim.im.tfm` and :mod:`arim.im.das`


TFM with HMC frame
------------------

In linear imaging, it is assumed that :math:`g_{ij}(t) = g_{ji}(t)`, which leads to acquire only half the full matrix
response (half matrix capture, HMC).

Under this assumption, if for all `i` and `r`,  :math:`B_i(r) = B'_i(r)` and :math:`\tau_i(r) = \tau'_i(r)`, then

.. math::

  I(r)
    &= \sum_{i,j = 1}^{n} B_{i}(r) B'_{j}(r) g_{ij}(\tau_{i}(r) + \tau'_{j}(r))) \\
    &= \sum_{i,j = 1}^{n} B_{i}(r) B_{j}(r) g_{ij}(\tau_{i}(r) + \tau_{j}(r))) \\
    &= \sum_{i = 1}^{n} B_{i}(r)^2 g_{ii}(2 \tau_{i}(r))
    + 2 \sum_{i,j = 1\\i < j}^{n} B_{i}(r) B_{j}(r) g_{ij}(\tau_{i}(r) + \tau_{j}(r)) \\
    &= \sum_{i,j = 1\\i \leq j}^{n} B_{i}(r) B_{j}(r) g'_{ij}(\tau_{i}(r) + \tau_{j}(r))

where

.. math::

  g'_{ij}(t) =
  \begin{cases}
  g_{ij},  & \text{if $i=j$} \\
  2 g_{ij},  & \text{if $i \neq j$}
  \end{cases}

This reduces the number of summands and therefore the computation time. This last equation is used in
:func:`contact_tfm`. Internally, the weighted timetraces :math:`g'_{ij}` are obtained with
:meth:`FocalLaw.weigh_timetraces`.

This technique is not used :func:`tfm_for_view` because the times of flight in transmission and reception are not
the same in general.

