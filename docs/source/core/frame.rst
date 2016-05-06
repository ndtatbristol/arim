.. _frame:

=====
Frame
=====

A frame is an acquisition for a given spatial location of the probe. A frame contains the voltage-time data
(:attr:`arim.core.frame.Frame.scanlines`).

Limits:

  - A frame can contain only one probe.
  - A frame contains only scanlines, i.e. data associated to exactly one transmitter and one receiver.


:class:`arim.core.frame.Frame`

