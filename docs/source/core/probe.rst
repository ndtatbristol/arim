.. _probe:

=====
Probe
=====

.. py:currentmodule:: arim.core.probe

A probe is a set of transducers called elements. Each element can be a receiver or transmitter.

Probe objects handle two Cartesian coordinate systems at a time: the probe coordinate system (PCS) and the global coordinate
system (GCS). For example, :attr:`Probe.locations` and :attr:`Probe.locations_pcs` contains the element centres in respectively
the GCS and the PCS.

The coordinates of the PCS expressed in the GCS are given in :attr:`Probe.pcs`.

The conventional centre of the probe is defined as the point (0., 0., 0.) in the PCS. Usually, this corresponds either
to first element of the element or the middle element. This can be changed by reassigning the attribute ``pcs`` with
a new coordinate system centered where you want.

To move the probe: it is recommended to use methods ``rotate`` and ``translate`` instead of changing directly the
attribute ``locations``.


Limits:

  - Dimensions of elements are defined in the PCS. Should we attach to each element a specific coordinate system? 

:class:`arim.core.probe.Probe`

