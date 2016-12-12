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

The origin of the PCS is by definition the reference point of the probe. Depending on the convention, it can be
the first element or the geometric centre of the probe. By default in arim, the geometric centre of the probe is used
(average location of all elements). The reference point of a probe can be changed with the method
:meth:`Probe.set_reference_element`.

A probe can be moved using the methods ``rotate`` and ``translate``. These functions update the PCS. 
It is not recommended to change directly the values of the attribute ``locations``.

Limits:

  - Dimensions of elements are defined in the PCS. Should we attach to each element a specific coordinate system? 

:class:`arim.core.probe.Probe`

