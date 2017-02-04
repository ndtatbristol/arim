.. _probe:

=====
Probe
=====

.. py:currentmodule:: arim.core

.. seealso::

  Reference of class :class:`Probe`

A probe is a set of transducers called elements. Each element can be a receiver or transmitter.
The corresponding object in arim is called .

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


Importing a probe from the library::

  import arim

  # Print available probes:
  print(arim.probes.keys())

  # Load a probe:
  probe = arim.probes['ima_50_MHz_128_1d']


Creating a linear or matrix probe (regularly spaced elements)::

  import arim
  nan = float('nan')
  probe = arim.Probe.make_matrix_probe(
      frequency=5.e6,
      numx=128,  # number of elements in a row
      pitch_x=0.3e-3,
      numy=1,  # 1 row of elements, i.e. linear probe
      pitch_y=nan,  # irrelevant for linear probe
      dimensions=[0.2e-3, 15.e-3, nan]  # dimensions along x, y and z
  )
