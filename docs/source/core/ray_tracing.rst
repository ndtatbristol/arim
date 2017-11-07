.. _ray_tracing:

===========
Ray tracing
===========

.. currentmodule:: arim.ray

Prerequisite: setting up the views (:ref:`path`).

The ray tracing function finds the physical paths between two points.
The recommended method is to use :func:`arim.ray.ray_tracing`.
The results are stored in ``View.tx_path.rays`` and ``View.rx_path.rays``, ie the attribute ``rays``
of the class :class:`Path` (:attr:`Path.rays`).

Example::

  import arim.ray
  arim.ray.ray_tracing(views.values())

  # Result of ray tracing:
  views['LT-L'].tx_path.rays

.. seealso ::

  :mod:`arim.ray`


