===========
Ray tracing
===========

.. py:currentmodule:: arim.im.fermat_solver

Prerequisite: setting up the paths (:ref:`path`).

Do the ray tracing: :class:`FermatSolver`

The result of ray tracing is stored in :class:`Rays` objects.

Using the rays:

  - compute angles of incidence: :meth:`Rays.get_incoming_angles`
  - compute angles of transmission or reflection: :meth:`Rays.get_outgoing_angles`

