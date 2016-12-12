===========
Ray tracing
===========

.. py:currentmodule:: arim.im.fermat_solver

Prerequisite: setting up the paths and the views (:ref:`path`).

Do the ray tracing: :func:`ray_tracing`, which relies on :class:`FermatSolver`

The results of ray tracing are :class:`Rays` objects. For each path, the corresponding rays
are stored in :attr:`Path.rays`.

Example::

  # assume views_dict is a dictionary containing the views
  arim.im.ray_tracing(views.values())

  # Result of ray tracing:
  views['LT-L'].tx_path.rays


Using the rays:

  - compute angles of incidence: :meth:`Rays.get_incoming_angles`
  - compute angles of transmission or reflection: :meth:`Rays.get_outgoing_angles`

