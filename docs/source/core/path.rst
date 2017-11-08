.. _path:

===============
Paths and views
===============

.. currentmodule:: arim.core

The class :class:`Path` stores information about the rays between two points (probe element, scatterer).
Typically, a :class:`Path` contains the rays starting from the probe and ending at
the scatterers. Examples: `L` (longitudinal mode from the front wall to the scatterer),
`LT` (L mode from the front wall to the back wall, T mode from the back wall to the front wall).
It is also possible to use :class:`Path` for modelling rays starting and ending at the probe; for example a ray starting
from the probe, reflected against the front wall and going back to the probe.

The class :class:`View` glues together two :class:`Path` objects for modelling rays from the probe to a scatterer
(transmit path) and from the scatterer to the probe (receive path).
Using a :class:`View` instead of one :class:`Path` that starts and ends at the probe allows reducing greatly the amount
of memory and calculation required for imaging and modelling. This is the *raison d'être* of the class :class:`View`.


Creating the views
==================

The recommended way is to use :func:`arim.models.block_in_immersion.make_views`.

This function requires the positions of the probe as a :class:`arim.geometry.OrientedPoints` object::

  probe = arim.Probe(...)  # to fill up
  probe_p = probe.to_oriented_points()

It also requires the position of the scatterers as :class:`arim.geometry.OrientedPoints`::

  # one scatterer in x=10e-3, y=0, z=10e-3
  scatterer_p = arim.geometry.default_oriented_points(arim.Points([[10e-3, 0., 10e-3]]))

Alternatively, the scatterers can be defined from a regularly spaced grid::

  grid = arim.geometry.Grid(xmin, xmax, ymin, ymax, zmin, zmax, pixel_size)
  scatterer_p = grid.to_oriented_points()

A :class:`BlockInImmersion` object is required, see :ref:`examobj`.

Finally::

  import arim.models.block_in_immersion as bim
  views = bim.make_views(exam_obj, probe_p, scatterer_p, max_number_of_reflection=1, tfm_unique_only=False)

For imaging with TFM, the views that give redundent results can be removed by setting ``tfm_unique_only=True``.

``views`` is a Python dictionary (dict)::

  views['L-T']  # this is the view L-T

At this stage, the views contains which interfaces are crossed and which modes are propagating but not exact interface
points through which the rays go. This step is performed in :ref:`ray_tracing`.


Nuts and bolts
==============

This section describes the internal machinery of arim for paths and views. It can be skipped at first reading.


A :class:`Path` is composed of :class:`Interface` objects. Each interface contains both the location of the points
and their orientations but also the description of the physical process happening there: solid-to-fluid transmission,
reflection against a liquid, etc. :class:`Path` also defines the modes between the interfaces.
