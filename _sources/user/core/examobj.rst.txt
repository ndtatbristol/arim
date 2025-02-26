.. _examobj:

==================
Examination object
==================

The :class:`ExaminationObject` class and its subclasses contain information about the inspected object.
This object is mainly a data container.
It typically contains a :class:`Material` object and geometry information.

Example::

  aluminium = arim.Material(6300., 3100., 2700.)
  exam_obj = arim.ExaminationObject(aluminium)

For performing immersion TFM or using the forward model, the geometry of the front and
back walls and the material of the couplant are needed.
The :class:`BlockInImmersion`, a subclass of :class:`ExaminationObject`, can
be used for storing this information.

Example::

  # material definition
  block_material = arim.Material(6300., 3100., 2700., 'solid', metadata=dict(long_name='Aluminium'))
  couplant_material = arim.Material(1480., None, 1000.., 'liquid', metadata=dict(long_name='Water')))

  # define the geometry
  numpoints = 500
  z_frontwall = 0.e-3
  xmin = 0e-3
  xmax = 20e-3
  frontwall = arim.geometry.points_1d_wall_z(xmin, xmax, z_frontwall, numpoints)
  backwall = arim.geometry.points_1d_wall_z(xmin, xmax, z_backwall, numpoints)

  exam_obj = arim.BlockInImmersion(block_material, couplant_material, frontwall, backwall)

The front and back walls are defined as clouds of points.
Using a fine enough sampling is important for the precision of the ray tracing and the model.

.. seealso::
  
  More about the definition of the geometry: :mod:`arim.geometry`
