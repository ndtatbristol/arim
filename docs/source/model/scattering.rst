.. highlight:: python

.. _scattering:

==========
Scattering
==========

The scattering is stored as either a dictionary of functions or a dictionary of matrices (2d ndarray).
In both cases, the keys are *LL*, *LT*, *TL*, *TT*; the first letter corresponds to the mode of the incident wave; the second letter corresponds to the mode of the scattered wave.
Pulse-echo corresponds to the incident angle being equal to the scattered angle.

:func:`arim.model.model_amplitudes_factory` accepts this dictionary. Using matrices is general faster.


Scattering as a function
========================

::

  vl = 6300
  vt = 3100
  freq = 5e6
  radius = 0.1e-3
  scat_funcs = arim.ut.scattering_2d_cylinder_funcs(vl/freq, vt/freq, radius)

``scat_funcs`` is a dictionary of function. For example, ``scat_funcs['LT']`` is a function.
Each function takes two arguments: the first is the incident angles, the second is the scattering angles.
The output is the scattering amplitude, given as an array of the same shape of the input.

Example::

  # scattering amplitude at 30° for an incident wave at 0° 
  scat_funcs['LT'](0., np.pi / 6)

.. seealso::
  :func:`arim.ut.scattering_2d_cylinder_funcs`, :func:`arim.ut.scattering_point_source_funcs`.

Scattering as a matrix
======================

Because computing the scattering amplitudes can be expensive, it is often useful to precompute
them all incident and scattered angles and then to interpolate the values.

The angles are discretised as a linear spaced vector in the inverval :math:`[-\pi, \pi[`. The number of points
is denoted ``n``. They can be obtained with :func:`arim.ut.scattering_angles`.
::

  theta[k] := -pi + 2 pi k / n for k=0...n-1.

The grids of incident and outgoing (scattered) angles are defined as follows.
They can be obtained with :func:`arim.ut.scattering_angles_grid`.
::

  inc_angles[i, j] := theta[j]
  out_angles[i, j] := theta[i]


At a given frequency, the scattering matrices are defined as matrices of size ``(n, n)``.
``scat_matrices['LT'][i, j]`` corresponds to the incident angle ``theta[j]`` and the scattered angle ``theta[i]``
for an incident wave L and a scattered wave T..

Example::

  vl = 6300
  vt = 3100
  freq = 5e6
  radius = 0.1e-3
  n = 100
  scat_matrices = arim.ut.scattering_2d_cylinder_matrices(n, vl/freq, vt/freq, radius)
  theta = arim.ut.scattering_angles(n)

.. seealso::

  :func:`arim.ut.scattering_2d_cylinder_matrices`, :func:`arim.ut.scattering_angles`.

To interpolate a scattering matrix, use :func:`arim.ut.interpolate_scattering_matrix` (one matrix) or
:func:`arim.ut.scattering_matrices_to_interp_funcs`.
They return interpolators that take as arguments the incident and scattered angles.


File format specification
=========================

`HDF5 file format <https://www.hdfgroup.org/downloads/hdf5/>`_

Complex datatype is defined as a compound type with fields ``r`` and ``i`` for real and imaginary parts.
The number of bits for datatypes is not specified and may be adjusted to one's needs.
Strings must be ASCII only.
Compression for datasets: none or gzip. Letter case must be respected.
Additional datasets/attributes are allowed but must not clash with optional datasets/attributes.

Required datasets
-----------------

======================== ====================== ============================================= ======================
Name                     Datatype               Shape                                         Comments
======================== ====================== ============================================= ======================
``/scattering_LL``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/scattering_LT``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/scattering_TL``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/scattering_TT``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/scattering_TT``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/frequencies``         Float                  ``(numfrequencies, )``                        Frequencies at which the scattering matrices are described.
======================== ====================== ============================================= ======================

Optional datasets
-----------------

======================== ====================== ============================================= ======================
Name                     Datatype               Shape                                         Comments
======================== ====================== ============================================= ======================
``/inc_angles``          Float                  ``(numangles, numangles)``                    As defined above.
``/out_angles``          Float                  ``(numangles, numangles)``                    As defined above.
``/material_velocity_L`` Float                  ``(1, )``                                     meter per second
``/material_velocity_T`` Float                  ``(1, )``                                     meter per second
``/material_density``    Float                  ``(1, )``                                     kg/m3
======================== ====================== ============================================= ======================

Required attribute
------------------

======================== ========================== ====================== ======================  ======================
Location                 Name                       Datatype               Shape                   Comments
======================== ========================== ====================== ======================  ======================
``/``                    ``file_format_version``    String                 N/A                     Must be: ``1.0``
======================== ========================== ====================== ======================  ======================


Optional attributes
-------------------

======================== ======================   =============== ======================  ======================
Location                 Name                     Datatype        Shape                   Comments
======================== ======================   =============== ======================  ======================
``/``                    ``author``               String          N/A
``/``                    ``creation_time``        String          N/A                     `ISO 8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_
======================== ======================   =============== ======================  ======================
