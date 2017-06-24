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

::

  vl = 6300
  vt = 3100
  freq = 5e6
  radius = 0.1e-3
  n = 100
  scat_matrices = arim.ut.scattering_2d_cylinder_matrices(n, vl/freq, vt/freq, radius)
  theta = arim.ut.theta_scattering_matrix(n)

Because computing the scattering amplitudes can be expensive, it is often useful to precompute
them all incident and scattered angles and then to interpolate the values.
In the example above, the interval :math:`[-\pi, \pi[` is discretised with 100 points.


``scat_matrices['LL']`` is a matrix of shape (100, 100).
``scat_matrices['LL'][i, j]`` corresponds to the incident angle ``theta[i]`` and the scattered angle ``theta[j]``.

.. seealso::

  :func:`arim.ut.scattering_2d_cylinder_matrices`, :func:`arim.ut.theta_scattering_matrix`.

To interpolate a scattering matrix, use :func:`arim.ut.interpolate_scattering_matrix` (one matrix) or :func:`arim.ut.scattering_matrices_to_interp_funcs`.
They return interpolators that take as arguments the incident and scattered angles.
