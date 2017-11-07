.. highlight:: python

.. _scattering:

==========
Scattering
==========

.. currentmodule:: arim.scat

The main module for scattering is :mod:`arim.scat`. To get started, read the documentation of :func:`scat_factory`
and :class:`Scattering2d`.



Pulse-echo corresponds to the incident angle being equal to the scattered angle.

:func:`arim.model.model_amplitudes_factory` accepts this dictionary. Using matrices is general faster.

Scattering object
=================

The easiest way to use scattering is to call :func:`scat_factory`. Examples::

    material = arim.Material(6300., 3120., 2700., 'solid', {'long_name': 'Aluminium'})
    scat_obj = scat_factory('file', material, 'scattering_data.mat')
    
    scat_obj = scat_factory('crack_centre', material, crack_length=2.0e-3)
    
    scat_obj = scat_factory('sdh', material, radius=0.5e-3)
    
    scat_obj = scat_factory('point', material) # unphysical, debug only

Scattering matrix
=================

Because computing the scattering amplitudes can be expensive, it is often useful to precompute
them all incident and scattered angles and then to interpolate the values.

The angles are discretised as a linear spaced vector in the inverval :math:`[-\pi, \pi[`. The number of points
is denoted ``n``. They can be obtained with :func:`arim.scat.make_angles`.
::

  theta[k] := -pi + 2 pi k / n for k=0...n-1.

The grids of incident and outgoing (scattered) angles are defined as follows.
They can be obtained with :func:`arim.scat.make_angles_grid`.
::

  inc_angles[i, j] := theta[j]
  out_angles[i, j] := theta[i]


At a given frequency, the scattering matrices are defined as matrices of size ``(n, n)``.
``scat_matrices['LT'][i, j]`` corresponds to the incident angle ``theta[j]`` and the scattered angle ``theta[i]``
for an incident wave L and a scattered wave T.


Import scattering data
======================

See :mod:`arim.io.scat`


File format specification
-------------------------

`HDF5 file format <https://www.hdfgroup.org/downloads/hdf5/>`_

Complex datatype is defined as a compound type with fields ``r`` and ``i`` for real and imaginary parts.
The number of bits for datatypes is not specified and may be adjusted to one's needs.
Strings must be ASCII only.
Compression for datasets: none or gzip. Letter case must be respected.
Additional datasets/attributes are allowed but must not clash with optional datasets/attributes.

Required datasets
^^^^^^^^^^^^^^^^^

======================== ====================== ============================================= ======================
Name                     Datatype               Shape                                         Comments
======================== ====================== ============================================= ======================
``/scattering_LL``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/scattering_LT``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/scattering_TL``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/scattering_TT``       Complex                ``(numfrequencies, numangles, numangles)``    As defined above.
``/frequencies``         Float                  ``(numfrequencies, )``                        Frequencies at which the scattering matrices are described.
======================== ====================== ============================================= ======================

Optional datasets
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^

======================== ========================== ====================== ======================  ======================
Location                 Name                       Datatype               Shape                   Comments
======================== ========================== ====================== ======================  ======================
``/``                    ``file_format_version``    String                 N/A                     Must be: ``1.0``
======================== ========================== ====================== ======================  ======================


Optional attributes
^^^^^^^^^^^^^^^^^^^

======================== ======================   =============== ======================  ======================
Location                 Name                     Datatype        Shape                   Comments
======================== ======================   =============== ======================  ======================
``/``                    ``author``               String          N/A
``/``                    ``creation_time``        String          N/A                     `ISO 8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_
======================== ======================   =============== ======================  ======================
