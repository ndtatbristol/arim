.. highlight:: python

.. _transmission_reflection:

====================================
Transmission-reflection coefficients
====================================

Fluid-solid interface
----------------------

  - Fluid to solid: :func:`arim.ut.fluid_solid`
  - Solid to fluid: :func:`arim.ut.solid_l_fluid` and :func:`arim.ut.solid_t_fluid`

References: [KK]_ and [Schmerr]_.

*Example:*

.. literalinclude:: ../../../examples/transmission_coefficients/solid_to_liquid.py
   :language: python
   :linenos:

.. literalinclude:: ../../../examples/transmission_coefficients/liquid_to_solid.py
   :language: python
   :linenos:

.. seealso::

    Full reference: :mod:`arim.model`, :mod:`arim.ut`


.. todo::
  Write a more complete example of :mod:`arim.model` functions.
