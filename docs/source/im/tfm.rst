=====================
Total focusing method
=====================

.. py:currentmodule:: arim.im.tfm

Original paper on TFM: [Holmes2005]_

Related module: :mod:`arim.im.tfm`

To compute TFM, instantiate a :class:`ContactTFM` or :class:`MultiviewTFM` object.

In arim the following general definition of TFM  is used [Budyn_engd1]_:

.. math::

  I(r) = \sum_{i=1}^{n} \sum_{j=1}^{n} 
    a_{i,\mathit{tx}}(r) a_{j,\mathit{rx}}(r)
    g_{ij}(
      \tau_{i,\mathit{tx}}(r) + \tau_{j,\mathit{rx}}(r)
      )

Where:

  - :math:`g_{ij}` corresponds to the scanline with such as ``tx = i`` and ``rx = j`` (cf. :doc:`../core/frame`)
  - :math:`a_{\cdot,\mathit{tx}}` is returned by :meth:`BaseTFM.get_amplitudes_tx`.
  - :math:`a_{\cdot,\mathit{rx}}` is returned by :meth:`BaseTFM.get_amplitudes_rx`.
  - :math:`\tau_{\cdot,\mathit{rx}}` is returned by :meth:`BaseTFM.get_lookup_times_tx`.
  - :math:`\tau_{\cdot,\mathit{rx}}` is returned by :meth:`BaseTFM.get_lookup_times_rx`.

Lookup times for multiview TFM are computed with :mod:`arim.im.fermat_solver`.

The module :mod:`arim.im.amplitudes` contains several classes for computing amplitudes.

