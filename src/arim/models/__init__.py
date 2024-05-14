"""
Imaging and forward models

Contribution of a scatterer for a view::

    F_ij(omega) = Q_i(omega, y) * Q'_j(omega, y) * S_ij(omega, y)
                  * exp(-i omega (tau_i(y) + tau_j(y))) * X(omega)

where X is the input signal, Q_i contains the physics for the direct path (from the
transmitter to the scatterer), Q_j contains the physics for the reverse path
(from the scatterer to the receiver), S_ij is the scattering function.

Q_i and Q'_j are called respectively 'tx_ray_weights' and 'rx_ray_weights'.
The coefficient P_ij = Q_i Q'_j S_ij is called 'full_ray_weights'.

tau_ij = tau_i + tau_j is the time of flight associated with the full ray path from
i to j.


.. autosummary::
   :toctree: .

   block_in_immersion
   block_in_contact
   generic_block_in_immersion
   generic_block_in_contact
   helpers


"""
