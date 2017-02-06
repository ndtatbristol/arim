"""
Compute and plot the transmission and reflection coefficients.

Case: an incident longitudinal in the fluid hits the wall of an aluminium part.
"""

import numpy as np
import warnings

from matplotlib.pyplot import *

from arim.model import fluid_solid, snell_angles

# %% Parameters

NULL_TRANSMISSION_ABOVE_CRIT_ANGLES = True
SAVEFIG = False

# water:
c_fluid = 1480.
rho_fluid = 1000.

# aluminium :
c_l = 6320.
c_t = 3130.
rho_solid = 2700.

# %% Computation of reflection and transmission coefficients

# Critical angles
critical_l = np.arcsin(c_fluid / c_l)
critical_t = np.arcsin(c_fluid / c_t)
print("Critical angle L: {:.3f}°".format(np.rad2deg(critical_l)))
print("Critical angle T: {:.3f}°".format(np.rad2deg(critical_t)))

# Remark: by using complex angles, complex reflection and transmission coefficients
# are computed.
alpha_fluid = np.asarray(np.linspace(0, np.pi / 2, 50000), dtype=complex)

alpha_l = snell_angles(alpha_fluid, c_fluid, c_l)
alpha_t = snell_angles(alpha_fluid, c_fluid, c_t)

reflection, transmission_l, transmission_t = fluid_solid(alpha_fluid, rho_fluid,
                                                         rho_solid, c_fluid, c_l, c_t,
                                                         alpha_l, alpha_t)

# Transmission coefficients above the critical angles are not physical.
# Force them to 0.
if NULL_TRANSMISSION_ABOVE_CRIT_ANGLES:
    transmission_l[alpha_fluid > critical_l] = 0
    transmission_t[alpha_fluid > critical_t] = 0

# %% Plot reflection and transmission coefficients (alpha_fluid axis)

figure()
plot(np.rad2deg(alpha_fluid.real), np.abs(reflection), label='reflection')
plot(np.rad2deg(alpha_fluid.real), np.abs(transmission_l), label='transmission L')
plot(np.rad2deg(alpha_fluid.real), np.abs(transmission_t), label='transmission T')
title('Tranmission and reflection coefficients from water to aluminium')
axvline(x=np.rad2deg(critical_l), color='k', linestyle='--', label='critical angle L')
axvline(x=np.rad2deg(critical_t), color='k', linestyle='-', label='critical angle T')
xlabel('angle of the incident wave in water (deg)')
ylabel('coefficient amplitude')
legend(loc='best')
if SAVEFIG:
    savefig("transrelf_alpha_fluid")

# %%Plot reflection and transmission coefficients (alpha_t axis)

figure()
plot(np.rad2deg(alpha_t.real), np.abs(reflection), label='reflection')
plot(np.rad2deg(alpha_t.real), np.abs(transmission_l), label='transmission L')
plot(np.rad2deg(alpha_t.real), np.abs(transmission_t), label='transmission T')
title('Tranmission and reflection coefficients from water to aluminium')
xlabel('angle of the transmitted wave T (deg)')
ylabel('coefficient amplitude')
legend(loc='best')
if SAVEFIG:
    savefig("transrelf_alpha_t")

# %%Plot reflection and transmission coefficients (alpha_l axis)

figure()
plot(np.rad2deg(alpha_l.real), np.abs(reflection), label='reflection')
plot(np.rad2deg(alpha_l.real), np.abs(transmission_l), label='transmission L')
plot(np.rad2deg(alpha_l.real), np.abs(transmission_t), label='transmission T')
title('Tranmission and reflection coefficients from water to aluminium')
xlabel('angle of the transmitted wave L (deg)')
ylabel('coefficient amplitude')
legend(loc='best')
if SAVEFIG:
    savefig("transrelf_alpha_l")

# %% Computation of the repartition of energy

# incident pressure (no effect because of normalisation, we keep it for clarity of the formulas)
pres_i = 1.

# cross section areas:
area_r = np.cos(alpha_fluid)
area_l = np.cos(alpha_l)
area_t = np.cos(alpha_t)

# Compute the energy incoming and outcoming: the principle of conservation of energy must be respected.
# Reference: Schmerr, §6.3.2

# Incoming energy
inc_energy = 0.5 * pres_i ** 2 / (rho_fluid * c_fluid) * area_r

# Outgoing energy
energy_refl = 0.5 * (reflection * pres_i) ** 2 / (rho_fluid * c_fluid) * area_r
energy_l = 0.5 * (transmission_l * pres_i) ** 2 / (rho_solid * c_l) * area_l
energy_t = 0.5 * (transmission_t * pres_i) ** 2 / (rho_solid * c_t) * area_t
out_energy = energy_refl + energy_l + energy_t

ratio_refl = np.abs(energy_refl / inc_energy)
ratio_l = np.abs(energy_l / inc_energy)
ratio_t = np.abs(energy_t / inc_energy)

# Verify the conservation of energy:
if not (np.allclose(ratio_refl + ratio_l + ratio_t, 1.0)):
    warnings.warn("The conservation of energy is not respected.")

# %% Plot energy

figure()
hold(True)
plot(np.rad2deg(alpha_fluid.real), ratio_refl, label='reflection')
plot(np.rad2deg(alpha_fluid.real), ratio_l, label='transmission L')
plot(np.rad2deg(alpha_fluid.real), ratio_t, label='transmission T')
plot(np.rad2deg(alpha_fluid.real), ratio_refl + ratio_l + ratio_t, '--', label='total')
axvline(x=np.rad2deg(critical_l), color='k', linestyle='--', label='critical angle L')
axvline(x=np.rad2deg(critical_t), color='k', linestyle='-', label='critical angle T')
title('Repartition of energy: water to aluminum interface')
xlabel('angle of the incident wave in water (deg)')
ylabel('normalised energy (1)')
# ylim([0, 1.05])
legend(loc='best')
if SAVEFIG:
    savefig("transrefl_energy")
