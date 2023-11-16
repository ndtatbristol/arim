"""
Compute and plot the transmission and reflection coefficients.

Case: an incident longitudinal in the fluid hits the wall of an solid part.
"""

import numpy as np
import warnings

import matplotlib.pyplot as plt

from arim.model import snell_angles, fluid_solid

# %% Parameters

# Transmission coefficients above the critical angles corresponds to
# inhomogeneous surface waves instead of bulk waves
# Set to True to force the coefficients to zero.
NULL_TRANSMISSION_ABOVE_CRIT_ANGLES = False
SAVEFIG = False

# water:
c_fluid = 1480.0
rho_fluid = 1000.0

# aluminium :
c_l = 6320.0
c_t = 3130.0
rho_solid = 2700.0

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

reflection, transmission_l, transmission_t = fluid_solid(
    alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_l, alpha_t
)

alpha_fluid_deg = np.rad2deg(alpha_fluid.real)
alpha_l_deg = np.rad2deg(alpha_l.real)
alpha_t_deg = np.rad2deg(alpha_t.real)

if NULL_TRANSMISSION_ABOVE_CRIT_ANGLES:
    transmission_l[alpha_fluid > critical_l] = 0
    transmission_t[alpha_fluid > critical_t] = 0

# %% Plot reflection and transmission coefficients (alpha_fluid axis)

fig, axes = plt.subplots(nrows=2, sharex=True)
ax = axes[0]
ax.plot(alpha_fluid_deg, np.abs(reflection), label="reflection")
ax.plot(alpha_fluid_deg, np.abs(transmission_l), label="transmission L")
ax.plot(alpha_fluid_deg, np.abs(transmission_t), label="transmission T")
ax.set_title("Tranmission and reflection coefficients from liquid to solid")
ax.axvline(
    x=np.rad2deg(critical_l), color="k", linestyle="--", label="critical angle L"
)
ax.axvline(x=np.rad2deg(critical_t), color="k", linestyle="-", label="critical angle T")
ax.set_ylabel("absolute value")
ax.legend(loc="best")

ax = axes[1]
ax.plot(alpha_fluid_deg, np.angle(reflection, True), label="reflection")
ax.plot(alpha_fluid_deg, np.angle(transmission_l, True), label="transmission L")
ax.plot(alpha_fluid_deg, np.angle(transmission_t, True), label="transmission T")
ax.axvline(
    x=np.rad2deg(critical_l), color="k", linestyle="--", label="critical angle L"
)
ax.axvline(x=np.rad2deg(critical_t), color="k", linestyle="-", label="critical angle T")

ax.set_ylabel("phase (deg)")
ax.set_xlabel("angle of the incident wave in liquid (deg)")

if SAVEFIG:
    fig.savefig("liquid_to_solid_alpha_fluid")

# %%Plot reflection and transmission coefficients (alpha_t axis)

fig, axes = plt.subplots(nrows=2, sharex=True)
ax = axes[0]
ax.plot(alpha_t_deg, np.abs(reflection), label="reflection")
ax.plot(alpha_t_deg, np.abs(transmission_l), label="transmission L")
ax.plot(alpha_t_deg, np.abs(transmission_t), label="transmission T")
ax.set_title("Tranmission and reflection coefficients from liquid to solid")
ax.set_ylabel("absolute value")
ax.legend(loc="best")

ax = axes[1]
ax.plot(alpha_t_deg, np.angle(reflection, True), label="reflection")
ax.plot(alpha_t_deg, np.angle(transmission_l, True), label="transmission L")
ax.plot(alpha_t_deg, np.angle(transmission_t, True), label="transmission T")

ax.set_ylabel("phase (deg)")
ax.set_xlabel("angle of the transmitted wave T (deg)")

if SAVEFIG:
    fig.savefig("liquid_to_solid_alpha_t")

# %%Plot reflection and transmission coefficients (alpha_l axis)

fig, axes = plt.subplots(nrows=2, sharex=True)
ax = axes[0]
ax.plot(alpha_l_deg, np.abs(reflection), label="reflection")
ax.plot(alpha_l_deg, np.abs(transmission_l), label="transmission L")
ax.plot(alpha_l_deg, np.abs(transmission_t), label="transmission T")
ax.set_title("Tranmission and reflection coefficients from liquid to solid")
ax.set_ylabel("absolute value")
ax.legend(loc="best")

ax = axes[1]
ax.plot(alpha_l_deg, np.angle(reflection, True), label="reflection")
ax.plot(alpha_l_deg, np.angle(transmission_l, True), label="transmission L")
ax.plot(alpha_l_deg, np.angle(transmission_t, True), label="transmission T")

ax.set_ylabel("phase (deg)")
ax.set_xlabel("angle of the transmitted wave L (deg)")

if SAVEFIG:
    fig.savefig("liquid_to_solid_alpha_l")

# %% Computation of the repartition of energy

# incident pressure (no effect because of normalisation, we keep it for clarity of the formulas)
pres_i = 10000.0

# cross section areas:
area_r = np.cos(alpha_fluid).real
area_l = np.cos(alpha_l).real
area_t = np.cos(alpha_t).real

# Compute the energy incoming and outcoming: the principle of conservation of
# energy must be respected.
# Reference: Schmerr, §6.3.2
# See also Schmerr, §6.2.5 for the considerations on the inhomogeneous waves
# which propagate after the critical angle.

# Incoming energy
inc_energy = 0.5 * pres_i**2 / (rho_fluid * c_fluid) * area_r

# Outgoing energy
energy_refl = 0.5 * (np.abs(reflection) * pres_i) ** 2 / (rho_fluid * c_fluid) * area_r
energy_l = 0.5 * (np.abs(transmission_l) * pres_i) ** 2 / (rho_solid * c_l) * area_l
energy_t = 0.5 * (np.abs(transmission_t) * pres_i) ** 2 / (rho_solid * c_t) * area_t
out_energy = energy_refl + energy_l + energy_t

ratio_refl = energy_refl / inc_energy
ratio_l = energy_l / inc_energy
ratio_t = energy_t / inc_energy

# Verify the conservation of energy:
if not (np.allclose(ratio_refl + ratio_l + ratio_t, 1.0)):
    warnings.warn("The conservation of energy is not respected.")

# %% Plot energy

fig, ax = plt.subplots()
ax.plot(alpha_fluid_deg, ratio_refl, label="reflection")
ax.plot(alpha_fluid_deg, ratio_l, label="transmission L")
ax.plot(alpha_fluid_deg, ratio_t, label="transmission T")
ax.plot(alpha_fluid_deg, ratio_refl + ratio_l + ratio_t, "--", label="total")
ax.axvline(
    x=np.rad2deg(critical_l), color="k", linestyle="--", label="critical angle L"
)
ax.axvline(x=np.rad2deg(critical_t), color="k", linestyle="-", label="critical angle T")
ax.set_title("Repartition of energy of the bulk waves: liquid to solid interface")
ax.set_xlabel("angle of the incident wave in liquid (deg)")
ax.set_ylabel("normalised energy (1)")
# ylim([0, 1.05])
ax.legend(loc="best")
if SAVEFIG:
    fig.savefig("liquid_to_solid_energy")
