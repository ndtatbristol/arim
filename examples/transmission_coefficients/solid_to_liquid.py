"""
Compute and plot the transmission and reflection coefficients.

"""

import numpy as np
import warnings

import matplotlib.pyplot as plt

from arim.model import snell_angles, solid_l_fluid

# %% Parameters

SAVEFIG = False

# water:
c_fluid = 1480.0
rho_fluid = 1000.0

# aluminium :
c_l = 6320.0
c_t = 3130.0
rho_solid = 2700.0

# %% Computation of reflection and transmission coefficients

# Remark: by using complex angles, complex reflection and transmission coefficients
# are computed.
alpha_l = np.asarray(np.linspace(0, np.pi / 2, 50000), dtype=float)

alpha_fluid = snell_angles(alpha_l, c_l, c_fluid)
alpha_t = snell_angles(alpha_l, c_l, c_t)

reflection_l, reflection_t, transmission = solid_l_fluid(
    alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid, alpha_t
)

alpha_fluid_deg = np.rad2deg(alpha_fluid.real)
alpha_l_deg = np.rad2deg(alpha_l.real)
alpha_t_deg = np.rad2deg(alpha_t.real)

phase = lambda x: np.angle(x, deg=True)

# %% Plot reflection and transmission coefficients (alpha_fluid axis)

fig, axes = plt.subplots(nrows=2, sharex=True)
ax = axes[0]
ax.plot(alpha_l_deg, np.abs(transmission), label="transmission")
ax.plot(alpha_l_deg, np.abs(reflection_l), label="reflection L")
ax.plot(alpha_l_deg, np.abs(reflection_t), label="reflection T")
ax.set_title("Tranmission and reflection coefficients from solid L to liquid")
ax.set_ylabel("absolute value")
ax.legend(loc="best")

ax = axes[1]
ax.plot(alpha_l_deg, phase(transmission), label="transmission", linewidth=2)
ax.plot(alpha_l_deg, phase(reflection_l), label="reflection L")
ax.plot(alpha_l_deg, phase(reflection_t), label="reflection T")

ax.set_ylabel("phase (deg)")
ax.set_xlabel("angle of the incident L wave in solid (deg)")

if SAVEFIG:
    fig.savefig("solid_l_to_liquid_alpha_fluid")

# %% Computation of the repartition of energy

# incident pressure (no effect because of normalisation, we keep it for clarity of the formulas)
pres_i = 1.0

# cross section areas:
area_fluid = np.cos(alpha_fluid).real
area_l = np.cos(alpha_l).real
area_t = np.cos(alpha_t).real

# Compute the energy incoming and outcoming: the principle of conservation of energy must be respected.
# Reference: Schmerr, §6.3.2

# Incoming energy
inc_energy = 0.5 * pres_i**2 / (rho_solid * c_l) * area_l

# Outgoing energy
energy_trans = (
    0.5 * (np.abs(transmission) * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
)
energy_refl_l = 0.5 * (np.abs(reflection_l) * pres_i) ** 2 / (rho_solid * c_l) * area_l
energy_refl_t = 0.5 * (np.abs(reflection_t) * pres_i) ** 2 / (rho_solid * c_t) * area_t
out_energy = energy_trans + energy_refl_l + energy_refl_t

ratio_trans = energy_trans / inc_energy
ratio_refl_l = energy_refl_l / inc_energy
ratio_refl_t = energy_refl_t / inc_energy

# Verify the conservation of energy:
if not (np.allclose(ratio_trans + ratio_refl_l + ratio_refl_t, 1.0)):
    warnings.warn("The conservation of energy is not respected.")

# %% Plot energy

fig, ax = plt.subplots()
ax.plot(alpha_l_deg, ratio_trans, label="transmission")
ax.plot(alpha_l_deg, ratio_refl_l, label="reflection L")
ax.plot(alpha_l_deg, ratio_refl_t, label="reflection T")
ax.plot(alpha_l_deg, ratio_trans + ratio_refl_l + ratio_refl_t, "--", label="total")
ax.set_title("Repartition of energy: solid to liquid interface")
ax.set_xlabel("angle of the incident wave in liquid (deg)")
ax.set_ylabel("normalised energy (1)")
ax.set_xlabel("angle of the incident L wave in solid (deg)")
ax.set_ylim([0, 1.05])
ax.legend(loc="best")
if SAVEFIG:
    fig.savefig("solid_l_to_liquid_energy")
