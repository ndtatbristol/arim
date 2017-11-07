# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.scat

vl = 6300
vt = 3100
freq = 5e6
radius = 0.1e-3

# %% Scattering as matrices

numangles = 100  # number of points for discretising the inverval [-pi, pi[
scat_obj = arim.scat.SdhScat(radius, vl, vt)
scat_matrices = scat_obj.as_single_freq_matrices(freq, numangles)

for key in ['LL', 'LT', 'TL', 'TT']:
    extent = (-180, 180, -180, 180)
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(scat_matrices[key]), extent=extent, origin='lower')
    ax.set_xlabel('incident angle (degree)')
    ax.set_ylabel('outgoing angle (degree)')
    ax.set_title('Scattering matrix of a SDH - {}'.format(key))
    fig.colorbar(im, ax=ax)


# %% Scattering as functions of incident and scattered angles

fig, ax = plt.subplots()
theta_in = 0.
theta_out = np.linspace(0, np.pi, 100)
scat_vals = scat_obj(theta_in, theta_out, freq)
for key in ['LL', 'LT', 'TL', 'TT']:
    ax.plot(np.rad2deg(theta_out), np.abs(scat_vals[key]), label=key)
ax.legend()
ax.set_xlabel('scattering angle (degree)')
ax.set_ylabel('scattering amplitude (abs val.)')
ax.set_title('Scattering amplitudes for a SDH (incident angle: {}°)'.format(np.rad2deg(theta_in)))

