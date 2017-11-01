# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.scat

vl = 6300
vt = 3100
freq = 5e6
radius = 0.1e-3

params = dict(longitudinal_wavelength=vl/freq,
              transverse_wavelength=vt/freq,
              radius=radius)

# %% Scattering as matrices

n = 100  # number of points for discretising the inverval [-pi, pi[
scat_matrices = arim.scat.scat_2d_cylinder_matrices(n, **params)
for key in ['LL', 'LT', 'TL', 'TT']:
    extent = (-180, 180, -180, 180)
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(scat_matrices[key]), extent=extent, origin='lower')
    ax.set_xlabel('incident angle (degree)')
    ax.set_ylabel('outgoing angle (degree)')
    ax.set_title('Scattering matrix of a SDH - {}'.format(key))
    fig.colorbar(im, ax=ax)


# %% Scattering as functions of incident and scattered angles
scat_funcs = arim.scat.scat_2d_cylinder_funcs(**params)
fig, ax = plt.subplots()
theta = arim.scat.make_angles(n)
for key in ['LL', 'LT', 'TL', 'TT']:
    scat_func = scat_funcs[key]
    ax.plot(theta, np.abs(scat_func(0., theta)), label=key)
ax.legend()
ax.set_xlabel('scattering angle (degree)')
ax.set_ylabel('scattering amplitude (abs val.)')
ax.set_title('Scattering amplitudes for a SDH (incident angle: 0°)')

