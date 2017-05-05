# -*- coding: utf-8 -*-
"""
To evaluate the scattering matrix on a point, consider using
arim.ut.interpolate_scattering_matrix.

"""
import matplotlib.pyplot as plt
import arim
import numpy as np

vl = 6300
vt = 3100
freq = 5e6
radius = 0.1e-3

params = dict(longitudinal_wavelength=vl/freq,
              transverse_wavelength=vt/freq,
              radius=radius)

inc_theta, out_theta, matrices = \
    arim.ut.elastic_scattering_2d_cylinder_matrices(100, **params)


for key in ['LL', 'LT', 'TL', 'TT']:
    
    extent = (-180, 180, -180, 180)
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(matrices[key]), extent=extent, origin='lower')
    ax.set_xlabel('incident angle (degree)')
    ax.set_ylabel('outgoing angle (degree)')
    ax.set_title('Scattering matrix of a SDH - {}'.format(key))
    fig.colorbar(im, ax=ax)

