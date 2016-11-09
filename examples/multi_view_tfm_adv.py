#!/usr/bin/env python3
# encoding: utf-8
"""
This script shows how to define advanced paths and how to use them for performing
multiview TFM.

Warning: this script can take up to several minutes to run and opens more than 20 windows.


"""
import logging
import time
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.plot as aplt

#%% Output and parameters

PLOT_TIME_TO_SURFACE = True
SHOW_RAY = True
PLOT_TFM = True
SAVEFIG = False

#%% Figure and logger
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['figure.figsize'] = [12., 6.]
mpl.rcParams['savefig.dpi'] = 300

logging.basicConfig()
logger = logging.getLogger("multi_view_tfm_adv")
logger.setLevel(logging.INFO)
logger.info("Start script")

#%% Load frame

expdata_filename = r'O:\arim-datasets\Aluminium_Notch_L12_128elts_5MHz\acq_02.mat'
frame = arim.io.load_expdata(expdata_filename)
frame.probe  = arim.probes['ima_50_MHz_128_1d']

# Set probe reference point to first element
# put the first element in O(0,0,0), then it will be in (0,0,z) later.
frame.probe.locations.translate(-frame.probe.locations[0], inplace=True)

#%% Set-up materials

water = arim.Material(longitudinal_vel=1480, transverse_vel=None, density=1000.,
                      state_of_matter='liquid', metadata={'long_name':'Water'})
aluminium = arim.Material(longitudinal_vel=6320., transverse_vel=3130., density=2700.,
                          state_of_matter='solid', metadata={'long_name':'Aluminium'})


# -------------------------------------------------------------------------
#%% Registration: get the position of the probe from the pulse-echo data

# Prepare registration
frame.apply_filter(arim.signal.Abs() + arim.signal.Hilbert())

ax, imag = aplt.plot_bscan_pulse_echo(frame, clim=[-40,0])

# Detect frontwall:
time_to_surface = arim.registration.detect_surface_from_extrema(frame, tmin=10e-6, tmax=30e-6)

if PLOT_TIME_TO_SURFACE:
    plt.figure()
    plt.plot(time_to_surface[frame.tx==frame.rx])
    plt.xlabel('element')
    plt.ylabel('time (µs)')
    plt.gca().yaxis.set_major_formatter(aplt.us_formatter)
    plt.gca().yaxis.set_minor_formatter(aplt.us_formatter)

    plt.title('time between elements and frontwall - must be a line!')


# Move probe:
distance_to_surface = time_to_surface * water.longitudinal_vel / 2
frame, iso = arim.registration.move_probe_over_flat_surface(frame, distance_to_surface, full_output=True)

logger.info('probe orientation: {:.2f}°'.format(np.rad2deg(iso.theta)))
logger.info('probe distance (min): {:.2f} mm'.format(-1e3*iso.z_o))

# -------------------------------------------------------------------------
#%% Define interfaces

numinterface = 1000
numinterface2 = 1000

probe = frame.probe.locations
probe.name = 'Probe'

xmin = -20e-3
xmax = 100e-3

frontwall = arim.geometry.Points.from_xyz(
    x=np.linspace(xmin, xmax, numinterface),
    y=np.zeros((numinterface, ), dtype=np.float),
    z=np.zeros((numinterface, ), dtype=np.float),
    name='Frontwall')

backwall = arim.geometry.Points.from_xyz(
    x=np.linspace(xmin, xmax, numinterface2),
    y=np.zeros((numinterface2, ), dtype=np.float),
    z=np.full((numinterface2, ), 40.18e-3, dtype=np.float),
    name='Backwall')


grid = arim.geometry.Grid(xmin, xmax,
                          ymin=0., ymax=0.,
                          zmin=0., zmax=45e-3,
                          pixel_size=1e-3)


# Get a full basis per point:
probe_orientation = arim.geometry.Points(np.zeros((*probe.shape, 3, 3)), 'ProbeOrientation')
probe_orientation.x[...] = frame.probe.pcs.i_hat
probe_orientation.y[...] = frame.probe.pcs.j_hat
probe_orientation.z[...] = frame.probe.pcs.k_hat

frontwall_orientation = arim.geometry.Points(np.zeros((*frontwall.shape, 3, 3)), 'FrontwallOrientation')
frontwall_orientation.coords[..., :, :] = np.identity(3)

backwall_orientation = arim.geometry.Points(np.zeros((*backwall.shape, 3, 3)), 'BackwallOrientation')
backwall_orientation.coords[..., :, :] = np.identity(3)

grid_orientation = arim.geometry.Points(np.zeros((*grid.as_points.shape, 3, 3)), 'GridOrientation')
grid_orientation.coords[..., :, :] = np.identity(3)

probe_interface = arim.Interface(
    probe, probe_orientation, are_normals_on_out_rays_side=True)
frontwall_interface = arim.Interface(
    frontwall, frontwall_orientation, 'fluid_solid',
    'transmission',
    are_normals_on_inc_rays_side=False, are_normals_on_out_rays_side=True)
backwall_interface = arim.Interface(
    backwall, backwall_orientation, 'solid_fluid',
    'reflection', reflection_against=water,
    are_normals_on_inc_rays_side=True, are_normals_on_out_rays_side=True)
grid_interface = arim.Interface(grid.as_points, grid_orientation, are_normals_on_inc_rays_side=False)
    
    
for p in [probe_interface, frontwall_interface, backwall_interface, grid_interface]:
    logger.debug(p)
    


#%% Plot interface
def plot_interface(title=None, show_grid=True, ax=None, element_normal=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    interfaces = [probe, frontwall, backwall]
    if show_grid:
        interfaces += [grid.as_points]
    for (interface, marker) in zip(interfaces, ['o', '.', '.', '.k']):
        ax.plot(interface.x, interface.z, marker, label=interface.name)
    if ax.get_xlabel() == '':
        ax.set_xlabel('x (mm)')
    if ax.get_ylabel() == '':
        ax.set_ylabel('z (mm)')

    ax.set_title(title)
    ax.grid()

    if element_normal:
        k = 4# one every k arrows
        ax.quiver(frame.probe.locations.x[::k],
                  frame.probe.locations.z[::k],
                  frame.probe.orientations.x[::k],
                  frame.probe.orientations.z[::k],
                  units='xy', angles='xy',
                  width=0.0003, color='c')


    ylim = ax.get_ylim()
    if ylim[0] < ylim[1]:
        ax.invert_yaxis()

    ax.axis('equal')
    fig.show()
    return ax

plot_interface("Interfaces", show_grid=False, element_normal=True)

# -------------------------------------------------------------------------
#%% Setup views

paths = OrderedDict()

paths['L'] = arim.Path(
    interfaces=(probe_interface, frontwall_interface, grid_interface),
    materials=(water, aluminium),
    modes=('L', 'L'),
    name='L')

paths['T'] = arim.Path(
    interfaces=(probe_interface, frontwall_interface, grid_interface),
    materials=(water, aluminium),
    modes=('L', 'T'),
    name='T')

paths['LL'] = arim.Path(
    interfaces=(probe_interface, frontwall_interface, backwall_interface, grid_interface),
    materials=(water, aluminium, aluminium),
    modes=('L', 'L', 'L'),
    name='LL')

paths['LT'] = arim.Path(
    interfaces=(probe_interface, frontwall_interface, backwall_interface, grid_interface),
    materials=(water, aluminium, aluminium),
    modes=('L', 'L', 'T'),
    name='LT')

paths['TL'] = arim.Path(
    interfaces=(probe_interface, frontwall_interface, backwall_interface, grid_interface),
    materials=(water, aluminium, aluminium),
    modes=('L', 'T', 'L'),
    name='TL')

paths['TT'] = arim.Path(
    interfaces=(probe_interface, frontwall_interface, backwall_interface, grid_interface),
    materials=(water, aluminium, aluminium),
    modes=('L', 'T', 'T'),
    name='TT')

# Make views
views = []
for view_name in arim.im.IMAGING_MODES:
    tx_name, rx_name = view_name.split('-')
    tx_path = arim.im.FermatPath.from_path(paths[tx_name])
    rx_path = arim.im.FermatPath.from_path(paths[rx_name])
    views.append(arim.im.View(tx_path, rx_path, view_name))


#%% Setup Fermat solver and compute rays

fermat_solver = arim.im.FermatSolver.from_views(views)
rays = fermat_solver.solve()

#%% Setups TFM
frame.apply_filter(arim.signal.Hilbert() + arim.signal.ButterworthBandpass(5, 3e6, 5.5e6, frame.time))

tfms = []
for i, view in enumerate(views):
    rays_tx = rays[view.tx_path]
    rays_rx = rays[view.rx_path]

    amps_tx = arim.im.UniformAmplitudes(frame, grid)
    amps_rx = arim.im.UniformAmplitudes(frame, grid)

    tfm = arim.im.MultiviewTFM(frame, grid, view, rays_tx, rays_rx,
                          amplitudes_tx=amps_tx, amplitudes_rx=amps_rx, fillvalue=0.)
    tfms.append(tfm)


#%% Run all TFM

tic = time.clock()
for tfm in tfms:
    tfm.run()
toc = time.clock()
logger.info("Performed {} delay-and-sum's in {:.2f} s".format(len(tfms), toc-tic))

#%% Plot all TFM

if PLOT_TFM:
    func_res = lambda x: arim.utils.decibel(x)
    # func_res = lambda x: np.imag(x)
    clim = [-40, 0]

    for i, tfm in enumerate(tfms):
        view = tfm.view

        ax, _ = aplt.plot_tfm(tfm, clim=clim, func_res=func_res)
        ax = plot_interface(view.name, show_grid=False, ax=ax)

        if SHOW_RAY:
            element_index = 0

            linestyle_tx = 'm--'
            linestyle_rx = 'c-.'

            aplt.draw_rays_on_click(grid, tfm.rays_tx, element_index, ax, linestyle_tx)
            aplt.draw_rays_on_click(grid, tfm.rays_rx, element_index, ax, linestyle_rx)

        ax.axis([-20e-3, 100e-3, 45e-3,-20e-3])

        if SAVEFIG:
            ax.figure.savefig("fig_{:02}_{}.png".format(i, view.name), bbox_inches='tight')

# Block script until windows are closed.
plt.show()
