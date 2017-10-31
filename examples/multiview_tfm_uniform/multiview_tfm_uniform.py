#!/usr/bin/env python3
# encoding: utf-8
"""
Computes for a block in immersion multi-view TFM with uniform amplitudes (21 views).

Input: configuration files (conf.yaml, dryrun.yaml, debug.yaml).
Output: figures, intensities as csv.
    
"""
import logging
import yaml
import hashlib
import pandas
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.geometry
import arim.plot as aplt
import arim.ray
from arim.registration import registration_by_flat_frontwall_detection

# %% Load configuration

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Start script")
logging.getLogger("arim").setLevel(logging.INFO)

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--debug', action='store_true', help='use debug conf file')
parser.add_argument('-n', '--dryrun', action='store_true', help='use dry-run conf file')
args = parser.parse_args()


# print(args)

def load_configuration(debug=False, dryrun=False):
    with open('conf.yaml', 'rb') as f:
        conf = arim.config.Config(yaml.load(f))

    if dryrun:
        try:
            with open('dryrun.yaml', 'rb') as f:
                conf.merge(yaml.load(f))
        except FileNotFoundError:
            logger.warning('Dry-run configuration not found')
    if debug:
        try:
            with open('debug.yaml', 'rb') as f:
                conf.merge(yaml.load(f))
        except FileNotFoundError:
            logger.warning('Debug configuration not found')
    try:
        module_conf = conf['arim.plot']
    except KeyError:
        pass
    else:
        aplt.conf.merge(module_conf)

    return conf


conf = load_configuration(**vars(args))
# print(yaml.dump(dict(conf), default_flow_style=False))

# %% Figure and logger
mpl.rcParams['savefig.dpi'] = 300

# %% Load frame

expdata_filename = conf['frame.datafile']
frame = arim.io.load_expdata(expdata_filename)
frame.probe = arim.Probe.make_matrix_probe(**conf['probe'])

# Set probe reference point to first element
# put the first element in O(0,0,0), then it will be in (0,0,z) later.
frame.probe.set_reference_element('first')
frame.probe.translate_to_point_O()

# %% Set-up materials

couplant = arim.Material(**conf['material.couplant'])
block = arim.Material(**conf['material.block'])

wavelength_in_couplant = couplant.longitudinal_vel / frame.probe.frequency

# -------------------------------------------------------------------------
# %% Registration: get the position of the probe from the pulse-echo data

# Prepare registration
frame.apply_filter(arim.signal.Abs() + arim.signal.Hilbert())

if conf['plot.bscan']:
    ax, imag = aplt.plot_bscan_pulse_echo(frame, clim=[-40, 0])

# Detect frontwall:
_, _, time_to_surface = \
    registration_by_flat_frontwall_detection(frame, couplant, **conf['registration'])

if conf['plot.registration']:
    plt.figure()
    plt.plot(time_to_surface[frame.tx == frame.rx])
    plt.xlabel('element')
    plt.ylabel('time (µs)')
    plt.gca().yaxis.set_major_formatter(aplt.micro_formatter)
    plt.gca().yaxis.set_minor_formatter(aplt.micro_formatter)

    plt.title('time between elements and frontwall - must be a line!')

# -------------------------------------------------------------------------
# %% Define interfaces

probe_points, probe_orientations = arim.geometry.points_from_probe(frame.probe)

frontwall_points, frontwall_orientations \
    = arim.geometry.points_1d_wall_z(**conf['interfaces.frontwall'], name='Frontwall')
backwall_points, backwall_orientations = \
    arim.geometry.points_1d_wall_z(**conf['interfaces.backwall'], name='Backwall')

grid = arim.geometry.Grid(**conf['interfaces.grid'], ymin=0., ymax=0.)
grid_points, grid_orientation = arim.geometry.points_from_grid(grid)
area_of_interest = grid.points_in_rectbox(**conf['area_of_interest'])
reference_area = grid.points_in_rectbox(**conf['reference_area'])

interfaces = arim.path.interfaces_for_block_in_immersion(couplant, probe_points,
                                                         probe_orientations,
                                                         frontwall_points,
                                                         frontwall_orientations,
                                                         backwall_points,
                                                         backwall_orientations,
                                                         grid_points, grid_orientation)

paths = arim.path.paths_for_block_in_immersion(block, couplant, interfaces)

if conf['plot.interfaces']:
    aplt.plot_interfaces(interfaces.values(), show_orientations=True, show_grid=True)

for p in interfaces:
    logger.debug(p)

# Make views
views = arim.path.views_for_block_in_immersion(paths)
if conf['views_to_use'] != 'all':
    views = OrderedDict([(viewname, view) for viewname, view in views.items()
                         if viewname in conf['views_to_use']])


# %% Setup Fermat solver and compute rays

arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

# %% Setups TFM
frame.apply_filter(
    arim.signal.Hilbert() + arim.signal.ButterworthBandpass(5, 3e6, 5.5e6, frame.time))

tfms = []
for i, view in enumerate(views.values()):
    amps_tx = arim.im.UniformAmplitudes(frame, grid)
    amps_rx = arim.im.UniformAmplitudes(frame, grid)

    tfm = arim.im.SingleViewTFM(frame, grid, view,
                                amplitudes_tx=amps_tx, amplitudes_rx=amps_rx)
    tfms.append(tfm)

# %% Run all TFM

with arim.helpers.timeit('Delay-and-sum', logger=logger):
    for tfm in tfms:
        tfm.run(fillvalue=conf['tfm.fillvalue'])

# %% Plot all TFM

if conf['plot.tfm']:
    # func_res = lambda x: np.imag(x)
    if conf['tfm.use_dynamic_scale']:
        scale = aplt.common_dynamic_db_scale([tfm.res for tfm in tfms], reference_area)
    else:
        scale = itertools.repeat((None, None))

    for i, tfm in enumerate(tfms):
        view = tfm.view
        viewname = tfm.view.name

        ref_db, clim = next(scale)

        ax, _ = aplt.plot_tfm(tfm, clim=clim, scale='db', ref_db=ref_db,
                              title='TFM {viewname}'.format(**locals()),
                              filename='tfm_{i:02}_{viewname}'.format(**locals()))

        if conf['plot.show_rays']:
            element_index = 0

            linestyle_tx = 'm--'
            linestyle_rx = 'c-.'

            aplt.draw_rays_on_click(grid, tfm.tx_rays, element_index, ax, linestyle_tx)
            aplt.draw_rays_on_click(grid, tfm.rx_rays, element_index, ax, linestyle_rx)

        if conf['plot.force_close']:
            plt.close(ax.figure)

# Block script until windows are closed.
plt.show()

# %% Intensities in area of interest

print('Amplitudes (dB) in areas of interest:')
print()
columns = 'viewname intensity intensity_db'
tmp = []
for tfm in tfms:
    viewname = tfm.view.name
    intensity = tfm.maximum_intensity_in_area(area_of_interest)
    tmp.append((viewname, intensity))
    # intensity_db = arim.ut.decibel(intensity, ref_db)
out = pandas.DataFrame(tmp, columns='view intensity'.split())
out['intensity_db'] = arim.ut.decibel(out['intensity'], ref_db)

if conf['save_to_csv']:
    out.to_csv('intensities.csv')

print(out)
print()
print('hash: {}'.format(hashlib.md5(str(out).encode()).hexdigest()))
