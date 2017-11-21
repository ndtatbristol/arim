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
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim, arim.ray, arim.io, arim.signal, arim.im
import arim.models.block_in_immersion as bim
import arim.plot as aplt
from arim.measurement import find_probe_loc_from_frontwall

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
parser.add_argument('-s', '--save', action='store_true', default=False,
                    help='save results')
args = parser.parse_args()


# print(args)

def load_configuration(debug=False, dryrun=False, **kwargs):
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

aplt.conf['savefig'] = args.save

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

# %% Set-up examination object and geometry

couplant = arim.Material(**conf['material.couplant'])
block = arim.Material(**conf['material.block'])
frontwall = \
    arim.geometry.points_1d_wall_z(**conf['interfaces.frontwall'], name='Frontwall')
backwall = \
    arim.geometry.points_1d_wall_z(**conf['interfaces.backwall'], name='Backwall')
frame.examination_object = arim.BlockInImmersion(block, couplant, frontwall, backwall)

grid = arim.geometry.Grid(**conf['interfaces.grid'], ymin=0., ymax=0.)
grid_p = grid.to_oriented_points()
area_of_interest = grid.points_in_rectbox(**conf['area_of_interest'])
reference_area = grid.points_in_rectbox(**conf['reference_area'])


# -------------------------------------------------------------------------
# %% Get the position of the probe from the pulse-echo data

# Prepare
frame_frontwall_meas = frame.apply_filter(arim.signal.Hilbert())

if conf['plot.bscan']:
    ax, imag = aplt.plot_bscan_pulse_echo(frame_frontwall_meas, clim=[-40, 0])

# Detect frontwall:
_, _, time_to_surface = \
    find_probe_loc_from_frontwall(frame_frontwall_meas, couplant, **conf['frontwall_meas'])

if conf['plot.frontwall_meas']:
    plt.figure()
    plt.plot(time_to_surface[frame_frontwall_meas.tx == frame_frontwall_meas.rx])
    plt.xlabel('element')
    plt.ylabel('time (µs)')
    plt.gca().yaxis.set_major_formatter(aplt.micro_formatter)
    plt.gca().yaxis.set_minor_formatter(aplt.micro_formatter)

    plt.title('time between elements and frontwall - must be a line!')

# %% Complete geometry definition

probe_p = frame.probe.to_oriented_points()
all_interfaces = [probe_p, frontwall, backwall, grid_p]

if conf['plot.interfaces']:
    aplt.plot_interfaces(all_interfaces, show_orientations=True, show_last=False)

# %% Make views
views = bim.make_views(frame.examination_object, probe_p, grid_p, **conf['views'])
if conf['views_to_use'] != 'all':
    views = OrderedDict([(viewname, view) for viewname, view in views.items()
                         if viewname in conf['views_to_use']])

# %% Perform ray tracing
    
arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

# %% Filter scanlines

with arim.helpers.timeit('Filtering scanlines'):
    frame_filtered = frame.apply_filter(
        arim.signal.Hilbert() + arim.signal.ButterworthBandpass(5, 3e6, 5.5e6, frame.time))

# Perform TFM on FMC rather than HMC (erroneous here) 
frame_filtered = frame_filtered.expand_frame_assuming_reciprocity()

# %% Run TFM

tfms = OrderedDict()
for i, view in enumerate(views.values()):
    with arim.helpers.timeit('TFM {}'.format(view.name), logger=logger):
        tfms[view.name] = arim.im.tfm.tfm_for_view(frame_filtered, grid, view,
                fillvalue=conf['tfm.fillvalue'])


# %% Plot all TFM

if conf['plot.tfm']:
    if conf['tfm.use_dynamic_scale']:
        scale = aplt.common_dynamic_db_scale([tfm.res for tfm in tfms.values()],
                                              reference_area)
    else:
        scale = itertools.repeat((None, None))

    for i, (viewname, view) in enumerate(views.items()):
        tfm = tfms[viewname]

        ref_db, clim = next(scale)

        ax, _ = aplt.plot_tfm(tfm, clim=clim, scale='db', ref_db=ref_db,
                              title='TFM {viewname}'.format(**locals()),
                              filename='tfm_{i:02}_{viewname}'.format(**locals()))

        if conf['plot.show_rays']:
            element_index = 0

            linestyle_tx = 'm--'
            linestyle_rx = 'c-.'

            aplt.draw_rays_on_click(grid, view.tx_path.rays, element_index, ax, linestyle_tx)
            aplt.draw_rays_on_click(grid, view.rx_path.rays, element_index, ax, linestyle_rx)

# Block script until windows are closed.
plt.show()

# %% Intensities in area of interest

print('Amplitudes (dB) in areas of interest:')
print()
columns = 'viewname intensity intensity_db'
tmp = []
for viewname, tfm in tfms.items():
    intensity = tfm.maximum_intensity_in_area(area_of_interest)
    tmp.append((viewname, intensity))
    # intensity_db = arim.ut.decibel(intensity, ref_db)
out = pandas.DataFrame(tmp, columns='view intensity'.split())
out['intensity_db'] = arim.ut.decibel(out['intensity'], ref_db)

if args.save:
    out.to_csv('intensities.csv')

print(out)
