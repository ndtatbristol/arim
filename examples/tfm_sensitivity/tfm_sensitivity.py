#!/usr/bin/env python3
# encoding: utf-8
"""
TFM sensitivity in immersion inspection using a single-frequency LTI model.

The sensitivity on a point is defined as the TFM intensity that a defect centered
on this point would have.

Input: configuration files (conf.yaml, dryrun.yaml, debug.yaml).
Output: figures.

To see usage: call this script with '--help'
(in Spyder: Run > Configure > Command line options)
        
"""
import logging
import yaml
import pandas
from collections import OrderedDict
import warnings
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.helpers
import arim.path
import arim.plot as aplt
import arim.models.block_in_immersion as bim

#warnings.simplefilter("once", DeprecationWarning)

# %% Load configuration

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Start script")
logging.getLogger("arim").setLevel(logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--debug', action='store_true', help='use debug conf file')
parser.add_argument('-n', '--dryrun', action='store_true', help='use dry-run conf file')
parser.add_argument('-s', '--save', action='store_true', default=False,
                    help='save results')
args = parser.parse_args()

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
conf['save_to_csv'] = args.save

conf['plot.force_close'] = False

mpl.rcParams['savefig.dpi'] = 300

# %% Load probe

probe = arim.Probe.make_matrix_probe(**conf['probe'])

# Set probe reference point to first element
# put the first element in O(0,0,0), then it will be in (0,0,z) later.
probe.set_reference_element('first')
probe.translate_to_point_O()
probe.rotate(arim.geometry.rotation_matrix_y(np.deg2rad(conf['probe.angle'])))
probe.translate_to_point_O()
probe.translate([0, 0, conf['probe.standoff']])

# %% Set-up materials

couplant = arim.Material(**conf['material.couplant'])
block = arim.Material(**conf['material.block'])

wavelength_in_couplant = couplant.longitudinal_vel / probe.frequency
wavelength_l = block.longitudinal_vel / probe.frequency
wavelength_t = block.transverse_vel / probe.frequency

# -------------------------------------------------------------------------
# %% Define interfaces

probe_points, probe_orientations = arim.path.points_from_probe(probe)

frontwall_points, frontwall_orientations \
    = arim.path.points_1d_wall_z(**conf['interfaces.frontwall'], name='Frontwall')
backwall_points, backwall_orientations = \
    arim.path.points_1d_wall_z(**conf['interfaces.backwall'], name='Backwall')

grid = arim.geometry.Grid(**conf['interfaces.grid'], ymin=0., ymax=0.)
grid_points, grid_orientation = arim.path.points_from_grid(grid)

interfaces = arim.path.interfaces_for_block_in_immersion(couplant, probe_points,
                                                         probe_orientations,
                                                         frontwall_points,
                                                         frontwall_orientations,
                                                         backwall_points,
                                                         backwall_orientations,
                                                         grid_points, grid_orientation)

paths = arim.path.paths_for_block_in_immersion(block, couplant, interfaces,
                                               max_number_of_reflection=2)

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

arim.im.ray_tracing(views.values())

# %% Compute forward model
# Remark: results are cached in model.tx_ray_weights and model.rx_ray_weights

# Simulate a HMC inspection
# FMC give the same result but with a slower runtime.
tx, rx = arim.ut.hmc(probe.numelements)
scanline_weights = arim.ut.default_scanline_weights(tx, rx)

# Step 1: compute coefficients Q_i and Q'j of forward model
model_options = dict(frequency=probe.frequency,
                     probe_element_width=probe.dimensions.x[0],
                     use_beamspread=conf['model.use_beamspread'],
                     use_directivity=conf['model.use_directivity'],
                     use_transrefl=conf['model.use_transrefl'])
with arim.helpers.timeit("Computation of ray weights for all paths", logger=logger):
    ray_weights = bim.ray_weights_for_views(views, **model_options)
logger.info('Memory footprint of ray weights: {}'
    .format(arim.helpers.sizeof_fmt(ray_weights.nbytes)))


# Step 2: compute scattering matrices
# Compute model coefficients with scattering
if conf['scatterer']['type'] != 'circle':
    raise NotImplemented('will do soon (tm)')

with arim.helpers.timeit('Computation of scattering matrices'):
    if conf['model.use_scattering']:
        _, _, scat_matrices = arim.ut.elastic_scattering_2d_cylinder_matrices(
            conf['scattering.interpolation'],
            radius=conf['scatterer']['radius'],
            longitudinal_wavelength=wavelength_l,
            transverse_wavelength=wavelength_t)
    else:
        scat_matrices = arim.ut.elastic_scattering_point_source_matrices(wavelength_l, wavelength_t)

#%%
model_amplitudes_dict = OrderedDict()
sensitivity_dict = OrderedDict()
for viewname, view in views.items():
    with arim.helpers.timeit(f'Computation of sensitivity for view {viewname}'):

        # Step 3: prepare model amplitudes P_ij = Q_i Q'_j S_ij
        # These amplitudes are not actually computed here, otherwise we could run out
        # of memory.
        # model_amps[p][k] is the model amplitude of point p and scanline k.
        model_amps = arim.model.model_amplitudes_factory(
            tx, rx, view, ray_weights, scat_matrices)
        model_amplitudes_dict[viewname] = model_amps
    
        # Step 4: compute sensitivitiy
        sensitivity_dict[viewname] = np.abs(model_amps.sensitivity_uniform_tfm(scanline_weights))

ref_db = max(np.nanmax(np.abs(v)) for v in sensitivity_dict.values())

# %% Plot sensitivity
if conf['plot.sensitivity.all_in_one']:
    ref_db = max(np.nanmax(np.abs(v)) for v in sensitivity_dict.values())
    aplt.plot_oxz_many(sensitivity_dict.values(), grid, 7, 4,
                       title_list=sensitivity_dict.keys(),
                       suptitle='Sensitivity of TFM for a SDH (dB)',
                       scale='db', ref_db=ref_db,
                       filename='sensitivity', clim=[-60, 0])
        
# %% Plot sensitivity
if conf['plot.sensitivity']:
    for i, (viewname, view) in enumerate(views.items()):
        aplt.plot_oxz(sensitivity_dict[viewname], grid,
                      title='Sensitivity of view {} (dB)'.format(viewname),
                      scale='db', ref_db=ref_db,
                      filename=f'{i:02}_sensitivity_{viewname}', clim=[-40, 0])

# %% Measure sensitivity at point of interest

_pao = conf['point_of_interest']
pao_idx = grid_points.closest_point(_pao['x'], 0, _pao['z'])

tmp = []
for viewname, sensitivity in sensitivity_dict.items():
    tmp.append((viewname, sensitivity[pao_idx]))
out = pandas.DataFrame(tmp, columns='view sensitivity'.split())
out['sensitivity_db'] = arim.ut.decibel(out['sensitivity'])

if conf['save_to_csv']:
    out.to_csv('sensitivity.csv')
print(out)
#%%

viewnames = [view.name for view in views.values()]
ax = out[['view', 'sensitivity']].plot(kind='bar', figsize=(6, 7))
ax.set_ylabel('TFM intensity')
ax.set_title('Sensitivity at x={:.1f} and z={:.1f} mm (linear)'.format(
        _pao['x'] * 1e-3, _pao['z'] * 1e3))
ax.set_xticklabels(viewnames, rotation=45)
if aplt.conf['savefig']:
    ax.figure.savefig('sensitivity_bar')

