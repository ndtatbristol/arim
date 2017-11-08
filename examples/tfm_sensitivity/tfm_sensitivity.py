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
from collections import OrderedDict
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.model, arim.ray, arim.scat
import arim.plot as aplt
import arim.models.block_in_immersion as bim

# warnings.simplefilter("once", DeprecationWarning)

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

# %% Set-up examination object and geometry

couplant = arim.Material(**conf['material.couplant'])
block = arim.Material(**conf['material.block'])
frontwall = \
    arim.geometry.points_1d_wall_z(**conf['interfaces.frontwall'], name='Frontwall')
backwall = \
    arim.geometry.points_1d_wall_z(**conf['interfaces.backwall'], name='Backwall')
exam_obj = arim.BlockInImmersion(block, couplant, frontwall, backwall)

probe_p = probe.to_oriented_points()
grid = arim.geometry.Grid(**conf['interfaces.grid'], ymin=0., ymax=0.)
grid_p = grid.to_oriented_points()

all_interfaces = [probe_p, frontwall, backwall, grid_p]

if conf['plot.interfaces']:
    aplt.plot_interfaces(all_interfaces, show_orientations=True, show_last=False)

# %% Make views
    
views = bim.make_views(exam_obj, probe_p, grid_p, **conf['views'])
if conf['views_to_use'] != 'all':
    views = OrderedDict([(viewname, view) for viewname, view in views.items()
                         if viewname in conf['views_to_use']])

# %% Perform ray tracing

with arim.helpers.timeit('Ray tracing'):
    arim.ray.ray_tracing(views.values())

# %% Compute scattering matrices

scat_params = conf['scatterer'].copy()
scat_angle = np.deg2rad(scat_params.pop('scat_angle_deg'))
with arim.helpers.timeit('Computation of scattering matrices'):
    scat_obj = arim.scat.scat_factory(material=block, **scat_params)
    try:
        numangles = scat_obj.numangles  # if loaded from data, use all angles
    except AttributeError:
        numangles = conf['scattering.numangles']
    scat_matrices = scat_obj.as_single_freq_matrices(probe.frequency, numangles)

# %% Compute ray weights

tx, rx = arim.ut.fmc(probe.numelements)
scanline_weights = arim.ut.default_scanline_weights(tx, rx)

model_options = dict(frequency=probe.frequency,
                     probe_element_width=probe.dimensions.x[0],
                     use_beamspread=conf['model.use_beamspread'],
                     use_directivity=conf['model.use_directivity'],
                     use_transrefl=conf['model.use_transrefl'])

with arim.helpers.timeit("Ray weights for all paths", logger=logger):
    ray_weights = bim.ray_weights_for_views(views, **model_options)
logger.info('Memory footprint of ray weights: {}'
            .format(arim.helpers.sizeof_fmt(ray_weights.nbytes)))

# %% Compute sensitivity
sensitivity_dict = OrderedDict()
for viewname, view in views.items():
    with arim.helpers.timeit(f'Computation of sensitivity for view {viewname}'):
        # These amplitudes are not actually computed here, otherwise we could run out
        # of memory.
        # model_amps[p][k] is the model amplitude of point p and scanline k.
        model_amps = arim.model.model_amplitudes_factory(
            tx, rx, view, ray_weights, scat_matrices, scat_angle)

        # Step 4: compute sensitivitiy
        sensitivity_dict[viewname] = np.abs(
            model_amps.sensitivity_uniform_tfm(scanline_weights))

ref_db = max(np.nanmax(np.abs(v)) for v in sensitivity_dict.values())

# %% Plot sensitivity
if conf['plot.sensitivity.all_in_one']:
    ref_db = max(np.nanmax(np.abs(v)) for v in sensitivity_dict.values())
    aplt.plot_oxz_many(sensitivity_dict.values(), grid, 7, 4,
                       title_list=sensitivity_dict.keys(),
                       suptitle='Sensitivity of TFM for {} (dB)'.format(scat_params['kind']),
                       scale='db', ref_db=ref_db,
                       filename='sensitivity', clim=[-40, 0])

# %% Plot sensitivity
if conf['plot.sensitivity']:
    for i, (viewname, view) in enumerate(views.items()):
        aplt.plot_oxz(sensitivity_dict[viewname], grid,
                      title='Sensitivity of view {} (dB)'.format(viewname),
                      scale='db', ref_db=ref_db,
                      filename=f'{i:02}_sensitivity_{viewname}', clim=[-40, 0])
