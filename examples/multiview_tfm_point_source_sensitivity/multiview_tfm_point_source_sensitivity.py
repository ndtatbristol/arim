#!/usr/bin/env python3
# encoding: utf-8
"""
Computes for a block in immersion multi-view TFM using sensitivity maps 
(21 views).
Sensitivity maps take into account the beam spread, the directivity of
the elements, the transmission-reflection coefficients and the scattering
matrice of point sources (omnidirectional).

Input: configuration files (conf.yaml, dryrun.yaml, debug.yaml).
Output: figures, intensities as csv.
    
"""

raise Exception('this script must be refactored')
import logging
import yaml
import hashlib
import pandas
from collections import OrderedDict
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.geometry
import arim.helpers
import arim.models.block_in_immersion
import arim.path
import arim.plot as aplt
import arim.ray
from arim.measurement import find_probe_loc_from_frontwall
import arim.models.block_in_immersion as bim

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
    find_probe_loc_from_frontwall(frame, couplant, **conf['registration'])

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

interfaces = arim.models.block_in_immersion.make_interfaces(couplant, probe_points,
                                                            probe_orientations,
                                                            frontwall_points,
                                                            frontwall_orientations,
                                                            backwall_points,
                                                            backwall_orientations,
                                                            grid_points, grid_orientation)

paths = arim.models.block_in_immersion.make_paths(block, couplant, interfaces)

if conf['plot.interfaces']:
    aplt.plot_interfaces(interfaces.values(), show_orientations=True, show_grid=True)

for p in interfaces:
    logger.debug(p)

# Make views
views = arim.models.block_in_immersion.make_views_from_paths(paths)
if conf['views_to_use'] != 'all':
    views = OrderedDict([(viewname, view) for viewname, view in views.items()
                         if viewname in conf['views_to_use']])


# %% Setup Fermat solver and compute rays

arim.ray.ray_tracing(views.values())

# %% Compute forward model
# Remark: results are cached in model.tx_ray_weights and model.rx_ray_weights

tx_ray_weights_dict = OrderedDict()
rx_ray_weights_dict = OrderedDict()
tx_sensitivity_dict = OrderedDict()
rx_sensitivity_dict = OrderedDict()

all_tx_paths = {view.tx_path for view in views.values()}
all_rx_paths = {view.rx_path for view in views.values()}

model_options = dict(frequency=frame.probe.frequency,
                     probe_element_width=frame.probe.dimensions.x[0],
                     use_beamspread=conf['model.use_beamspread'],
                     use_directivity=conf['model.use_directivity'],
                     use_transrefl=conf['model.use_transrefl'])

for pathname, path in paths.items():
    if path not in all_tx_paths and path not in all_rx_paths:
        continue
    ray_geometry = arim.ray.RayGeometry.from_path(path)

    debug_data = OrderedDict()

    with arim.helpers.timeit('Ray weights for {}'.format(pathname), logger=logger):
        if path in all_tx_paths:
            ray_weights, ray_weights_debug = bim.tx_ray_weights(path, ray_geometry,
                                                                **model_options)

            sensitivity = arim.model.sensitivity_conjugate_for_path(ray_weights)
            assert path not in tx_ray_weights_dict
            assert path not in tx_sensitivity_dict
            tx_ray_weights_dict[path] = ray_weights
            tx_sensitivity_dict[path] = sensitivity
            debug_data['direct'] = dict(ray_weights=ray_weights_debug,
                                        sensitivity=sensitivity)
        if path in all_rx_paths:
            ray_weights, ray_weights_debug = bim.rx_ray_weights(path, ray_geometry,
                                                                **model_options)

            sensitivity = arim.model.sensitivity_conjugate_for_path(ray_weights)
            assert path not in rx_ray_weights_dict
            assert path not in rx_sensitivity_dict
            rx_ray_weights_dict[path] = ray_weights
            rx_sensitivity_dict[path] = sensitivity
            debug_data['reverse'] = dict(ray_weights=ray_weights_debug,
                                         sensitivity=sensitivity)

    if conf.get('debug', False):
        if pathname not in conf['debug.paths_to_show']:
            continue

        if conf.get('debug.angles', False):
            for i in range(1, ray_geometry.numinterfaces - 1):
                data = ray_geometry.conventional_inc_angle(i)
                interface_name = ray_geometry.interfaces[i].points.name
                aplt.plot_oxz(
                    np.rad2deg(data[0]), grid,
                    title='{pathname} conv. angle inc {i} ({interface_name})'.format(
                        **globals()),
                    savefig=False)

            # For directivity:
            i = 0
            data = ray_geometry.conventional_out_angle(i)
            aplt.plot_oxz(
                np.rad2deg(data[0]), grid,
                title='{pathname} conv. angle out {i}'.format(**globals()),
                savefig=False)

        for tx_or_rx, d in debug_data.items():
            if conf.get('debug.transrefl', False):
                i = 0
                aplt.plot_oxz(
                    np.abs(d['ray_weights']['transrefl'][i]), grid,
                    title='{tx_or_rx} {pathname} transrefl (abs) (elt {i})'.format(
                        **globals()),
                    savefig=False)
            if conf.get('debug.directivity', False):
                i = 0
                aplt.plot_oxz(
                    np.abs(d['ray_weights']['directivity'][i]), grid,
                    title='{tx_or_rx} {pathname} directivity (elt {i})'.format(
                        **globals()),
                    savefig=False)
            if conf.get('debug.beamspread', False):
                i = 0
                aplt.plot_oxz(
                    d['ray_weights']['beamspread'][i], grid,
                    title='{tx_or_rx} {pathname} beamspread (elt {i})'.format(
                        **globals()),
                    savefig=False)
            if conf.get('debug.sensitivity', False):
                aplt.plot_oxz(
                    d['sensitivity'], grid,
                    title='{tx_or_rx} {pathname} sensitivity'.format(**globals()),
                    savefig=False)
                # End of debug plots

    del ray_geometry, debug_data

# %% Plot sensitivity
if conf['plot.sensitivity_view']:
    view_sensitivity_dict = OrderedDict()
    for viewname, view in views.items():
        view_sensitivity_dict[viewname] = arim.model.sensitivity_conjugate_for_view(
            tx_sensitivity_dict[view.tx_path],
            rx_sensitivity_dict[view.rx_path]
        )

    ref_db = max(np.nanmax(np.abs(v)) for v in view_sensitivity_dict.values())
    aplt.plot_oxz_many(view_sensitivity_dict.values(), grid, 7, 3,
                       title_list=view_sensitivity_dict.keys(),
                       suptitle='Sensitivity', scale='db', ref_db=ref_db,
                       filename='sensitivity', clim=[-60, 0])
    del view_sensitivity_dict

# %% Setups TFM
frame.apply_filter(
    arim.signal.Hilbert() + arim.signal.ButterworthBandpass(5, 3e6, 5.5e6, frame.time))

tfms = []
for viewname, view in views.items():
    amps_tx = arim.im.SensitivityConjugateAmplitudes(
        frame, grid, tx_ray_weights_dict[view.tx_path],
        tx_sensitivity_dict[view.tx_path],
        divide_by_sensitivity=conf['tfm.divide_by_sensitivity']
    )
    amps_rx = arim.im.SensitivityConjugateAmplitudes(
        frame, grid, tx_ray_weights_dict[view.tx_path],
        rx_sensitivity_dict[view.rx_path],
        divide_by_sensitivity=conf['tfm.divide_by_sensitivity']
    )

    tfm = arim.im.SingleViewTFM(frame, grid, view, amplitudes_tx=amps_tx,
                                amplitudes_rx=amps_rx)
    tfms.append(tfm)

# %% Run all TFM

with arim.helpers.timeit('Delay-and-sum', logger=logger):
    for tfm in tfms:
        tfm.run(fillvalue=conf['tfm.fillvalue'])

# %% Plot all TFM

ref_db = 1.  # fallback is no TFM is plotted

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
