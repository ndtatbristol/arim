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
import logging
import yaml
import hashlib
import pandas
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.helpers
import arim.path
import arim.plot as aplt
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
        module_conf = conf.pop('arim.plot')
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

expdata_filename = conf.pop('frame.datafile')
frame = arim.io.load_expdata(expdata_filename)
frame.probe = arim.Probe.make_matrix_probe(**conf.pop('probe'))

# Set probe reference point to first element
# put the first element in O(0,0,0), then it will be in (0,0,z) later.
frame.probe.set_reference_element('first')
frame.probe.translate_to_point_O()

# %% Set-up materials

couplant = arim.Material(**conf.pop('material.couplant'))
block = arim.Material(**conf.pop('material.block'))

wavelength_in_couplant = couplant.longitudinal_vel / frame.probe.frequency

# -------------------------------------------------------------------------
# %% Registration: get the position of the probe from the pulse-echo data

# Prepare registration
frame.apply_filter(arim.signal.Abs() + arim.signal.Hilbert())

if conf['plot.bscan']:
    ax, imag = aplt.plot_bscan_pulse_echo(frame, clim=[-40, 0])

# Detect frontwall:
_, _, time_to_surface = \
    registration_by_flat_frontwall_detection(frame, couplant, **conf.pop('registration'))

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

probe_points, probe_orientations = arim.path.points_from_probe(frame.probe)

frontwall_points, frontwall_orientations \
    = arim.path.points_1d_wall_z(**conf.pop('interfaces.frontwall'), name='Frontwall')
backwall_points, backwall_orientations = \
    arim.path.points_1d_wall_z(**conf.pop('interfaces.backwall'), name='Backwall')

grid = arim.geometry.Grid(**conf.pop('interfaces.grid'), ymin=0., ymax=0.)
grid_points, grid_orientation = arim.path.points_from_grid(grid)
area_of_interest = grid.points_in_rectbox(**conf.pop('area_of_interest'))
reference_area = grid.points_in_rectbox(**conf.pop('reference_area'))

interfaces = arim.path.interfaces_for_block_in_immersion(couplant, probe_points,
                                                         probe_orientations,
                                                         frontwall_points,
                                                         frontwall_orientations,
                                                         backwall_points,
                                                         backwall_orientations,
                                                         grid_points, grid_orientation)

paths = arim.path.paths_for_block_in_immersion(block, couplant, *interfaces)

if conf.pop('plot.interfaces'):
    aplt.plot_interfaces(interfaces, show_orientations=True, show_grid=True)

for p in interfaces:
    logger.debug(p)

# Make views
views = arim.path.views_for_block_in_immersion(paths)

# %% Setup Fermat solver and compute rays

arim.im.ray_tracing(views.values())

# %% Precompute ray geometry

# Precomputing is not mandatory but is helpful for performance monitoring.
with arim.helpers.timeit('Computation of ray geometry', logger=logger):
    ray_geometry_dict = OrderedDict()
    for path in paths.values():
        ray_geometry = arim.path.RayGeometry.from_path(path)
        with ray_geometry.precompute():
            # For directivity:
            ray_geometry.conventional_out_angle(0)
            
            # For beamspread and transmission-reflection:
            for i in range(path.numinterfaces - 1):
                ray_geometry.conventional_inc_angle(i)

        ray_geometry_dict[path.name] = ray_geometry

# %% Debug angle

if conf.get('debug', None):
    for pathname, path in paths.items():
        if pathname not in ['L', 'LT']:
            continue
        ray_geometry = ray_geometry_dict[pathname]
        for i in range(ray_geometry.numinterfaces):
            data = ray_geometry.conventional_inc_angle(i)
            if data is None:
                continue
            aplt.plot_oxz(np.rad2deg(data[0]), grid,
                          title='{pathname} conv. angle inc {i}'.format(**globals()),
                          savefig=False)
        # For scattering:
        data = ray_geometry.signed_inc_angle(i)
        aplt.plot_oxz(np.rad2deg(data[0]), grid,
                      title='{pathname} signed angle inc {i}'.format(**globals()),
                      savefig=False)
        

        # For directivity:
        i = 0
        data = ray_geometry.conventional_out_angle(i)
        
        if data is None:
            continue
        aplt.plot_oxz(np.rad2deg(data[0]), grid,
                      title='{pathname} conv. angle out {i}'.format(**globals()),
                      savefig=False)


# %% Computation ray weights

ray_weights_dict = OrderedDict()
with arim.helpers.timeit('Computation of ray weights'):
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        shape = (frame.probe.numelements, grid.numpoints)
        if conf['model.use_directivity']:
            directivity = arim.model.directivity_finite_width_2d_for_path(
                ray_geometry, frame.probe.dimensions.x[0], wavelength_in_couplant)
        else:
            directivity = 1.
        if conf['model.use_transrefl']:
            transrefl = arim.model.transmission_reflection_for_path(path, ray_geometry,
                                                                    force_complex=True)
        else:
            transrefl = 1.
        if conf['model.use_beamspread']:
            beamspread = arim.model.beamspread_for_path(ray_geometry)
        else:
            beamspread = 1.
        # use resize for the case where directivity, transrefl and beamspread are all 1.
        ray_weights_dict[pathname] = np.resize(directivity * transrefl * beamspread,
                                               shape)

with arim.helpers.timeit('Computation of sensitivity'):
    sensitivity_path_dict = OrderedDict((k, arim.model.sensitivity_conjugate_for_path(v))
                                        for k, v in ray_weights_dict.items())
    sensitivity_view_dict = OrderedDict()
    for viewname, view in views.items():
        sensitivity_view_dict[viewname] = arim.model.sensitivity_conjugate_for_view(
            sensitivity_path_dict[view.tx_path.name],
            sensitivity_path_dict[view.rx_path.name],
        )

# %% Plot sensitivity

if conf['plot.sensitivity_view']:
    ref_db = max(np.nanmax(np.abs(v)) for v in sensitivity_view_dict.values())
    aplt.plot_oxz_many(sensitivity_view_dict.values(), grid, 7, 3,
                       title_list=sensitivity_view_dict.keys(),
                       suptitle='Sensitivity', scale='db', ref_db=ref_db,
                       filename='sensitivity', clim=[-60, 0])

# %% Setups TFM
frame.apply_filter(
    arim.signal.Hilbert() + arim.signal.ButterworthBandpass(5, 3e6, 5.5e6, frame.time))

tfms = []
for viewname, view in views.items():
    amps_tx = arim.im.SensitivityConjugateAmplitudes(
        frame, grid, ray_weights_dict[view.tx_path.name],
        sensitivity_path_dict[view.tx_path.name],
        divide_by_sensitivity=conf['tfm.divide_by_sensitivity']
    )
    amps_rx = arim.im.SensitivityConjugateAmplitudes(
        frame, grid, ray_weights_dict[view.rx_path.name],
        sensitivity_path_dict[view.rx_path.name],
        divide_by_sensitivity=conf['tfm.divide_by_sensitivity']
    )

    tfm = arim.im.SingleViewTFM(frame, grid, view,
                                amplitudes_tx=amps_tx, amplitudes_rx=amps_rx)
    tfms.append(tfm)

# %% Run all TFM

with arim.helpers.timeit('Delay-and-sum', logger=logger):
    for tfm in tfms:
        tfm.run(fillvalue=conf['tfm.fillvalue'])

# %% Plot all TFM

ref_db = 1.  # fallback is no TFM is plotted

if conf['plot.tfm']:
    # func_res = lambda x: np.imag(x)
    scale = aplt.common_dynamic_db_scale([tfm.res for tfm in tfms], reference_area)

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
