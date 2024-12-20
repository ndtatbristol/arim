"""
Forward model for a contact inspection

Model was not validated experimentally, buyer beware.
"""


import logging
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from scipy.signal import hilbert

import arim
import arim.im  # for imaging
import arim.io
import arim.model
import arim.models.block_in_contact as bic
import arim.plot as aplt
import arim.scat
import arim.signal

save = False
aplt.conf["savefig"] = False

use_multifreq = True
max_number_of_reflection = 1  # for scatterer
tfm_unique_only = False
numangles_for_scat_precomp = 120  # 0 to disable precomputation

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("arim").setLevel(logging.INFO)

conf = arim.io.load_conf(".")
imaging_walls = ["Backwall"]

# %% Define inspection set-up
probe = arim.io.probe_from_conf(conf)
tx_list, rx_list = arim.ut.fmc(probe.numelements)
numtimetraces = len(tx_list)

examination_object = arim.io.block_in_contact_from_conf(conf)

defect_centre = conf["scatterer"]["location"]
scatterer = arim.geometry.default_oriented_points(
    arim.geometry.Points(
        [
            [
                conf["scatterer"]["location"]["x"],
                conf["scatterer"]["location"]["y"],
                conf["scatterer"]["location"]["z"],
            ]
        ],
        name="Scatterer",
    )
)

grid = arim.geometry.Grid(**conf["grid"], ymin=0.0, ymax=0.0)
grid_p = grid.to_oriented_points()

aplt.plot_interfaces(
    [probe.to_oriented_points(), *examination_object.walls.values(), scatterer, grid_p],
    show_last=False,
    markers=[".", "-", "d", ".k"],
)

# %% Ray tracing for scatterer
views = bic.make_views(
    examination_object,
    probe.to_oriented_points(),
    scatterer,
    walls_for_imaging=imaging_walls,
    tfm_unique_only=tfm_unique_only,
)
# views = {viewname: view for viewname, view in views.items() if viewname in {"L-T", "T-L"}}  # debug
print("Views: " + ", ".join(views.keys()))
arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

# %% Ray tracing for backwall echoes
backwall_paths = bic.backwall_paths(
    examination_object.block_material,
    probe.to_oriented_points(),
    examination_object.walls["Backwall"],
    examination_object.under_material,
)
# backwall_paths = {pathname: path for pathname, path in backwall_paths.items() if pathname in {"LL"}} # debug
wall_paths = {f"Backwall {key}": path for key, path in backwall_paths.items()}
arim.ray.ray_tracing_for_paths(wall_paths.values())
print("Wall paths: " + ", ".join(wall_paths.keys()))
# %% Toneburst and time vector
max_delay_scat = max(
    view.tx_path.rays.times.max() + view.rx_path.rays.times.max()
    for view in views.values()
)
max_delay_backwall = max(path.rays.times.max() for path in backwall_paths.values())
max_delay = max(max_delay_scat, max_delay_backwall)


dt = 0.25 / probe.frequency  # to adjust so that the whole toneburst is sampled
_tmax = max_delay + 4 * conf["toneburst"]["num_cycles"] / probe.frequency

numsamples = scipy.fftpack.next_fast_len(math.ceil(_tmax / dt))
time = arim.Time(0.0, dt, numsamples)
freq_array = np.fft.rfftfreq(len(time), dt)
numfreq = len(freq_array)

toneburst = arim.model.make_toneburst(
    conf["toneburst"]["num_cycles"],
    conf["probe"]["frequency"],
    dt,
    numsamples,
    wrap=True,
)
toneburst_f = np.fft.rfft(toneburst)

toneburst_ref = np.abs(hilbert(toneburst)[0])

# plot toneburst
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(1e6 * time.samples, toneburst)
plt.title("toneburst (time domain)")
plt.xlabel("time (µs)")

plt.subplot(1, 2, 2)
plt.plot(1e-6 * np.fft.rfftfreq(len(toneburst), dt), abs(toneburst_f))
plt.title("toneburst (frequency domain)")
plt.xlabel("frequency (MHz)")
if aplt.conf["savefig"]:
    plt.savefig("toneburst")


# %% Compute transfer functions (init)
model_options = dict(
    probe_element_width=probe.dimensions.x[0],
    use_directivity=True,
    use_beamspread=True,
    use_transrefl=True,
    use_attenuation=True,
    turn_off_invalid_rays=False,
)

# %% Compute transfer functions for scatterers
scat_obj = arim.scat.scat_factory(
    material=examination_object.block_material, **conf["scatterer"]["specs"]
)
scat_angle = np.deg2rad(conf["scatterer"]["angle_deg"])

transfer_function_f = np.zeros((numtimetraces, numfreq), complex)
tfms_scat = OrderedDict()


if use_multifreq:
    # Multi frequency model
    transfer_function_iterator = bic.multifreq_scat_transfer_functions(
        views,
        tx_list,
        rx_list,
        freq_array=freq_array,
        scat_obj=scat_obj,
        scat_angle=scat_angle,
        numangles_for_scat_precomp=numangles_for_scat_precomp,
        **model_options,
    )
else:
    # Single frequency model
    transfer_function_iterator = bic.singlefreq_scat_transfer_functions(
        views,
        tx_list,
        rx_list,
        freq_array=freq_array,
        scat_obj=scat_obj,
        scat_angle=scat_angle,
        frequency=probe.frequency,
        numangles_for_scat_precomp=numangles_for_scat_precomp,
        **model_options,
    )

with arim.helpers.timeit("Main loop for scatterer"):
    for viewname, partial_transfer_func in transfer_function_iterator:
        transfer_function_f += partial_transfer_func
# At this stage, transfer_function_f contains the transfer function for scatterer for all views

# %% Compute transfer functions for walls

transfer_function_wall_f = np.zeros((numtimetraces, numfreq), complex)

if use_multifreq:
    transfer_function_iterator = bic.multifreq_wall_transfer_functions(
        wall_paths, tx_list, rx_list, freq_array, **model_options
    )
else:
    transfer_function_iterator = bic.singlefreq_wall_transfer_functions(
        wall_paths, tx_list, rx_list, probe.frequency, freq_array, **model_options
    )

with arim.helpers.timeit("Main loop for walls:"):
    for pathname, partial_transfer_func in transfer_function_iterator:
        transfer_function_wall_f += partial_transfer_func

# %% Compute the response in frequency then time domain
response_timetraces_f = (transfer_function_f + transfer_function_wall_f) * toneburst_f
# response_timetraces_f = transfer_function_f  * toneburst_f
# response_timetraces_f = transfer_function_wall_f  * toneburst_f
response_timetraces = arim.signal.rfft_to_hilbert(
    response_timetraces_f, numsamples, axis=-1
)
real_response_timetraces = np.real(response_timetraces)

frame = arim.Frame(
    response_timetraces, time, tx_list, rx_list, probe, examination_object
)

plt.figure()
idx = 31
plt.plot(
    frame.time.samples * 1e6,
    np.real(frame.timetraces[idx]),
    label=f"tx={frame.tx[idx]}, rx={frame.rx[idx]}",
)
plt.xlabel("time (µs)")
plt.title("time-domain response")
plt.legend()
if aplt.conf["savefig"]:
    plt.savefig("time_domain_response")


# %% Bscan
aplt.plot_bscan_pulse_echo(frame)
aplt.plot_bscan(frame, frame.tx == 0)


# %% Check reciprocity
tx = 1
rx = 19

idx1 = np.nonzero(np.logical_and(tx_list == tx, rx_list == rx))[0][0]
idx2 = np.nonzero(np.logical_and(tx_list == rx, rx_list == tx))[0][0]

real_response_timetraces = np.real(response_timetraces)

plt.figure()
plt.plot(
    time.samples * 1e6,
    real_response_timetraces[idx1],
    label=f"tx={tx_list[idx1]}, rx={rx_list[idx1]}",
)
plt.plot(
    time.samples * 1e6,
    real_response_timetraces[idx2],
    label=f"tx={tx_list[idx2]}, rx={rx_list[idx2]}",
)
plt.legend()
plt.xlabel("time (µs)")
plt.title("reciprocity - signals must overlap perfectly")
if aplt.conf["savefig"]:
    plt.savefig("reciprocity")
response_timetraces_1 = real_response_timetraces.reshape(
    (probe.numelements, probe.numelements, len(time))
)
response_timetraces_2 = np.swapaxes(response_timetraces_1, 0, 1)
error_reciprocity = np.max(
    np.abs(response_timetraces_1 - response_timetraces_2), axis=-1
)
logger.info(
    f"Reciprocity error: {np.max(error_reciprocity)} on timetrace {np.argmax(error_reciprocity)}"
)
for viewname, view in views.items():
    plt.text(
        (view.tx_path.rays.times[tx, 0] + view.rx_path.rays.times[rx, 0]) * 1e6,
        0.0,
        viewname,
    )
plt.legend()

# %% Full TFM
views_imaging = bic.make_views(
    examination_object,
    probe.to_oriented_points(),
    grid.to_oriented_points(),
    tfm_unique_only=True,
    walls_for_imaging=imaging_walls,
)
# views_imaging = {viewname: view for viewname, view in views_imaging.items() if viewname in ["L-L", "L-T", "T]}# debug
arim.ray.ray_tracing(views_imaging.values(), convert_to_fortran_order=True)

tfms = {}
for i, view in enumerate(views_imaging.values()):
    with arim.helpers.timeit(f"TFM {view.name}", logger=logger):
        tfms[view.name] = arim.im.tfm.tfm_for_view(
            frame, grid, view, fillvalue=0.0, interpolation=("lanczos", 3)
        )

# %%
size_box_x = 5e-3
size_box_z = 5e-3

reference_area = grid.points_in_rectbox(
    xmin=defect_centre["x"] - size_box_x / 2,
    xmax=defect_centre["x"] + size_box_x / 2,
    zmin=defect_centre["z"] - size_box_z / 2,
    zmax=defect_centre["z"] + size_box_z / 2,
)
scale = aplt.common_dynamic_db_scale(
    [tfm.res for tfm in tfms.values()], reference_area, db_range=40.0
)
# scale = aplt.common_dynamic_db_scale(
#    [tfm.res for tfm in tfms.values()], None, db_range=40.0
# )


for i, (viewname, tfm) in enumerate(tfms.items()):
    ref_db, clim = next(scale)

    ax, _ = aplt.plot_tfm(
        tfm,
        clim=clim,
        scale="db",
        ref_db=ref_db,
        title="TFM {viewname}".format(**locals()),
        filename="tfm_{i:02}_{viewname}".format(**locals()),
    )
    ax.plot(defect_centre["x"], defect_centre["z"], "ow")
