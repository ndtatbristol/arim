"""
TFM sensitivity in immersion inspection using a single-frequency LTI model.

The sensitivity on a point is defined as the TFM intensity that a defect centered
on this point would have.

Input
-----
conf.yaml, conf.d/*.yaml
    Configuration files

Output
------
sensitivity_images.pickle
    Sensitivity images in binary format

sensitivity.png
    Sensitivity images in image format

"""

import logging
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import arim
import arim.im
import arim.io
import arim.models.block_in_immersion as bim
import arim.plot as aplt
import arim.ray
import arim.scat

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_sensitivity(dataset_name, save):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]

    probe = arim.io.probe_from_conf(conf)
    examination_object = arim.io.block_in_immersion_from_conf(conf)
    tx, rx = arim.ut.fmc(probe.numelements)
    numtimetraces = len(tx)

    model_options = dict(
        frequency=probe.frequency, probe_element_width=probe.dimensions.x[0]
    )

    tic = time.time()
    grid = arim.io.grid_from_conf(conf)
    grid_p = grid.to_oriented_points()
    logger.info(f"grid numpoints: {grid.numpoints}")
    probe_p = probe.to_oriented_points()
    views = bim.make_views(
        examination_object,
        probe_p,
        grid_p,
        max_number_of_reflection=1,
        tfm_unique_only=True,
    )

    aplt.plot_interfaces(
        [probe_p, *examination_object.walls, grid_p],
        show_orientations=False,
        show_last=True,
        # markers=[".", "-", "-"],
        filename=str(result_dir / "interfaces"),
        savefig=save,
    )

    arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)

    scat_obj = arim.scat.scat_factory(
        **conf["scatterer"]["specs"], material=examination_object.block_material
    )
    scat_angle = np.deg2rad(conf["scatterer"]["angle_deg"])
    with arim.helpers.timeit("Scattering matrices", logger=logger):
        scat_mat = scat_obj.as_single_freq_matrices(
            model_options["frequency"], 180
        )  # use precomputation

    with arim.helpers.timeit("Computation of ray weights for all paths", logger=logger):
        ray_weights = bim.ray_weights_for_views(views, **model_options)

    sensitivity_images_dict = dict()
    timetrace_weights = np.ones(numtimetraces)

    for viewname, view in views.items():
        model_coefficients = arim.model.model_amplitudes_factory(
            tx.astype(int),
            rx.astype(int),
            view,
            ray_weights,
            scat_mat,
            scat_angle=scat_angle,
        )

        sensitivity_images_dict[viewname] = model_coefficients.sensitivity_uniform_tfm(
            timetrace_weights
        )

    toc = time.time()
    elapsed = toc - tic
    logger.info(
        f"Total time for sensitivity images: {elapsed:.2f} s ({grid.numpoints} points)"
    )

    # %%
    out = {"images": sensitivity_images_dict, "grid": grid}
    with open(result_dir / "sensitivity_images.pickle", "wb") as f:
        pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    return sensitivity_images_dict


def plot_sensitivity(dataset_name, save):
    conf = arim.io.load_conf(dataset_name)
    result_dir = conf["result_dir"]
    with open(result_dir / "sensitivity_images.pickle", "rb") as f:
        loaded = pickle.load(f)

    grid = loaded["grid"]

    ncols = 3
    nrows = 7
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(7.4, 8), sharex=False, sharey=False
    )

    xmin = grid.xmin
    xmax = grid.xmax
    zmin = conf["frontwall"]["z"]
    zmax = conf["backwall"]["z"]

    ref_db = max(np.nanmax(np.abs(data)) for data in loaded["images"].values())

    for (viewname, data), ax in zip(loaded["images"].items(), axes.ravel()):
        clim = [-40, 0.0]
        ax, im = aplt.plot_oxz(
            data,
            grid,
            ax=ax,
            scale="db",
            ref_db=ref_db,
            clim=clim,
            interpolation="none",
            savefig=False,
            draw_cbar=False,
        )
        ax.set_title(viewname, y=0.9, size="small")
        ax.set_adjustable("box")
        ax.axis([xmin, xmax, zmax, zmin])

        if ax in axes[-1, :]:
            ax.set_xlabel("x (mm)")
            ax.set_xticks([xmin, xmax, np.round((xmin + xmax) / 2, decimals=3)])
        else:
            ax.set_xlabel("")
            ax.set_xticks([])
        if ax in axes[:, 0]:
            ax.set_ylabel("z (mm)")
            ax.set_yticks([zmax, zmin, np.round((zmin + zmax) / 2, decimals=3)])
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), location="top", fraction=0.05, aspect=40, pad=0.03
    )
    cbar.ax.set_ylabel("dB")

    if save:
        fig.savefig(str(result_dir / "sensitivity"))


if __name__ == "__main__":
    dataset_name = "."  # current directory
    save = False
    compute_sensitivity(dataset_name, save)
    plot_sensitivity(dataset_name, save)
