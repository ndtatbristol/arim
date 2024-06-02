"""
Multi-view TFM for a contact inspection

If the configuration entry "reference_rect" is defined, the dB scale for each
view will be adjusted to the maximum intensity in the area.

A backwall must be defined for imaging with skip paths.
A frontwall and a backwall must be defined for imaging with double-skip paths.


Input
-----
conf.yaml
    Configuration file

Output
------
TFM images

"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import arim
import arim.im
import arim.io
import arim.models.block_in_contact as bic
import arim.plot as aplt
import arim.ray
import arim.signal

# %% Load configuration

conf = arim.io.load_conf(".")
save = True
aplt.conf["savefig"] = save
result_dir = conf["result_dir"]

# %% Load frame
frame = arim.io.frame_from_conf(conf, use_probe_from_conf=True)
frame = frame.apply_filter(
    arim.signal.Hilbert()
    + arim.signal.ButterworthBandpass(**conf["filter_for_tfm"], time=frame.time)
)
frame = frame.expand_frame_assuming_reciprocity()
probe_p = frame.probe.to_oriented_points()

# %% Plot interfaces
aplt.plot_interfaces(
    [probe_p, *frame.examination_object.walls],
    show_orientations=False,
    show_last=True,
    markers=[".", "-", "-"],
    filename=str(result_dir / "interfaces"),
    savefig=save,
)

# %% Ray tracing
grid = arim.io.grid_from_conf(conf)
grid_p = grid.to_oriented_points()

views = bic.make_views(
    frame.examination_object,
    probe_p,
    grid_p,
    tfm_unique_only=True,
    max_number_of_reflection=2,
)
views_to_use = conf.get("views_to_use", "all")
if views_to_use != "all":
    views = {
        viewname: view for viewname, view in views.items() if viewname in views_to_use
    }
arim.ray.ray_tracing(views.values())

# %% Compute TFM
tfms = dict()
for viewname, view in views.items():
    with arim.helpers.timeit(f"TFM {view.name}"):
        tfms[viewname] = arim.im.tfm.tfm_for_view(
            frame, grid, view, fillvalue=0.0, interpolation="nearest"
        )

# %% Plot TFM
reference_rect = conf["reference_rect"]
if reference_rect is None:
    reference_area = None
else:
    reference_area = grid.points_in_rectbox(**reference_rect)


# dynamic dB scale:
scale = aplt.common_dynamic_db_scale(
    [tfm.res for tfm in tfms.values()], reference_area, db_range=40.0
)

for i, (viewname, tfm) in enumerate(tfms.items()):
    assert tfm.grid is grid

    ref_db, clim = next(scale)

    if reference_rect is None:
        patches = None
    else:
        patches = [
            mpl.patches.Rectangle(
                (reference_rect["xmin"], reference_rect["zmin"]),
                reference_rect["xmax"] - reference_rect["xmin"],
                reference_rect["zmax"] - reference_rect["zmin"],
                fill=False,
                edgecolor="magenta",
            )
        ]

    ax, _ = aplt.plot_tfm(
        tfm,
        clim=clim,
        scale="db",
        ref_db=ref_db,
        title=f"TFM {viewname}",
        savefig=False,
        patches=patches,
        draw_cbar=True,
        interpolation="none",
    )
    ax.set_adjustable("box")
    ax.axis([grid.xmin, grid.xmax, grid.zmax, 0])
    if save:
        ax.figure.savefig(str(result_dir / f"tfm_{i:02}_{viewname}"))

# Block script until windows are closed.
plt.show()
