"""
Plotting utilities based on `matplotib <http://matplotlib.org/>`_.

Some default values are configurable via the dictionary ``arim.plot.conf``.

.. py:data:: conf

    Dictionary of default values. For some functions, if an argument is not populated,
    its values will be populated from this dictionary. Example::

        # save the figure (independently on conf['savefig])
        plot_oyz(data, grid, savefig=True, filename='foo')

        # do not save the figure independently on conf['savefig])
        plot_oyz(data, grid, savefig=False, filename='foo')

        # save the figure depending only if conf['savefig'] is True
        plot_oyz(data, grid, filename='foo')

.. py:data:: micro_formatter
.. py:data:: milli_formatter
.. py:data:: mega_formatter

    Format the labels of an axis in a given unit prefix. Usage::

        import matplotlib.pyplot as plt
        ax = plt.plot(distance_vector, data)
        ax.xaxis.set_major_formatter(arim.plot.milli_formatter)

"""

from warnings import warn
import logging

from matplotlib import ticker
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from . import ut
from .exceptions import ArimWarning
from . import geometry as g
from .config import Config

__all__ = [
    "micro_formatter",
    "mega_formatter",
    "milli_formatter",
    "plot_bscan",
    "plot_bscan_pulse_echo",
    "plot_oxz",
    "plot_oxz_many",
    "plot_tfm",
    "plot_directivity_finite_width_2d",
    "draw_rays_on_click",
    "RayPlotter",
    "conf",
    "common_dynamic_db_scale",
]

logger = logging.getLogger(__name__)

micro_formatter = ticker.FuncFormatter(lambda x, pos: "{:.1f}".format(x * 1e6))
micro_formatter.__doc__ = "Format an axis to micro (µ).\nExample: ``ax.xaxis.set_major_formatter(micro_formatter)``"

milli_formatter = ticker.FuncFormatter(lambda x, pos: "{:.1f}".format(x * 1e3))
milli_formatter.__doc__ = "Format an axis to milli (m).\nExample: ``ax.xaxis.set_major_formatter(milli_formatter)``"

mega_formatter = ticker.FuncFormatter(lambda x, pos: "{:.1f}".format(x * 1e-6))
mega_formatter.__doc__ = "Format an axis to mega (M).\nExample: ``ax.xaxis.set_major_formatter(mega_formatter)``"

conf = Config(
    [
        ("savefig", False),  # save the figure?
        ("plot_oxz.figsize", None),
        ("plot_oxz_many.figsize", None),
    ]
)


def plot_bscan(
    frame,
    scanlines_idx,
    use_dB=True,
    ax=None,
    title="B-scan",
    clim=None,
    interpolation="none",
    draw_cbar=True,
    cmap=None,
    savefig=None,
    filename="bscan",
):
    """Plot Bscan (scanlines vs time)
    
    Parameters
    ----------
    frame : Frame
    scanlines_idx : slice or tuple or ndarray
        Scanlines to use. Any valid numpy array is accepted.
    use_dB : bool, optional
    ax : matplotlib axis, optional
        Where to draw. Default: create a new figure and axis.
    title : str, optional
        Title of the image (default: "Bscan")
    clim : tuple, optional
        Color limits of the image.
    interpolation : str, optional
        Image interpolation type (default: "none")
    draw_cbar : bool, optional
    cmap : str, optional
    savefig : bool, optional
        Default: use ``conf["savefig"]``
    filename : str, optional
        Default: "bscan"
    
    Returns
    -------
    ax : matplotlib axis
    im : matplotlib image

    Examples
    --------

    >>> arim.plot.plot_bscan(frame, frame.tx == 0)

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if savefig is None:
        savefig = conf["savefig"]

    scanlines = frame.scanlines[scanlines_idx]
    numscanlines = scanlines.shape[0]
    if use_dB:
        scanlines = ut.decibel(scanlines)
        if clim is None:
            clim = [-40.0, 0.0]

    im = ax.imshow(
        scanlines,
        extent=[frame.time.start, frame.time.end, 0, numscanlines - 1],
        interpolation=interpolation,
        cmap=cmap,
        origin="lower",
    )
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("TX/RX index")
    ax.xaxis.set_major_formatter(micro_formatter)
    ax.xaxis.set_minor_formatter(micro_formatter)

    # Use element index instead of scanline index (may be different)
    tx = frame.tx[scanlines_idx]
    rx = frame.rx[scanlines_idx]

    def _y_formatter(i, pos):
        i = int(i)
        try:
            return f"({tx[i]}, {rx[i]})"
        except IndexError:
            return ""

    y_formatter = ticker.FuncFormatter(_y_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.set_minor_formatter(y_formatter)

    if draw_cbar:
        fig.colorbar(im, ax=ax)
    if clim is not None:
        im.set_clim(clim)
    if title is not None:
        ax.set_title(title)

    ax.axis("tight")
    if savefig:
        ax.figure.savefig(filename)
    return ax, im


def plot_bscan_pulse_echo(
    frame,
    use_dB=True,
    ax=None,
    title="B-scan (pulse-echo)",
    clim=None,
    interpolation="none",
    draw_cbar=True,
    cmap=None,
    savefig=None,
    filename="bscan",
):
    """
    Plot a B-scan. Use the pulse-echo scanlines.

    Parameters
    ----------
    frame
    use_dB
    ax
    title
    clim
    interpolation
    draw_cbar
    cmap

    Returns
    -------
    axis, image

    See Also
    --------
    :func:`plot_bscan`

    """
    pulse_echo = frame.tx == frame.rx
    elements = frame.tx[pulse_echo]
    ax, im = plot_bscan(
        frame,
        pulse_echo,
        use_dB=use_dB,
        ax=ax,
        title=title,
        clim=clim,
        interpolation=interpolation,
        draw_cbar=draw_cbar,
        cmap=cmap,
        savefig=False,  # save later
        filename=filename,
    )
    ax.set_ylabel("Element")
    # Use element index instead of scanline index (may be different)
    def _y_formatter(i, pos):
        i = int(i)
        if i >= len(elements):
            return ""
        else:
            return str(elements[i])

    y_formatter = ticker.FuncFormatter(_y_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.set_minor_formatter(y_formatter)

    if savefig:
        ax.figure.savefig(filename)

    return ax, im


def plot_psd(
    frame,
    idx="all",
    to_show="filtered",
    welch_params=None,
    ax=None,
    title="Power spectrum estimation",
    show_legend=True,
    savefig=None,
    filename="psd",
):
    """
    Plot the estimated power spectrum of a scanline using Welch's method.

    Parameters
    ----------
    frame : Frame
    idx : int or slice or list
        Index or indices of the scanline to use. If multiple indices are given,
        the arithmetical mean of all PSDs is plotted. Default: use all
    to_show
    welch_params : dict
        Arguments to pass to ``scipy.signal.welch``.
    ax : matplotlib.axes.Axes or None
    title
    show_legend
    savefig
    filename

    Returns
    -------
    ax : matplotlib.axes.Axes
    lines : dict

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if welch_params is None:
        welch_params = {}

    if savefig is None:
        savefig = conf["savefig"]

    if isinstance(idx, str) and idx == "all":
        idx = slice(None)

    fs = 1 / frame.time.step

    to_show = to_show.lower()
    if to_show == "both":
        show_raw = True
        show_filtered = True
    elif to_show == "raw":
        show_raw = True
        show_filtered = False
    elif to_show == "filtered":
        show_raw = False
        show_filtered = True
    else:
        raise ValueError("Valid values for 'to_show' are: filtered, raw, both")

    lines = {}

    if show_raw:
        x = frame.scanlines_raw[idx].real
        freq, pxx = scipy.signal.welch(x, fs, **welch_params)
        if pxx.ndim == 2:
            pxx = np.mean(pxx, axis=0)
        line = ax.plot(freq, pxx, label="raw".format(idx=idx))
        lines["raw"] = line
    if show_filtered:
        x = frame.scanlines[idx].real
        freq, pxx = scipy.signal.welch(x, fs, **welch_params)
        if pxx.ndim == 2:
            pxx = np.mean(pxx, axis=0)
        line = ax.plot(freq, pxx, label="filtered".format(idx=idx))
        lines["filtered"] = line
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("power spectrum estimation")
    ax.xaxis.set_major_formatter(mega_formatter)
    ax.xaxis.set_minor_formatter(mega_formatter)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        ax.legend(loc="best")

    if savefig:
        fig.savefig(filename)
    return ax, lines


def plot_oxz(
    data,
    grid,
    ax=None,
    title=None,
    clim=None,
    interpolation="none",
    draw_cbar=True,
    cmap=None,
    figsize=None,
    savefig=None,
    patches=None,
    filename=None,
    scale="linear",
    ref_db=None,
):
    """
    Plot data in the plane Oxz.

    Parameters
    ----------
    data : ndarray
        Shape: 2D matrix ``(grid.numx, grid.numz)`` or 3D matrix ``(grid.numx, 1, grid.numz)``
        or 1D matrix ``(grid.numx * grid.numz)``
    grid : Grid
    ax : matplotlib.Axis or None
        Axis where to plot.
    title : str or None
    clim : List[Float] or None
    interpolation : str or None
    draw_cbar : boolean
    cmap
    figsize : List[Float] or None
        Default: ``conf['plot_oxz.figsize']``
    savefig : boolean or None
        If True, save the figure. Default: ``conf['savefig']``
    patches : List[matplotlib.patches.Patch] or None
        Patches to draw
    filename : str or None
        If True
    scale : str or None
        'linear' or 'db'. Default: 'linear'
    ref_db : float or None
        Value for 0 dB. Used only for scale=db.

    Returns
    -------
    axis
    image

    Examples
    --------
    ::

        grid = arim.geometry.Grid(-5e-3, 5e-3, 0, 0, 0, 15e-3, .1e-3)
        k = 2 * np.pi / 10e-3
        data = (np.cos(grid.x * 2 * k) * np.sin(grid.z * k))
        ax, im = aplt.plot_oxz(data, grid)


    """
    if figsize is None:
        figsize = conf["plot_oxz.figsize"]
    else:
        if ax is not None:
            warn(
                "figsize is ignored because an axis is provided",
                ArimWarning,
                stacklevel=2,
            )
    if savefig is None:
        savefig = conf["savefig"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if patches is None:
        patches = []

    valid_shapes = [
        (grid.numx, 1, grid.numz),
        (grid.numx, grid.numz),
        (grid.numx * grid.numz,),
    ]
    if data.shape in valid_shapes:
        data = data.reshape((grid.numx, grid.numz))
    else:
        msg = "invalid data shape (got {}, expected {} or {} or {})".format(
            data.shape, *valid_shapes
        )
        raise ValueError(msg)

    data = np.rot90(data)

    scale = scale.lower()
    if scale == "linear":
        if ref_db is not None:
            warn("ref_db is ignored for linear plot", ArimWarning, stacklevel=2)
    elif scale == "db":
        data = ut.decibel(data, ref_db)
    else:
        raise ValueError("invalid scale: {}".format(scale))

    image = ax.imshow(
        data,
        interpolation=interpolation,
        origin="lower",
        extent=(grid.xmin, grid.xmax, grid.zmax, grid.zmin),
        cmap=cmap,
    )
    if ax.get_xlabel() == "":
        # avoid overwriting labels
        ax.set_xlabel("x (mm)")
    if ax.get_ylabel() == "":
        ax.set_ylabel("z (mm)")
    ax.xaxis.set_major_formatter(milli_formatter)
    ax.xaxis.set_minor_formatter(milli_formatter)
    ax.yaxis.set_major_formatter(milli_formatter)
    ax.yaxis.set_minor_formatter(milli_formatter)
    if draw_cbar:
        # necessary magic to make the colorbar the same height as the image
        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.set_aspect(aspect=20, adjustable="box")
        fig.colorbar(image, ax=ax, cax=cax)
    if clim is not None:
        image.set_clim(clim)
    if title is not None:
        ax.set_title(title)
    for p in patches:
        ax.add_patch(p)

    # Like axis('equal') but mitigates https://github.com/matplotlib/matplotlib/issues/11416
    # adjustable=box to avoid white space (default in matplotlib 3)
    ax.axis(aspect=1, adjustable="box")
    ax.axis([grid.xmin, grid.xmax, grid.zmax, grid.zmin])
    if savefig:
        if filename is None:
            raise ValueError("filename must be provided when savefig is true")
        fig.savefig(filename)
    return ax, image


def plot_oxz_many(
    data_list,
    grid,
    nrows,
    ncols,
    title_list=None,
    suptitle=None,
    draw_colorbar=True,
    figsize=None,
    savefig=None,
    clim=None,
    filename=None,
    y_title=1.0,
    y_suptitle=1.0,
    **plot_oxz_kwargs,
):
    """
    Plot many Oxz plots on the same figure.

    Parameters
    ----------
    data_list : List[ndarray]
        Data are plotted from top left to bottom right, row per row.
    grid : Grid
    nrows : int
    ncols : int
    title_list : List[str] or None
    suptitle : str or None
    draw_colorbar : boolean
        Default: True
    figsize : List[Float] or None
        Default: ``conf['plot_oxz_many.figsize']``
    savefig: boolean
        Default: ``conf['savefig']``
    clim :
        Color limit. Common for all plots.
    filename
    y_title : float
        Adjust y location of the titles.
    y_suptitle : float
        Adjust y location of the titles.

    plot_oxz_kwargs

    Returns
    -------
    ax_list
    im_list

    """
    if savefig is None:
        savefig = conf["savefig"]
    if figsize is None:
        figsize = conf["plot_oxz_many.figsize"]

    if title_list is None:
        title_list = [None] * len(data_list)

    # must use a common clim (otherwise the figure does not make sense)
    if clim is None:
        clim = (
            min(np.nanmin(x) for x in data_list),
            max(np.nanmax(x) for x in data_list),
        )

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize
    )
    images = []
    for data, title, ax in zip(data_list, title_list, axes.ravel()):
        # the current function handles saving fig, drawing the cbar and displaying the title
        # so we prevent plot_oxz to do it.
        ax, im = plot_oxz(
            data,
            grid,
            ax=ax,
            clim=clim,
            draw_cbar=False,
            savefig=False,
            **plot_oxz_kwargs,
            title=None,
        )
        images.append(im)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if title is not None:
            ax.set_title(title, y=y_title)
    for ax in axes[-1, :]:
        ax.set_xlabel("x (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z (mm)")
    if suptitle is not None:
        fig.suptitle(suptitle, y=y_suptitle, size="x-large")
    if draw_colorbar:
        fig.colorbar(im, ax=axes.ravel().tolist())
    if savefig:
        if filename is None:
            raise ValueError("filename must be provided when savefig is true")
        fig.savefig(filename)
    return axes, images


def plot_tfm(tfm, y=0.0, func_res=None, interpolation="bilinear", **plot_oxz_kwargs):
    """
    Plot a TFM in plane Oxz.

    Parameters
    ----------
    tfm : BaseTFM
    y : float
    interpolation : str
        Cf matplotlib.pyplot.imshow
    func_res : function
        Function to apply on tfm.res before plotting it. Example: ``lambda x: np.abs(x)``
    plot_oxz_kwargs : dict

    Returns
    -------
    ax
    image

    See Also
    --------
    :func:`plot_oxz`

    """
    grid = tfm.grid
    iy = np.argmin(np.abs(grid.y - y))

    if tfm.res is None:
        raise ValueError("No result in this TFM object.")
    if func_res is None:
        func_res = lambda x: x

    data = func_res(tfm.res[:, iy, :])

    return plot_oxz(data, grid=grid, interpolation=interpolation, **plot_oxz_kwargs)


def plot_directivity_finite_width_2d(element_width, wavelength, ax=None, **kwargs):
    """

    Parameters
    ----------
    element_width
    wavelength
    ax : matplotlib.axes._subplots.AxesSubplot
    kwargs

    Returns
    -------

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    title = kwargs.get(
        "title", "Directivity of an element (uniform sources along a straight line)"
    )

    ratio = element_width / wavelength
    theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
    directivity = ut.directivity_finite_width_2d(theta, element_width, wavelength)

    ax.plot(
        np.rad2deg(theta),
        directivity,
        label=r"$a/\lambda = {:.2f}$".format(ratio),
        **kwargs,
    )
    ax.set_xlabel(r"Polar angle $\theta$ (deg)")
    ax.set_ylabel("directivity (1)")
    ax.set_title(title)
    ax.set_xlim([-90, 90])
    ax.set_ylim([0, 1.2])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(15.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.legend()
    return ax


class RayPlotter:
    def __init__(
        self, grid, ray, element_index, linestyle="m--", tolerance_distance=1e-3
    ):
        self.grid = grid
        self.ray = ray
        self.element_index = element_index
        self.linestyle = linestyle
        self._lines = []
        self.debug = False
        self.y = 0
        self.tolerance_distance = tolerance_distance

    def __call__(self, event):
        logger.debug(
            "button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
            % (event.button, event.x, event.y, event.xdata, event.ydata)
        )
        ax = event.canvas.figure.axes[0]
        if event.button == 1:
            self.draw_ray(ax, event.xdata, event.ydata)
        elif event.button == 3:
            self.clear_rays(ax)
        if self.debug:
            print("show_ray_on_clic() finish with no error")

    def draw_ray(self, ax, x, z):
        gridpoints = self.grid.to_1d_points()
        wanted_point = (x, self.y, z)
        point_index = gridpoints.closest_point(*wanted_point)
        obtained_point = gridpoints[point_index]

        distance = g.norm2(*(obtained_point - wanted_point))
        if distance > self.tolerance_distance:
            logger.warning(
                "The closest grid point is far from what you want (dist: {:.2f} mm)".format(
                    distance * 1000
                )
            )

        legs = self.ray.get_coordinates_one(self.element_index, point_index)
        line = ax.plot(legs.x, legs.z, self.linestyle)
        self._lines.extend(line)
        logger.debug("Draw a ray")
        ax.figure.canvas.draw_idle()

    def clear_rays(self, ax):
        """Clear all rays"""
        lines_to_clear = [line for line in ax.lines if line in self._lines]
        for line in lines_to_clear:
            ax.lines.remove(line)
            self._lines.remove(line)
        logger.debug("Clear {} ray(s) on figure".format(len(lines_to_clear)))
        ax.figure.canvas.draw_idle()

    def connect(self, ax):
        """Connect to matplotlib event backend"""
        ax.figure.canvas.mpl_connect("button_press_event", self)


def draw_rays_on_click(grid, ray, element_index, ax=None, linestyle="m--"):
    """
    Dynamic plotting of rays on a plot.

    Left-click: draw a ray between the probe element and the mouse point.
    Right-click: clear all rays in the plot.

    Parameters
    ----------
    grid : Grid
    ray : Rays

    element_index : int
    ax : Axis
        Matplotlib axis on which to plot. If None: current axis.
    linestyle : str
        A valid matplotlib linestyle. Default: 'm--'

    Returns
    -------
    ray_plotter : RayPlotter

    """
    if ax is None:
        ax = plt.gca()
    ray_plotter = RayPlotter(
        grid=grid, ray=ray, element_index=element_index, linestyle=linestyle
    )
    ray_plotter.connect(ax)
    return ray_plotter


def plot_interfaces(
    oriented_points_list,
    ax=None,
    show_probe=True,
    show_last=True,
    show_orientations=False,
    n_arrows=10,
    title="Interfaces",
    savefig=None,
    filename="interfaces",
    markers=None,
    show_legend=True,
    quiver_kwargs=None,
):
    """
    Plot interfaces on the Oxz plane.

    Assume the first interface is for the probe and the last is for the grid.

    Parameters
    ----------
    oriented_points_list : list[OrientedPoints]
    ax : matplotlib.axis.Axis
    show_probe : boolean
        Default True
    show_last : boolean
        Default: True. Useful for hiding the grid.
    show_orientations : boolean
        Plot arrows for the orientations. Default: False
    n_arrows : int
        Number of arrows per interface to plot.
    title : str or None
        Title to display. None for no title.
    savefig : boolean
        If True, the plot will be saved. Default: ``conf['savefig']``.
    filename : str
        Filename of the plot, used if savefig is True. Default: 'interfaces'
    markers : List[str]
        Matplotlib markers for each interfaces. Default: '.' for probe, ',k' for the grid,
        '.' for the rest.
    show_legend : boolean
        Default True
    quiver_kwargs : dict
        Arguments for displaying the arrows (cf. matplotlib function 'quiver')

    Returns
    -------
    ax : matplotlib.axis.Axis

    """
    if savefig is None:
        savefig = conf["savefig"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if quiver_kwargs is None:
        quiver_kwargs = dict(width=0.0003)

    numinterfaces = len(oriented_points_list)

    if markers is None:
        markers = ["."] + ["."] * (numinterfaces - 2) + [",k"]

    for i, (interface, marker) in enumerate(zip(oriented_points_list, markers)):
        if i == 0 and not show_probe:
            continue
        if i == numinterfaces - 1 and not show_last:
            continue
        line, = ax.plot(
            interface.points.x, interface.points.z, marker, label=interface.points.name
        )

        if show_orientations:
            # arrow every k points
            k = len(interface.points) // n_arrows
            if k == 0:
                k = 1
            # import pytest; pytest.set_trace()
            ax.quiver(
                interface.points.x[::k],
                interface.points.z[::k],
                interface.orientations.x[::k, 2],
                interface.orientations.z[::k, 2],
                color=line.get_color(),
                units="xy",
                angles="xy",
                **quiver_kwargs,
            )

    # set labels only if there is none in the axis yet
    if ax.get_xlabel() == "":
        ax.set_xlabel("x (mm)")
    if ax.get_ylabel() == "":
        ax.set_ylabel("z (mm)")

    ax.xaxis.set_major_formatter(milli_formatter)
    ax.yaxis.set_major_formatter(milli_formatter)
    ax.xaxis.set_minor_formatter(milli_formatter)
    ax.yaxis.set_minor_formatter(milli_formatter)

    if title is not None:
        ax.set_title(title)

    ylim = ax.get_ylim()
    if ylim[0] < ylim[1]:
        ax.invert_yaxis()

    if show_legend:
        ax.legend(loc="best")

    ax.axis("equal")

    if savefig:
        fig.savefig(filename)
    return ax


def common_dynamic_db_scale(data_list, area=None, db_range=40.0, ref_db=None):
    """
    Scale such as:
      - 0 dB corresponds to the maximum value in the area for all data arrays,
      - the clim for each data array are bound by the maximum value in the area.

    Parameters
    ----------
    data_list
    db_range : float

    Yields
    ------
    ref_db
    (clim_min, clim_max)

    Examples
    --------

        >>> area = grid.points_in_rectbox(xmin=10, xmax=20)
        >>> common_db_scale_iter = common_dynamic_db_scale(data_list, area)
        >>> for data in data_list:
        ...     ref_db, clim = next(common_db_scale_iter)
        ...     plot_oxz(data, grid, scale='db', ref_db=ref_db, clim=clim)

    """
    data_max_list = []

    if area is None:
        area = slice(None)

    for data in data_list:
        data_max_list.append(np.nanmax(np.abs(data[area])))
    if ref_db is None:
        ref_db = max(data_max_list)

    data_max_db_list = ut.decibel(data_max_list, ref_db)

    for data_max_db in data_max_db_list:
        yield ref_db, (data_max_db - db_range, data_max_db)
