"""
Plotting utilities

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

.. py:data:: mm_formatter

    Format an axis whose values are in meter to millimeter. Usage::

        ax = plt.plot(distance_vector, data)
        ax.xaxis.set_major_formatter(mm_formatter)

.. py:data:: us_formatter

    Format an axis whose values are in second to microsecond. Usage::

        ax = plt.plot(time_vector, data)
        ax.xaxis.set_major_formatter(us_formatter)


"""

import functools
from warnings import warn
import logging
import os

import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

from . import utils, model
from .exceptions import ArimWarning
from . import geometry as g
from .config import Config

__all__ = ['mm_formatter', 'us_formatter',
           'plot_bscan_pulse_echo', 'plot_oxz', 'plot_oxz_many', 'plot_tfm', 'plot_directivity_finite_width_2d',
           'draw_rays_on_click', 'RayPlotter', 'conf']

logger = logging.getLogger(__name__)

mm_formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 1e3))
us_formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 1e6))

conf = Config([
    ('savefig', False),  # save the figure?
    ('plot_oxz.figsize', None),
    ('plot_oxz.axis_limits', None),
    ('plot_oxz.add_scale_to_title', True),
    ('plot_oxz.add_scale_to_filename', True),

    ('plot_oxz_many.figsize', None),

])


def _elements_ticker(i, pos, elements):
    i = int(i)
    if i >= len(elements):
        return ''
    else:
        return str(elements[i])


def plot_bscan_pulse_echo(frame, use_dB=True, ax=None, title='B-scan', clim=None, interpolation='none', draw_cbar=True,
                          cmap=None):
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

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    pulse_echo = frame.tx == frame.rx
    numpulseecho = sum(pulse_echo)
    elements = frame.tx[pulse_echo]

    scanlines = frame.scanlines[pulse_echo, :]
    if use_dB:
        scanlines = utils.decibel(scanlines)
        if clim is None:
            clim = [-40., 0.]

    im = ax.imshow(scanlines, extent=[frame.time.start, frame.time.end, 0, numpulseecho - 1],
                   interpolation=interpolation, cmap=cmap, origin='lower')
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Element')
    ax.xaxis.set_major_formatter(us_formatter)
    ax.xaxis.set_minor_formatter(us_formatter)

    def elements_ticker_func(i, pos):
        print('call elements_ticker: {}'.format((i, pos)))
        s = '{:d}'.format(elements[i])
        print(s)
        return s

    # elements_ticker = ticker.FuncFormatter(lambda x, pos: '{:d}'.format(elements[int(x)]))
    elements_ticker = ticker.FuncFormatter(functools.partial(_elements_ticker, elements=elements))
    ax.yaxis.set_major_formatter(elements_ticker)
    ax.yaxis.set_minor_formatter(elements_ticker)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))

    if draw_cbar:
        fig.colorbar(im, ax=ax)
    if clim is not None:
        im.set_clim(clim)
    if title is not None:
        ax.set_title(title)

    ax.axis('tight')
    return ax, im


def plot_oxz(data, grid, ax=None, title=None, clim=None, interpolation='none',
             draw_cbar=True, cmap=None, axis_limits=None, figsize=None, savefig=None,
             filename=None, scale='none', add_scale_to_title=True,
             add_scale_to_filename=True, ref_db=None,
             ):
    """
    Plot data in the plane Oxz.

    Parameters
    ----------
    data : ndarray
        Shape: 2D matrix ``(grid.numx, grid.numz)`` or 3D matrix ``(grid.numx, 1, grid.numz)``
    grid : Grid
    ax : matplotlib.Axis or None
        Axis where to plot.
    title : str or None
    clim : List[Float] or None
    interpolation : str or None
    draw_cbar : boolean
    cmap
    axis_limits
    figsize
    savefig : boolean or None
        If True, save the figure. Default: ``conf['savefig']``
    filename : str or None
        If True
    scale : str or None
        'linear', 'db' or 'none. 'none' adds no mention of the scale in the figure. Default: 'none'
    add_scale_to_title : str or None
        Add the name of the scale in the title. Default: ``conf['plot_oyz.add_scale_to_title']``
    add_scale_to_filename
        Add the name of the scale in the filename. Default: ``conf['plot_oyz.add_scale_to_filename']``
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
        data = (np.cos(grid.xx * 2 * k) * np.sin(grid.zz * k))
        ax, im = aplt.plot_oxz(data, grid)


    """
    if figsize is None:
        figsize = conf['plot_oxz.figsize']
    else:
        if ax is not None:
            warn('figsize is ignored because an axis is provided', ArimWarning)
    if axis_limits is None:
        axis_limits = conf['plot_oxz.axis_limits']
    if savefig is None:
        savefig = conf['savefig']
    if add_scale_to_title:
        add_scale_to_title = conf['plot_oxz.add_scale_to_title']
    if add_scale_to_filename:
        add_scale_to_filename = conf['plot_oxz.add_scale_to_filename']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if data.ndim == 3 and data.shape == (grid.numx, 1, grid.numz):
        data = data.reshape((grid.numx, grid.numz))
    elif data.ndim == 2 and data.shape == (grid.numx, grid.numz):
        pass
    else:
        raise ValueError('invalid data shape (got {}, expected {} or {})'.format(data.shape, (grid.numx, 1, grid.numz),
                                                                                 (grid.numx, grid.numz)))

    data = np.rot90(data)

    scale = scale.lower()
    if scale == 'linear':
        if ref_db is not None:
            warn("ref_db is ignored for linear plot", ArimWarning)
        filename_scale = '_linear'
        title_scale = ' [lin. scale]'
    elif scale == 'db':
        filename_scale = '_db'
        title_scale = ' [dB scale]'
        data = utils.decibel(data, ref_db)
    elif scale == 'none':
        filename_scale = ''
        title_scale = ''
    else:
        raise ValueError('invalid scale: {}'.format(scale))
    if not add_scale_to_filename:
        # overwrite:
        filename_scale = ''
    if not add_scale_to_title:
        # overwrite:
        title_scale = ''

    image = ax.imshow(data, interpolation=interpolation, origin='lower',
                      extent=(grid.xmin, grid.xmax, grid.zmax, grid.zmin),
                      cmap=cmap)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.xaxis.set_major_formatter(mm_formatter)
    ax.xaxis.set_minor_formatter(mm_formatter)
    ax.yaxis.set_major_formatter(mm_formatter)
    ax.yaxis.set_minor_formatter(mm_formatter)
    if draw_cbar:
        fig.colorbar(image, ax=ax)
    if clim is not None:
        image.set_clim(clim)
    if title is not None:
        ax.set_title(title + title_scale)

    ax.axis('equal')
    ax.axis([grid.xmin, grid.xmax, grid.zmax, grid.zmin])
    if savefig:
        if filename is None:
            raise ValueError('filename must be provided when savefig is true')
        root, ext = os.path.splitext(filename)
        filename = root + filename_scale + ext
        fig.savefig(filename)
    return ax, image


def plot_oxz_many(data_list, grid, nrows, ncols, title_list=None, suptitle=None,
                  draw_colorbar=True, figsize=None, savefig=None, clim=None, filename=None,
                  y_title=1.0, y_subtitle=1.0, **plot_oxz_kwargs):
    if savefig is None:
        savefig = conf['savefig']
    if figsize is None:
        figsize = conf['plot_oxz_many.figsize']

    if title_list is None:
        title_list = [None] * len(data_list)

    # must use a common clim (otherwise the figure does not make sense)
    if clim is None:
        clim = (min(np.nanmin(x) for x in data_list), max(np.nanmax(x) for x in data_list))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                             figsize=figsize)
    images = []
    for data, title, ax in zip(data_list, title_list, axes.ravel()):
        # the current function handles saving fig, drawing the cbar and displaying the title
        # so we prevent plot_oxz to do it.
        ax, im = plot_oxz(data, grid, ax=ax, clim=clim, draw_cbar=False, savefig=False, **plot_oxz_kwargs, title=None)
        images.append(im)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if title is not None:
            ax.set_title(title, y=y_title)
    for ax in axes[-1, :]:
        ax.set_xlabel("x (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z (mm)")
    if suptitle is not None:
        fig.suptitle(suptitle, y=y_subtitle, size="x-large")
    if draw_colorbar:
        fig.colorbar(im, ax=axes.ravel().tolist())
    if savefig:
        if filename is None:
            raise ValueError('filename must be provided when savefig is true')
        fig.savefig(filename)
    return axes, images


def plot_tfm(tfm, y=0., func_res=None, ax=None, title='TFM', clim=None, interpolation='bilinear', draw_cbar=True,
             cmap=None):
    """
    Plot a TFM in plane Oxz.

    Parameters
    ----------
    tfm : BaseTFM
    y : float
    ax
    clim
    interpolation : str
        Cf matplotlib.pyplot.imshow
    draw_cbar : boolean, optional
        Default: True
    func_res : function
        Function to apply on tfm.res before plotting it. Example: ``lambda x: np.abs(x)``


    Returns
    -------
    ax
    image

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    grid = tfm.grid
    iy = np.argmin(np.abs(grid.y - y))

    if tfm.res is None:
        raise ValueError('No result in this TFM object.')
    if func_res is None:
        func_res = lambda x: x

    data = func_res(tfm.res[:, iy, :])

    return plot_oxz(data, grid=grid, ax=ax, title=title, clim=clim,
                    interpolation=interpolation, draw_cbar=draw_cbar, cmap=cmap)


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

    title = kwargs.get('title', 'Directivity of an element (uniform sources along a straight line)')

    ratio = element_width / wavelength
    theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
    directivity = model.directivity_finite_width_2d(theta, element_width, wavelength)

    ax.plot(np.rad2deg(theta), directivity, label=r'$a/\lambda = {:.2f}$'.format(ratio), **kwargs)
    ax.set_xlabel(r'Polar angle $\theta$ (deg)')
    ax.set_ylabel('directivity (1)')
    ax.set_title(title)
    ax.set_xlim([-90, 90])
    ax.set_ylim([0, 1.2])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30.))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(15.))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.legend()
    return ax


class RayPlotter:
    def __init__(self, grid, ray, element_index, linestyle='m--', tolerance_distance=1e-3):
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
            'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
        ax = event.canvas.figure.axes[0]
        if event.button == 1:
            self.draw_ray(ax, event.xdata, event.ydata)
        elif event.button == 3:
            self.clear_rays(ax)
        if self.debug:
            print('show_ray_on_clic() finish with no error')

    def draw_ray(self, ax, x, z):
        gridpoints = self.grid.as_points
        wanted_point = (x, self.y, z)
        point_index = gridpoints.closest_point(*wanted_point)
        obtained_point = gridpoints[point_index]

        distance = g.norm2(*(obtained_point - wanted_point))
        if distance > self.tolerance_distance:
            logger.warning("The closest grid point is far from what you want (dist: {:.2f} mm)".format(distance * 1000))

        legs = self.ray.get_coordinates_one(self.element_index, point_index)
        line = ax.plot(legs.x, legs.z, self.linestyle)
        self._lines.extend(line)
        logger.debug('Draw a ray')
        ax.figure.canvas.draw_idle()

    def clear_rays(self, ax):
        """Clear all rays"""
        lines_to_clear = [line for line in ax.lines if line in self._lines]
        for line in lines_to_clear:
            ax.lines.remove(line)
            self._lines.remove(line)
        logger.debug('Clear {} ray(s) on figure'.format(len(lines_to_clear)))
        ax.figure.canvas.draw_idle()

    def connect(self, ax):
        """Connect to matplotlib event backend"""
        ax.figure.canvas.mpl_connect('button_press_event', self)


def draw_rays_on_click(grid, ray, element_index, ax=None, linestyle='m--', ):
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
    ray_plotter = RayPlotter(grid=grid, ray=ray, element_index=element_index, linestyle=linestyle)
    ray_plotter.connect(ax)
    return ray_plotter
