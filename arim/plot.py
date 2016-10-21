"""
Plotting utilities
"""

import functools
from warnings import warn
import logging

import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

from . import utils, model
from .exceptions import ArimWarning
from . import geometry as g

__all__ = ['mm_formatter', 'us_formatter',
           'plot_bscan_pulse_echo', 'plot_oxz', 'plot_tfm', 'plot_directivity_finite_width_2d',
           'draw_rays_on_click', 'RayPlotter']

logger = logging.getLogger(__name__)

mm_formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 1e3))
us_formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 1e6))


def _elements_ticker(i, pos, elements):
    i = int(i)
    if i >= len(elements):
        return ''
    else:
        return str(elements[i])


def plot_bscan_pulse_echo(frame, use_dB=True, ax=None, title='B-scan', clim=None, interpolation='none', draw_cbar=True,
                          cmap=None):
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


def plot_oxz(data, grid, ax=None, title=None, clim=None, interpolation='none', draw_cbar=True, cmap=None):
    """
    Plot an image on plane Oxz.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    assert data.ndim == 2
    if data.shape != (grid.numx, grid.numz):
        warn('The shape of the data is not (grid.numx, grid.numz).', ArimWarning)

    data = np.rot90(data)
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
        ax.set_title(title)

    ax.axis('equal')
    return ax, image


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
