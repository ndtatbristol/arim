"""
Plotting utilities
"""


import functools
from warnings import warn

import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

from . import utils, model
from .exceptions import ArimWarning

# __all__ = ['mm_formatter', 'us_formatter']

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
