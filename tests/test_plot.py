import arim
import arim.plot as aplt
import numpy as np
import matplotlib.pyplot as plt


def test_plot_oxz(show_plots):
    grid = arim.geometry.Grid(-5e-3, 5e-3, 0, 0, 0, 15e-3, .1e-3)
    k = 2 * np.pi / 10e-3
    data = (np.cos(grid.xx * 2 * k) * np.sin(grid.zz * k)) * grid.zz ** 2

    # check it works without error
    ax, im = aplt.plot_oxz(data, grid)
    plt.close('all')

    ax, im = aplt.plot_oxz(data.reshape((grid.numx, grid.numz)), grid)
    plt.close('all')

    ax, im = aplt.plot_oxz(data, grid, title='some linear stuff', scale='linear')
    if show_plots:
        plt.show()
    else:
        plt.close('all')

    ax, im = aplt.plot_oxz(data, grid, title='some db stuff', scale='db', clim=[-12, 0])
    if show_plots:
        plt.show()
    else:
        plt.close('all')
