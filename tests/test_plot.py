import arim
import arim.plot as aplt
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import matplotlib as mpl


def test_plot_oxz_many(show_plots):
    grid = arim.Grid(-5e-3, 5e-3, 0, 0, 0, 15e-3, .1e-3)
    k = 0.01e-3
    data = np.exp(-grid.xx ** 2 / k - (grid.zz - 5e-3) ** 2 / (2*k))

    nrows = 2
    ncols = 3
    data_list = [data * (i + 1) for i in range(nrows * ncols)]
    title_list = ['Plot {}'.format(i + 1) for i in range(nrows * ncols)]

    figsize = (12, 8)

    ax_list, im_list = aplt.plot_oxz_many(data_list, grid, nrows, ncols, figsize=figsize)
    plt.close('all')

    ax_list, im_list = aplt.plot_oxz_many(data_list, grid, nrows, ncols, title_list=title_list,
                                          suptitle='Many plots', figsize=figsize, y_suptitle=0.98)
    if show_plots:
        plt.show()
    else:
        plt.close('all')


def test_plot_oxz(show_plots):
    grid = arim.Grid(-5e-3, 5e-3, 0, 0, 0, 15e-3, .1e-3)
    k = 2 * np.pi / 10e-3
    data = (np.cos(grid.xx * 2 * k) * np.sin(grid.zz * k)) * (grid.zz ** 2)

    # check it works without error
    ax, im = aplt.plot_oxz(data, grid)
    plt.close('all')

    ax, im = aplt.plot_oxz(data.reshape((grid.numx, grid.numz)), grid, scale='linear', title='some linear stuff')
    if show_plots:
        plt.show()
    else:
        plt.close('all')

    with tempfile.TemporaryDirectory() as dirname:
        out_file = Path(dirname) / Path('toto.png')
        ax, im = aplt.plot_oxz(data, grid, title='some db stuff', scale='db', clim=[-12, 0], savefig=True,
                               filename=str(out_file))
        if show_plots:
            plt.show()
        else:
            plt.close('all')
        assert out_file.exists()
