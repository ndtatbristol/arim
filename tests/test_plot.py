import arim
import arim.plot as aplt
import arim.geometry as g
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import pytest
from collections import OrderedDict


def test_plot_oxz_many(show_plots):
    grid = arim.Grid(-5e-3, 5e-3, 0, 0, 0, 15e-3, .1e-3)
    k = 0.01e-3
    data = np.exp(-grid.xx ** 2 / k - (grid.zz - 5e-3) ** 2 / (2 * k))

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


@pytest.mark.parametrize('plot_interfaces_kwargs',
                         [dict(),
                          dict(show_orientations=True, show_grid=True),
                          ])
def test_plot_interfaces(show_plots, plot_interfaces_kwargs):
    # setup interfaces
    numinterface = 200
    numinterface2 = 200

    xmin = -5e-3
    xmax = 60e-3
    z_backwall = 20e-3

    points, orientations = arim.path.points_1d_wall_z(0, 12e-3, z=0., numpoints=64, name='Probe')
    rot = g.rotation_matrix_y(np.deg2rad((12)))
    points = points.rotate(rot)
    points = points.translate((0, 0, -10e-3))
    orientations = orientations.rotate(rot)
    probe = arim.Interface(points, orientations)
    assert probe.orientations[0, 2, 0] > 0
    assert probe.orientations[0, 2, 2] > 0

    points, orientations = arim.path.points_1d_wall_z(xmin, xmax,
                                                 z=0., numpoints=numinterface,
                                                 name='Frontwall')
    frontwall = arim.Interface(points, orientations)

    points, orientations = arim.path.points_1d_wall_z(xmin, xmax, z=z_backwall, numpoints=numinterface2, name='Backwall')
    backwall = arim.Interface(points, orientations)

    grid_obj = arim.Grid(xmin, xmax, 0, 0, 0, z_backwall, 1e-3)
    grid = arim.Interface(*arim.path.points_from_grid(grid_obj))

    interfaces = [probe, frontwall, backwall, grid]
    # end setup interfaces

    aplt.plot_interfaces(interfaces, **plot_interfaces_kwargs)

    # non_unique_interfaces = [probe, frontwall, backwall, grid, frontwall]
    # aplt.plot_interfaces(non_unique_interfaces, **plot_interfaces_kwargs,
    #                      unique_points_only=True)

    if show_plots:
        plt.show()
    else:
        plt.close('all')
