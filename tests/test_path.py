import arim
import numpy as np


def test_points_1d_wall_z():
    args = dict(xmin=10, xmax=20, numpoints=6, y=1, z=2, name='toto')
    points, orientations = arim.points_1d_wall_z(**args)
    assert points.shape == (6, )
    assert orientations.shape == (6, 3)

    np.testing.assert_allclose(points.x, [10, 12, 14, 16, 18, 20])
    np.testing.assert_allclose(points.y, args['y'])
    np.testing.assert_allclose(points.z, args['z'])
    for i in range(6):
        np.testing.assert_allclose(orientations[i], np.eye(3))


def test_path_in_immersion():
    xmin = -20e-3
    xmax = 100e-3

    couplant = arim.Material(longitudinal_vel=1480, transverse_vel=None, density=1000.,
                          state_of_matter='liquid', metadata={'long_name': 'Water'})
    block = arim.Material(longitudinal_vel=6320., transverse_vel=3130., density=2700.,
                          state_of_matter='solid', metadata={'long_name': 'Aluminium'})

    probe_points, probe_orientations = arim.points_1d_wall_z(0e-3, 15e-3,
        z=0., numpoints=16, name='Frontwall')

    frontwall_points, frontwall_orientations = arim.points_1d_wall_z(xmin, xmax,
        z=0., numpoints=20, name='Frontwall')
    backwall_points, backwall_orientations = arim.points_1d_wall_z(xmin, xmax,
        z=40.18e-3, numpoints=21, name='Backwall')

    grid = arim.geometry.Grid(xmin, xmax,
                              ymin=0., ymax=0.,
                              zmin=0., zmax=20e-3,
                              pixel_size=5e-3)
    grid_points, grid_orientation = arim.points_from_grid(grid)

    interfaces = arim.interfaces_for_block_in_immersion(couplant, probe_points, probe_orientations,
        frontwall_points, frontwall_orientations, backwall_points, backwall_orientations,
        grid_points, grid_orientation)
    assert interfaces[0].points is probe_points
    assert interfaces[-1].points is grid_points

    paths = arim.paths_for_block_in_immersion(block, couplant, *interfaces)
    assert paths['L'].to_fermat_path() == (probe_points, couplant.longitudinal_vel, 
        frontwall_points, block.longitudinal_vel, grid_points)
    assert paths['TL'].to_fermat_path() == (probe_points, couplant.longitudinal_vel, 
        frontwall_points, block.transverse_vel, backwall_points,
        block.longitudinal_vel, grid_points)

    for path_key, path in paths.items():
        assert path_key == path.name

    # Make views
    views = arim.views_for_block_in_immersion(paths)

    view = views['LT-LT']
    assert view.tx_path is paths['LT']
    assert view.rx_path is paths['TL']
