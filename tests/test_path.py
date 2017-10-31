import numpy as np

import arim
import arim.path as p


def test_filter_unique_views():
    unique_views = p.filter_unique_views([('AB', 'CD'), ('DC', 'BA'), ('X', 'YZ'),
                                          ('ZY', 'X')])
    assert unique_views == [('AB', 'CD'), ('X', 'YZ')]


def test_make_viewnames():
    L = 'L'
    T = 'T'
    LL = 'LL'

    viewnames = p.make_viewnames(['L', 'T'], unique_only=False)
    assert viewnames == [(L, L), (L, T), (T, L), (T, T)]

    viewnames = p.make_viewnames(['L', 'T'], unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T)]

    viewnames = p.make_viewnames(['L', 'T'], unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T)]

    # legacy IMAGING_MODES
    legacy_imaging_views = ["L-L", "L-T", "T-T",
                            "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L",
                            "TT-T",
                            "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                            "LT-LT", "LT-TL", "LT-TT",
                            "TL-LT", "TL-TT",
                            "TT-TT"]
    legacy_imaging_views = [tuple(view.split('-')) for view in legacy_imaging_views]

    viewnames = p.make_viewnames(['L', 'T', 'LL', 'LT', 'TL', 'TT'])
    assert viewnames == legacy_imaging_views

    viewnames = p.make_viewnames(p.DIRECT_PATHS + p.SKIP_PATHS)
    assert viewnames == legacy_imaging_views

    viewnames = p.make_viewnames(p.DIRECT_PATHS + p.SKIP_PATHS + p.DOUBLE_SKIP_PATHS)
    assert viewnames[:21] == legacy_imaging_views
    assert len(viewnames) == 105

    viewnames = p.make_viewnames(p.DIRECT_PATHS + p.SKIP_PATHS + p.DOUBLE_SKIP_PATHS,
                                 unique_only=False)
    assert len(viewnames) == 14 * 14

    viewnames = p.make_viewnames(['L', 'T', 'LL'], unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T), (LL, L), (LL, T), (LL, LL)]


def test_points_1d_wall_z():
    args = dict(xmin=10, xmax=20, numpoints=6, y=1, z=2, name='toto')
    points, orientations = p.points_1d_wall_z(**args)
    assert points.shape == (6,)
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

    probe_points, probe_orientations = p.points_1d_wall_z(0e-3, 15e-3,
                                                          z=0., numpoints=16,
                                                          name='Frontwall')

    frontwall_points, frontwall_orientations = p.points_1d_wall_z(xmin, xmax,
                                                                  z=0., numpoints=20,
                                                                  name='Frontwall')
    backwall_points, backwall_orientations = p.points_1d_wall_z(xmin, xmax,
                                                                z=40.18e-3, numpoints=21,
                                                                name='Backwall')

    grid = arim.geometry.Grid(xmin, xmax,
                              ymin=0., ymax=0.,
                              zmin=0., zmax=20e-3,
                              pixel_size=5e-3)
    grid_points, grid_orientation = p.points_from_grid(grid)

    interfaces = p.interfaces_for_block_in_immersion(couplant, probe_points,
                                                     probe_orientations,
                                                     frontwall_points,
                                                     frontwall_orientations,
                                                     backwall_points,
                                                     backwall_orientations,
                                                     grid_points, grid_orientation)
    assert interfaces['probe'].points is probe_points
    assert interfaces['probe'].orientations is probe_orientations
    assert interfaces['frontwall_trans'].points is frontwall_points
    assert interfaces['frontwall_trans'].orientations is frontwall_orientations
    assert interfaces['frontwall_refl'].points is frontwall_points
    assert interfaces['frontwall_refl'].orientations is frontwall_orientations
    assert interfaces['backwall_refl'].points is backwall_points
    assert interfaces['backwall_refl'].orientations is backwall_orientations
    assert interfaces['grid'].points is grid_points
    assert interfaces['grid'].orientations is grid_orientation

    # ------------------------------------------------------------------------------------
    # 6 paths, 21 views

    paths = p.paths_for_block_in_immersion(block, couplant, interfaces)
    assert len(paths) == 6
    assert paths['L'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                           frontwall_points, block.longitudinal_vel,
                                           grid_points)
    assert paths['TL'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                            frontwall_points, block.transverse_vel,
                                            backwall_points,
                                            block.longitudinal_vel, grid_points)

    for path_key, path in paths.items():
        assert path_key == path.name

    # Make views
    views = p.views_for_block_in_immersion(paths)
    assert len(views) == 21

    view = views['LT-LT']
    assert view.tx_path is paths['LT']
    assert view.rx_path is paths['TL']

    # ------------------------------------------------------------------------------------
    # 14 paths, 105 views
    paths = p.paths_for_block_in_immersion(block, couplant, interfaces,
                                           max_number_of_reflection=2)
    assert len(paths) == 14
    assert paths['L'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                           frontwall_points, block.longitudinal_vel,
                                           grid_points)
    assert paths['TL'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                            frontwall_points, block.transverse_vel,
                                            backwall_points,
                                            block.longitudinal_vel, grid_points)
    assert paths['TTL'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                             frontwall_points, block.transverse_vel,
                                             backwall_points, block.transverse_vel,
                                             frontwall_points, block.longitudinal_vel,
                                             grid_points)

    for path_key, path in paths.items():
        assert path_key == path.name

    # Make views
    views = p.views_for_block_in_immersion(paths)
    assert len(views) == 105

    view = views['LT-LT']
    assert view.tx_path is paths['LT']
    assert view.rx_path is paths['TL']
