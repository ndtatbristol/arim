import numpy as np

from tests.test_model import make_context
import arim
import arim.models.block_in_immersion as bim
import arim.ray

def test_ray_weights():
    context = make_context()
    paths = context['paths']
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    model_options = dict(frequency=context['freq'],
                         probe_element_width=context['element_width'],
                         use_beamspread=True,
                         use_directivity=True,
                         use_transrefl=True)

    for pathname, path in paths.items():
        # Direct
        ray_geometry = ray_geometry_dict[pathname]
        ray_geometry2 = arim.ray.RayGeometry.from_path(path, use_cache=False)
        weights, weights_dict = bim.tx_ray_weights(path, ray_geometry, **model_options)
        weights2, _ = bim.tx_ray_weights(path, ray_geometry2, **model_options)
        assert 'beamspread' in weights_dict
        assert 'directivity' in weights_dict
        assert 'transrefl' in weights_dict
        np.testing.assert_allclose(weights, weights2)

        # Reverse
        ray_geometry = ray_geometry_dict[pathname]
        ray_geometry2 = arim.ray.RayGeometry.from_path(path, use_cache=False)
        weights, weights_dict = bim.rx_ray_weights(path, ray_geometry, **model_options)
        weights2, _ = bim.rx_ray_weights(path, ray_geometry2, **model_options)
        assert 'beamspread' in weights_dict
        assert 'directivity' in weights_dict
        assert 'transrefl' in weights_dict
        np.testing.assert_allclose(weights, weights2)


def test_ray_weights_for_views():
    context = make_context()
    views = context['views']
    paths = context['paths']
    paths_set = set(paths.values())

    ray_weights_cache = bim.ray_weights_for_views(
        views, frequency=context['freq'], probe_element_width=context['element_width'])

    assert ray_weights_cache.tx_ray_weights_debug_dict is None
    assert ray_weights_cache.rx_ray_weights_debug_dict is None
    assert len(paths) >= len(ray_weights_cache.tx_ray_weights_dict) > 3
    assert len(paths) >= len(ray_weights_cache.rx_ray_weights_dict) > 3
    assert set(ray_weights_cache.rx_ray_weights_dict.keys()) == paths_set
    nbytes_without_debug = ray_weights_cache.nbytes
    assert nbytes_without_debug > 0

    ray_weights_cache = bim.ray_weights_for_views(
        views, frequency=context['freq'], probe_element_width=context['element_width'],
        save_debug=True)

    assert ray_weights_cache.tx_ray_weights_debug_dict.keys() == \
           ray_weights_cache.tx_ray_weights_dict.keys()
    assert ray_weights_cache.rx_ray_weights_debug_dict.keys() == \
           ray_weights_cache.rx_ray_weights_dict.keys()
    assert len(paths) >= len(ray_weights_cache.tx_ray_weights_dict) > 3
    assert len(paths) >= len(ray_weights_cache.rx_ray_weights_dict) > 3
    assert set(ray_weights_cache.rx_ray_weights_dict.keys()) == paths_set
    nbytes_with_debug = ray_weights_cache.nbytes
    assert nbytes_with_debug > nbytes_without_debug


def test_model_amplitudes():
    context = make_context()

    block = context['block']
    """:type : arim.Material"""
    couplant = context['couplant']
    """:type : arim.Material"""
    vl = block.longitudinal_vel
    vt = block.transverse_vel

    scattering_func = {
        'LL': lambda inc, out: np.full_like(inc, 1.),
        'LT': lambda inc, out: np.full_like(inc, vl / vt),
        'TL': lambda inc, out: np.full_like(inc, vt / vl),
        'TT': lambda inc, out: np.full_like(inc, 1.),
    }




    interfaces = context['interfaces']
    """:type : list[arim.Interface]"""
    rev_paths = context['rev_paths']
    """:type : dict[str, arim.Path]"""
    paths = context['paths']
    """:type : dict[str, arim.Path]"""
    views = context['views']
    """:type : dict[str, arim.View]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""
    frontwall_points = context['frontwall_points']
    frontwall_orientations = context['frontwall_orientations']
    backwall_points = context['backwall_points']
    backwall_orientations = context['backwall_orientations']
    scatterer_points = context['scatterer_points']
    scatterer_orientations = context['scatterer_orientations']


def test_path_in_immersion():
    xmin = -20e-3
    xmax = 100e-3

    couplant = arim.Material(longitudinal_vel=1480, transverse_vel=None, density=1000.,
                             state_of_matter='liquid', metadata={'long_name': 'Water'})
    block = arim.Material(longitudinal_vel=6320., transverse_vel=3130., density=2700.,
                          state_of_matter='solid', metadata={'long_name': 'Aluminium'})

    probe_points, probe_orientations = arim.geometry.points_1d_wall_z(0e-3, 15e-3,
                                                                      z=0., numpoints=16,
                                                                      name='Frontwall')

    frontwall_points, frontwall_orientations = arim.geometry.points_1d_wall_z(xmin, xmax,
                                                                              z=0., numpoints=20,
                                                                              name='Frontwall')
    backwall_points, backwall_orientations = arim.geometry.points_1d_wall_z(xmin, xmax,
                                                                            z=40.18e-3, numpoints=21,
                                                                            name='Backwall')

    grid = arim.geometry.Grid(xmin, xmax,
                              ymin=0., ymax=0.,
                              zmin=0., zmax=20e-3,
                              pixel_size=5e-3)
    grid_points, grid_orientation = arim.geometry.points_from_grid(grid)

    interfaces = arim.models.block_in_immersion.make_interfaces(couplant, probe_points,
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

    paths = arim.models.block_in_immersion.make_paths(block, couplant, interfaces)
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
    views = arim.models.block_in_immersion.make_views(paths)
    assert len(views) == 21

    view = views['LT-LT']
    assert view.tx_path is paths['LT']
    assert view.rx_path is paths['TL']

    # ------------------------------------------------------------------------------------
    # 14 paths, 105 views
    paths = arim.models.block_in_immersion.make_paths(block, couplant, interfaces,
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
    views = arim.models.block_in_immersion.make_views(paths)
    assert len(views) == 105

    view = views['LT-LT']
    assert view.tx_path is paths['LT']
    assert view.rx_path is paths['TL']