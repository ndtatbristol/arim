import numpy as np
from collections import OrderedDict

from tests.test_model import make_context, make_point_source_scattering_func
import arim
import arim.models.block_in_immersion as bim


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
        ray_geometry2 = arim.path.RayGeometry.from_path(path, use_cache=False)
        weights, weights_dict = bim.tx_ray_weights(path, ray_geometry, **model_options)
        weights2, _ = bim.tx_ray_weights(path, ray_geometry2, **model_options)
        assert 'beamspread' in weights_dict
        assert 'directivity' in weights_dict
        assert 'transrefl' in weights_dict
        np.testing.assert_allclose(weights, weights2)

        # Reverse
        ray_geometry = ray_geometry_dict[pathname]
        ray_geometry2 = arim.path.RayGeometry.from_path(path, use_cache=False)
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
