import numpy as np
from collections import OrderedDict

from tests.test_model import make_context
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
