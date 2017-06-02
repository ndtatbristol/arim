from collections import namedtuple, OrderedDict

import numpy as np

from .. import model
from .. import core as c

BlockInImmersionCache = namedtuple('BlockInImmersionCache', [
    'ray_geometry_dict', 'direct_path_weights_dict', 'reverse_path_weights_dict'
])

_RayWeightsCommon = namedtuple('_RayWeightsCommon',
                               ['couplant', 'numgridpoints', 'wavelength_in_couplant',
                                'wavelengths_in_block'])


def _init_ray_weights(path, frequency, probe_element_width, use_directivity):
    if path.rays is None:
        raise ValueError('Ray tracing must have been performed first.')

    couplant, block = path.materials[:2]
    numgridpoints = len(path.interfaces[-1].points)

    if use_directivity and probe_element_width is None:
        raise ValueError(
            'probe_element_width must be provided to compute directivity')

    wavelength_in_couplant = couplant.longitudinal_vel / frequency
    wavelengths_in_block = dict([(c.Mode.L, block.longitudinal_vel / frequency),
                                 (c.Mode.T, block.transverse_vel / frequency)])

    return _RayWeightsCommon(couplant, numgridpoints, wavelength_in_couplant,
                             wavelengths_in_block)


def tx_ray_weights(path, ray_geometry, frequency, probe_element_width=None,
                   use_directivity=True, use_beamspread=True, use_transrefl=True):
    """
    Coefficients Q_i(r, omega) in forward model.

    Parameters
    ----------
    path
    ray_geometry
    frequency
    probe_element_width
    use_directivity
    use_beamspread
    use_transrefl

    Returns
    -------

    """
    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints))

    if use_directivity:
        weights_dict['directivity'] = model.radiation_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant)
    else:
        weights_dict['directivity'] = one
    if use_transrefl:
        # use conjugate because we forget to put it in the model
        weights_dict['transrefl'] = model.transmission_reflection_for_path(
            path, ray_geometry, unit='displacement').conjugate()
    else:
        weights_dict['transrefl'] = one
    if use_beamspread:
        weights_dict['beamspread'] = (model.beamspread_2d_for_path(ray_geometry) *
                                      np.sqrt(d.wavelength_in_couplant))
    else:
        weights_dict['beamspread'] = one

    weights = (weights_dict['directivity'] *
               weights_dict['transrefl'] *
               weights_dict['beamspread'])
    return weights, weights_dict


def rx_ray_weights(path, ray_geometry, frequency, probe_element_width=None,
                   use_directivity=True, use_beamspread=True, use_transrefl=True):
    """
    Coefficients Q'_i(r, omega) in forward model.

    Parameters
    ----------
    path
    ray_geometry
    frequency
    probe_element_width
    use_directivity
    use_beamspread
    use_transrefl

    Returns
    -------

    """
    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints))

    if use_directivity:
        weights_dict['directivity'] = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, probe_element_width, d.wavelength_in_couplant)
    else:
        weights_dict['directivity'] = one
    if use_transrefl:
        # use conjugate because we forget to put it in the model
        weights_dict['transrefl'] = model.reverse_transmission_reflection_for_path(
            path, ray_geometry, unit='displacement').conjugate()
    else:
        weights_dict['transrefl'] = one
    if use_beamspread:
        weights_dict['beamspread'] = (
            model.reverse_beamspread_2d_for_path(ray_geometry)
            * np.sqrt(d.wavelengths_in_block[path.modes[-1]]))
    else:
        weights_dict['beamspread'] = one

    weights = (weights_dict['directivity'] *
               weights_dict['transrefl'] *
               weights_dict['beamspread'])
    return weights, weights_dict
