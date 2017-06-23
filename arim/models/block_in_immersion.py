import logging
from collections import namedtuple

import numpy as np

from .. import model
from .. import core as c
from ..path import RayGeometry

logger = logging.getLogger(__name__)

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
    path : Path
    ray_geometry : RayGeometry
    frequency : float
    probe_element_width : float or None
        Mandatory if use_directivity is True
    use_directivity : bool
        Default True
    use_beamspread : bool
        Default True
    use_transrefl : bool
        Default: True

    Returns
    -------
    weights : ndarray
        Shape (numelements, numgridpoints)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection
    """
    d = _init_ray_weights(path, frequency, probe_element_width, use_directivity)

    weights_dict = dict()
    one = np.ones((len(path.interfaces[0].points), d.numgridpoints), order='F')

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
    path : Path
    ray_geometry : RayGeometry
    frequency : float
    probe_element_width : float or None
        Mandatory if use_directivity is True
    use_directivity : bool
        Default True
    use_beamspread : bool
        Default True
    use_transrefl : bool
        Default: True

    Returns
    -------
    weights : ndarray
        Shape (numelements, numgridpoints)
    weights_dict : dict[str, ndarray]
        Components of the ray weights: beamspread, directivity, transmission-reflection
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


def ray_weights_for_views(views, frequency, probe_element_width=None,
                          use_directivity=True, use_beamspread=True,
                          use_transrefl=True, save_debug=False):
    """
    Compute coefficients Q_i(r, omega) and Q'_j(r, omega) from the forward model for
    all views.
    NB: do not compute the scattering.

    Internally use :func:`tx_ray_weights` and :func:`rx_way_weights`.

    Parameters
    ----------
    views
    frequency
    probe_element_width
    use_directivity
    use_beamspread
    use_transrefl
    save_debug

    Returns
    -------
    RayWeights
    """
    tx_ray_weights_dict = {}
    rx_ray_weights_dict = {}
    if save_debug:
        tx_ray_weights_debug_dict = {}
        rx_ray_weights_debug_dict = {}
    else:
        tx_ray_weights_debug_dict = None
        rx_ray_weights_debug_dict = None
    scat_angle_dict = {}

    all_tx_paths = {view.tx_path for view in views.values()}
    all_rx_paths = {view.rx_path for view in views.values()}
    all_paths = all_tx_paths | all_rx_paths

    model_options = dict(frequency=frequency,
                         probe_element_width=probe_element_width,
                         use_beamspread=use_beamspread,
                         use_directivity=use_directivity,
                         use_transrefl=use_transrefl)

    # By proceeding this way, geometrical computations can be reused for both
    # tx and rx path.
    for path in all_paths:
        ray_geometry = RayGeometry.from_path(path)
        scat_angle_dict[path] = np.asfortranarray(ray_geometry.signed_inc_angle(-1))
        scat_angle_dict[path].flags.writeable = False

        if path in all_tx_paths:
            ray_weights, ray_weights_debug = tx_ray_weights(path, ray_geometry,
                                                            **model_options)
            ray_weights = np.asfortranarray(ray_weights)
            ray_weights.flags.writeable = False
            tx_ray_weights_dict[path] = ray_weights
            if save_debug:
                tx_ray_weights_debug_dict[path] = ray_weights_debug
            del ray_weights, ray_weights_debug
        if path in all_rx_paths:
            ray_weights, ray_weights_debug = rx_ray_weights(path, ray_geometry,
                                                            **model_options)
            ray_weights = np.asfortranarray(ray_weights)
            ray_weights.flags.writeable = False
            rx_ray_weights_dict[path] = ray_weights
            if save_debug:
                rx_ray_weights_debug_dict[path] = ray_weights_debug
            del ray_weights, ray_weights_debug

    return model.RayWeights(tx_ray_weights_dict, rx_ray_weights_dict,
                            tx_ray_weights_debug_dict, rx_ray_weights_debug_dict,
                            scat_angle_dict)
