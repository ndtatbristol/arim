import copy

import numpy as np

from .. import core, _probes, geometry

__all__ = ['block_in_immersion_from_conf', 'grid_from_conf', 'probe_from_conf']

def probe_from_conf(conf):
    """
    load probe from conf

    Parameters
    ----------
    conf : dict

    Returns
    -------
    Probe

    """
    # load from probe library
    if 'probe_key' in conf:
        probe = _probes.probes[conf['probe_key']]
    else:
        probe = core.Probe.make_matrix_probe(**conf['probe'])

    if 'probe_location' in conf:
        probe_location = conf['probe_location']

        if 'ref_element' in probe_location:
            probe.set_reference_element(conf['probe_location']['ref_element'])
            probe.translate_to_point_O()

        if 'angle_deg' in probe_location:
            probe.rotate(geometry.rotation_matrix_y(
                np.deg2rad(conf['probe_location']['angle_deg'])))

        if 'standoff' in probe_location:
            probe.translate([0, 0, conf['probe_location']['standoff']])

    return probe


def block_in_immersion_from_conf(conf):
    """
    load block in immersion from conf

    Parameters
    ----------
    conf : dict

    Returns
    -------
    arim.BlockInImmersion

    """
    couplant = core.Material(**conf['couplant_material'])
    block = core.Material(**conf['block_material'])
    frontwall = geometry.points_1d_wall_z(**conf['frontwall'], name='Frontwall')
    backwall = geometry.points_1d_wall_z(**conf['backwall'], name='Backwall')
    return core.BlockInImmersion(block, couplant, frontwall, backwall)


def grid_from_conf(conf):
    """
    load grid from conf

    Parameters
    ----------
    conf : dict

    Returns
    -------
    arim.Grid

    """
    conf_grid = copy.deepcopy(conf['grid'])
    if 'ymin' not in conf_grid:
        conf_grid['ymin'] = 0.
    if 'ymax' not in conf_grid:
        conf_grid['ymax'] = 0.
    return geometry.Grid(**conf_grid)
