import pathlib
import copy
import yaml
import os
import numpy as np

from .. import core, _probes, geometry, config

__all__ = ['load_conf', 'load_conf_file', 'probe_from_conf', 'examination_object_from_conf',
           'block_in_immersion_from_conf', 'grid_from_conf']


class InvalidConf(Exception):
    pass


class _ConfigLoader(yaml.Loader):
    """
    Yaml loader with "!include ext_file.yaml" directive

    Adapted from https://stackoverflow.com/a/9577670/2996578
    """

    def __init__(self, stream):
        if isinstance(stream, str):
            self._root = '.'
        else:
            self._root = os.path.split(stream.name)[0]

        super(_ConfigLoader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, _ConfigLoader)


_ConfigLoader.add_constructor('!include', _ConfigLoader.include)


def load_conf_file(filename):
    """Load configuration file"""
    with open(filename, 'r') as f:
        return config.Config(yaml.load(f, _ConfigLoader))


def load_conf(dirname):
    """
    Load the configuration from a '.arim' directory

    Parameters
    ----------
    dirname

    Returns
    -------
    arim.config.Config
    """
    root_dir = pathlib.Path(dirname).resolve(strict=True)

    # format: foo/bar/{dataset_name}.arim
    dataset_name = root_dir.parts[-1]
    if dataset_name.endswith('.arim'):
        dataset_name = dataset_name[:-5]

    # Load root conf
    root_conf_filename = root_dir / 'conf.yaml'
    if root_conf_filename.exists():
        conf = load_conf_file(root_conf_filename)
    else:
        conf = config.Config({})

    # Load conf.d fragment files
    # Remark: no error if root_dir/conf.d doesn't exist
    for conf_filename in root_dir.glob('conf.d/*.yaml'):
        conf.merge(load_conf_file(conf_filename))

    # Populate extra keys
    conf['dataset_name'] = dataset_name
    conf['root_dir'] = root_dir

    # populate result_dir if missing, create dir if needed
    result_dir = conf.get('result_dir', None)
    if result_dir is None:
        result_dir = root_dir
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = pathlib.Path(root_dir / pathlib.Path(result_dir)).resolve(strict=True)
    conf['result_dir'] = result_dir

    return conf


def probe_from_conf(conf, apply_probe_location=True):
    """
    load probe from conf

    Parameters
    ----------
    conf : dict
    apply_probe_location: bool

    Returns
    -------
    Probe

    """
    # load from probe library
    if 'probe_key' in conf and 'probe' in conf:
        raise config.InvalidConf("'probe' and 'probe_key' mutually exclusive")
    if 'probe_key' in conf:
        probe = _probes.probes[conf['probe_key']]
    else:
        probe = core.Probe.make_matrix_probe(**conf['probe'])

    if apply_probe_location:
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


def examination_object_from_conf(conf):
    """
    Load examination object

    Parameters
    ----------
    conf : dict

    Returns
    -------
    arim.core.ExaminationObject

    """
    if ('frontwall' in conf.keys() and
            'backwall' in conf.keys() and
            'couplant_material' in conf.keys() and
            'block_material' in conf.keys()):
        return block_in_immersion_from_conf(conf)
    else:
        raise NotImplementedError


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
