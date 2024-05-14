"""
Load .arim file format

An .arim file is a directory which contains:

- a conf.yaml file (base configuration file)
- a conf.d directory which contains additional configuration files (optional)
- intermediary and final results (optional).

The recommended way to load the configuration is to use :func:`load_conf`.
The configuration is loaded according to the following pseudo-code:

.. code-block:: none

    conf := read(conf.yaml)
    For each file in conf.d:
        tmp_conf := read(file)
        conf := merge(conf, tmp_conf)
    Return conf

The files are conf.d are read in alphabetical order.
If a configuration entry is present in two files, only the entry from the file
read the latest will be kept.


conf.yaml format
----------------

The ultrasonic data is provided using either:

1. ``frame.datafile``: path to the data file, either absolute or relative to
   ``conf.yaml``,
2. ``frame.dataset_name`` and ``frame.dataset_item``: a dataset name
   (``arim.datasets``) and the path of the item to fetch


"""

import copy
import logging
import pathlib

import numpy as np
import yaml

from .. import _probes, config, core, datasets, geometry
from . import brain

LOGGER = logger = logging.getLogger(__name__)

__all__ = [
    "load_conf",
    "load_conf_file",
    "load_conf_from_str",
    "probe_from_conf",
    "examination_object_from_conf",
    "block_in_immersion_from_conf",
    "block_in_contact_from_conf",
    "material_from_conf",
    "material_attenuation_from_conf",
    "grid_from_conf",
    "frame_from_conf",
]


class InvalidConf(Exception):
    pass


def load_conf_from_str(stream):
    """Load a single configuration file from a stream or string formatted in YAML.

    Parameters
    ----------
    stream : stream, str

    Returns
    -------
    arim.config.Config
    """
    return config.Config(yaml.safe_load(stream))


def load_conf_file(filename):
    """Load a single configuration file

    Parameters
    ----------
    filename : str
        Filename

    Returns
    -------
    arim.config.Config
    """
    with open(filename) as f:
        return load_conf_from_str(f)


def load_conf(dirname, filepath_keys={"filename", "datafile"}):
    """
    Load the configuration from a `.arim` directory

    Parameters
    ----------
    dirname
    filepath_keys : set
        Config keys that stores files. If they are relative paths, they will be replaced
        by an absolute path (str object) assuming the root dir is the `.arim` directory.
        Set to False to disable.

    Returns
    -------
    arim.config.Config

    Notes
    -----
    Load {dirname}/conf.yaml and all yaml files in {dirname}/conf.d/.
    """
    root_dir = pathlib.Path(dirname).resolve(strict=True)

    # format: foo/bar/{dataset_name}.arim
    dataset_name = root_dir.parts[-1]
    if dataset_name.endswith(".arim"):
        dataset_name = dataset_name[:-5]

    # Load root conf
    root_conf_filename = root_dir / "conf.yaml"
    if root_conf_filename.exists():
        conf = load_conf_file(root_conf_filename)
    else:
        conf = config.Config({})

    # Load conf.d fragment files
    # Remark: no error if root_dir/conf.d doesn't exist
    for conf_filename in root_dir.glob("conf.d/*.yaml"):
        conf.merge(load_conf_file(conf_filename))

    # Populate extra keys
    conf["dataset_name"] = dataset_name
    conf["root_dir"] = root_dir

    # populate result_dir if missing, create dir if needed
    result_dir = conf.get("result_dir", None)
    if result_dir is None:
        result_dir = root_dir
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = pathlib.Path(root_dir / pathlib.Path(result_dir)).resolve(
            strict=True
        )
    conf["result_dir"] = result_dir

    if filepath_keys:
        _resolve_filenames(conf, root_dir, filepath_keys)

    return conf


def _resolve_filenames(d, root_dir, target_keys):
    """
    Replace target keys by an absolute pathlib.Path where the root dir
    is `root_dir`

    Parameters
    ----------
    d : dict or anything
    root_dir : pathlib.Path
    target_keys : set

    Returns
    -------
    d
        Updated dictionary

    """
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        if k in target_keys:
            d[k] = str(root_dir / v)
        else:
            _resolve_filenames(v, root_dir, target_keys)


def probe_from_conf(conf, apply_probe_location=True):
    """
    Load probe

    Parameters
    ----------
    conf : dict
        Root conf
    apply_probe_location: bool

    Returns
    -------
    Probe

    """
    # load from probe library
    if "probe_key" in conf and "probe" in conf:
        raise config.InvalidConf("'probe' and 'probe_key' mutually exclusive")
    if "probe_key" in conf:
        probe = _probes.probes[conf["probe_key"]]
    else:
        probe = core.Probe.make_matrix_probe(**conf["probe"])

    if apply_probe_location:
        probe_location = conf["probe_location"]

        if "ref_element" in probe_location:
            probe.set_reference_element(conf["probe_location"]["ref_element"])
            probe.translate_to_point_O()

        if "angle_deg" in probe_location:
            probe.rotate(
                geometry.rotation_matrix_y(
                    np.deg2rad(conf["probe_location"]["angle_deg"])
                )
            )

        if "standoff" in probe_location:
            probe.translate([0, 0, conf["probe_location"]["standoff"]])

    return probe


def examination_object_from_conf(conf):
    """
    Load examination object

    Parameters
    ----------
    conf : dict
        Root conf

    Returns
    -------
    arim.core.ExaminationObject

    """
    if ((
            ("frontwall" in conf.keys() and "backwall" in conf.keys())
            or "contiguous_geometry" in conf.keys()
        )
        and "couplant_material" in conf.keys()
        and "block_material" in conf.keys()
    ):
        return block_in_immersion_from_conf(conf)
    elif "block_material" in conf.keys():
        return block_in_contact_from_conf(conf)
    else:
        raise NotImplementedError


def material_attenuation_from_conf(mat_att_conf):
    """
    Load material attenuation

    Parameters
    ----------
    mat_att_conf : dict
        Material attenuation conf

    Returns
    -------
    func

    See Also
    --------
    :func:`arim.core.material_attenuation_factory`
    """
    if isinstance(mat_att_conf, float):
        return core.material_attenuation_factory("constant", mat_att_conf)
    else:
        # at this stage, assume we have a dict
        return core.material_attenuation_factory(**mat_att_conf)


def _material_from_conf(conf_or_none):
    if conf_or_none is None:
        return None
    else:
        return material_attenuation_from_conf(conf_or_none)


def material_from_conf(conf):
    """
    Load material

    Parameters
    ----------
    conf : dict
        Material conf

    Returns
    -------
    arim.core.Material
    """
    material_kwargs = copy.deepcopy(conf)
    material_kwargs["longitudinal_att"] = _material_from_conf(
        material_kwargs.get("longitudinal_att")
    )
    material_kwargs["transverse_att"] = _material_from_conf(
        material_kwargs.get("transverse_att")
    )
    return core.Material(**material_kwargs)


def block_in_immersion_from_conf(conf):
    """
    Load block in immersion (examination object)

    Parameters
    ----------
    conf : dict
        Root conf

    Returns
    -------
    arim.BlockInImmersion

    """
    couplant = material_from_conf(conf["couplant_material"])
    block = material_from_conf(conf["block_material"])
    # Initialise geometry storage
    walls, imaging = [], []
    
    # Simple geometry
    if "frontwall" in conf.keys() or "backwall" in conf.keys():
        frontwall_conf = conf.get("frontwall", None)
        backwall_conf  = conf.get("backwall", None)
        if backwall_conf is not None:
            walls.append(geometry.points_1d_wall_z(**backwall_conf, name="Backwall"))
            imaging.append(1)
        if frontwall_conf is not None:
            walls.append(geometry.points_1d_wall_z(**frontwall_conf, name="Frontwall"))
            imaging.append(0)
    
    # Contiguous (polygonal) geometry.
    # By convention, if not already defined then frontwall is first and move clockwise.
    if "contiguous_geometry" in conf.keys():
        geom_conf = conf["contiguous_geometry"]
        geom_coords = np.squeeze(geom_conf["coords"])
        geom_walls = geometry.make_contiguous_geometry(
            geom_coords,
            geom_conf["numpoints"],
            geom_conf["names"],
        )
        for wall in geom_walls:
            walls.append(wall)
        if (
            0 not in imaging
            or 0 not in geom_conf["wall_idxs"]
        ):
            imaging.append(wall)
    return core.BlockInImmersion(block, couplant, walls, imaging)


def block_in_contact_from_conf(conf):
    block = material_from_conf(conf["block_material"])
    # Initialise geometry storage
    walls, imaging = [], []
    
    # Simple geometry
    if "frontwall" in conf.keys() or "backwall" in conf.keys():
        frontwall_conf = conf.get("frontwall", None)
        backwall_conf  = conf.get("backwall", None)
        if frontwall_conf is not None:
            walls.append(geometry.points_1d_wall_z(**frontwall_conf, name="Frontwall"))
            imaging.append(0)
        if backwall_conf is not None:
            walls.append(geometry.points_1d_wall_z(**backwall_conf, name="Backwall"))
            imaging.append(1)
    
    # Polygonal geometry.
    # By convention, if not already defined then frontwall is first and move clockwise.
    if "contiguous_geometry" in conf.keys():
        geom_conf = conf["contiguous_geometry"]
        geom_coords = np.squeeze(geom_conf["coords"])
        geom_walls = geometry.make_contiguous_geometry(
            geom_coords,
            geom_conf["numpoints"],
            geom_conf["names"],
        )
        for wall in geom_walls:
            walls.append(wall)
        if (
            0 not in imaging
            or 0 not in geom_conf["wall_idxs"]
        ):
            imaging.append(wall)
    under_material_conf = conf.get("under_material", None)
    if under_material_conf is None:
        under_material = None
    else:
        under_material = material_from_conf(under_material_conf)
    return core.BlockInContact(block, walls, imaging, under_material)


def grid_from_conf(conf):
    """
    Load grid

    Parameters
    ----------
    conf : dict
        Root conf

    Returns
    -------
    arim.Grid

    """
    conf_grid = copy.deepcopy(conf["grid"])
    if "ymin" not in conf_grid:
        conf_grid["ymin"] = 0.0
    if "ymax" not in conf_grid:
        conf_grid["ymax"] = 0.0
    return geometry.Grid(**conf_grid)


def frame_from_conf(
    conf, use_probe_from_conf=True, use_examination_object_from_conf=True
):
    """
    Load a Frame.

    Current limitation: read only from Brain (relies on :func:`arim.io.brain.load_expdata`).


    Parameters
    ----------
    conf : dict or Conf
        Root configuration
    use_probe_from_conf : bool
        If True, load probe from conf (ignores the one defined in datafile)
    use_examination_object_from_conf : bool
        If True, load examination from conf (ignores the one defined in datafile)

    Returns
    -------
    Frame

    """
    frame_conf = conf["frame"]
    if "datafile" in frame_conf:
        # Load from absolute or relative path:
        fname = frame_conf["datafile"]
        if "dataset_name" in frame_conf:
            LOGGER.warning("ignoring frame.dataset_name")
        if "dataset_item" in frame_conf:
            LOGGER.warning("ignoring frame.dataset_item")
    else:
        # Load from arim.datasets:
        dataset_name = frame_conf["dataset_name"]
        try:
            dataset_pooch = datasets.DATASETS[dataset_name]
        except (KeyError, TypeError):
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. Valid values are: {tuple(datasets.DATASETS.keys())}"
            )
        fname = dataset_pooch.fetch(frame_conf["dataset_item"])

    frame = brain.load_expdata(fname)

    instrument_delay = None
    try:
        instrument_delay = frame_conf["instrument_delay"]
    except KeyError:
        pass

    if instrument_delay is not None:
        # Adjust time vector
        frame.time = core.Time(
            frame.time.start - instrument_delay, frame.time.step, len(frame.time)
        )

    if use_probe_from_conf:
        frame.probe = probe_from_conf(conf)
    if use_examination_object_from_conf:
        frame.examination_object = examination_object_from_conf(conf)

    return frame
