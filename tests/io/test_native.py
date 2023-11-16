import arim.io

PROBE_1 = """
probe:
  frequency: 5.e6
  numx: 110
  pitch_x: 0.17e-3
  numy: 1
  pitch_y: .nan
  dimensions: [0.17e-3, 15.e-3, .nan]
  metadata:
    short_name:
    long_name: Modelled
    probe_type: linear
"""

PROBE_2 = """
probe_key: ima_50_MHz_128_1d
"""

PROBE_LOC = """
probe_location:
  ref_element: first
  standoff: -30.e-3 # m
  angle_deg: 12.5 # degrees
"""

ATT_1 = "longitudinal_att: 777."

ATT_2 = """
longitudinal_att:
    kind: constant
    value: 777.
"""


BLOCK_IN_IMMERSION = """
couplant_material:
  metadata:
    long_name: Water
  longitudinal_vel: 1480.
  density: 1000.
  longitudinal_att: 0.1
  state_of_matter: liquid

block_material:
  metadata:
    long_name: Aluminium
    source: Krautkrämer 1990
  longitudinal_vel: 6320.
  transverse_vel: 3130.
  density: 2700.
  state_of_matter: solid

frontwall:
  numpoints: 1000
  xmin: 0.e-3
  xmax: 100.e-3
  z: 0.

backwall:
  numpoints: 1000
  xmin: 0.e-3
  xmax: 100.e-3
  z: 20.e-3
"""

BLOCK_IN_CONTACT = """
block_material:
  metadata:
    long_name: Aluminium
    source: Krautkrämer 1990
  longitudinal_vel: 6320.
  transverse_vel: 3130.
  density: 2700.
  state_of_matter: solid

# If imaging with internal reflection of the frontwall:
frontwall:
  numpoints: 1000
  xmin: -42.e-3
  xmax: 42.e-3
  z: 0.

# If imaging with internal reflection of the backwall:
backwall:
  numpoints: 1000
  xmin: -42.e-3
  xmax: 42.e-3
  z: 40.e-3
"""


def test_probe_from_conf():
    for conf_str in (PROBE_1, PROBE_2):
        conf_str += PROBE_LOC
        conf = arim.io.load_conf_from_str(conf_str)
        probe = arim.io.probe_from_conf(conf)
        assert isinstance(probe, arim.Probe)


def test_block_in_immersion_from_conf():
    conf = arim.io.load_conf_from_str(BLOCK_IN_IMMERSION)
    examination_object = arim.io.block_in_immersion_from_conf(conf)
    assert isinstance(examination_object, arim.BlockInImmersion)


def test_examination_object_from_conf():
    conf = arim.io.load_conf_from_str(BLOCK_IN_IMMERSION)
    examination_object = arim.io.examination_object_from_conf(conf)
    assert isinstance(examination_object, arim.BlockInImmersion)

    conf = arim.io.load_conf_from_str(BLOCK_IN_CONTACT)
    examination_object = arim.io.examination_object_from_conf(conf)
    assert isinstance(examination_object, arim.BlockInContact)


def test_material_attenuation_from_conf():
    for conf_str in (ATT_1, ATT_2):
        conf = arim.io.load_conf_from_str(conf_str)
        att = arim.io.material_attenuation_from_conf(conf["longitudinal_att"])
        assert att(10) == 777.0
