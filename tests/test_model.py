"""
Hard-code results and hope they do not evolve over time.

"""
import pytest
import arim
from collections import OrderedDict
import numpy as np
from numpy import array


def make_context():
    """
    2D case, immersion

    dimensions in mm

    probe (1 element): x=0, z=-10

    frontwall: z=0
    backwall: z=30

    scatterers:
        x=0, z=20
        x=50, z=20

    """
    couplant = arim.Material(longitudinal_vel=1480., density=1000.,
                             state_of_matter='liquid')
    block = arim.Material(longitudinal_vel=6320., transverse_vel=3130., density=2700.,
                          state_of_matter='solid')

    probe_points = arim.Points([[0., 0., -10e-3]], 'Probe')
    probe_orientations = arim.path.default_orientations(probe_points)

    frontwall_points, frontwall_orientations \
        = arim.path.points_1d_wall_z(numpoints=1000, xmin=-5.e-3, xmax=20.e-3,
                                     z=0., name='Frontwall')
    backwall_points, backwall_orientations = \
        arim.path.points_1d_wall_z(numpoints=1000, xmin=-5.e-3, xmax=50.e-3, z=30 - 3,
                                   name='Backwall')

    scatterer_points = arim.Points([[0., 0., 20e-3], [50e-3, 0., 20e-3]],
                                   'Scatterers')
    scatterer_orientations = arim.path.default_orientations(scatterer_points)

    interfaces = arim.path.interfaces_for_block_in_immersion(couplant, probe_points,
                                                             probe_orientations,
                                                             frontwall_points,
                                                             frontwall_orientations,
                                                             backwall_points,
                                                             backwall_orientations,
                                                             scatterer_points,
                                                             scatterer_orientations)

    paths = arim.path.paths_for_block_in_immersion(block, couplant, *interfaces)
    views = arim.path.views_for_block_in_immersion(paths)

    # Do the ray tracing manually
    # Hardcode the result of ray-tracing in order to write tests with lower coupling
    expected_ray_indices = {
        'L': [[[200, 288]]],
        'T': [[[200, 391]]],
        'LL': [[[200, 200]], [[91, 545]]],
        'LT': [[[200, 200]], [[91, 698]]],
        'TL': [[[200, 200]], [[91, 392]]],
        'TT': [[[200, 200]], [[91, 545]]],
    }
    expected_ray_times = {
        'L': [[9.921314664158826e-06, 1.5116980035762513e-05]],
        'T': [[1.314653416095097e-05, 2.3286238145058415e-05]],
        'LL': [[0.008547895998109739, 0.00854789966141407]],
        'LT': [[0.012898716963047152, 0.01289872186234908]],
        'TL': [[0.012901942182442729, 0.012901947082967543]],
        'TT': [[0.01725276314738014, 0.01725277054421189]],
    }
    for pathname, path in paths.items():
        rays = arim.im.Rays(np.asarray(expected_ray_times[pathname]),
                            np.asarray(expected_ray_indices[pathname], np.uint32),
                            path.to_fermat_path())
        path.rays = rays

    # ray tracing
    ray_geometry_dict = OrderedDict([(k, arim.model.RayGeometry.from_path(v))
                                     for (k, v) in paths.items()])

    context = dict()
    context['block'] = block
    context['couplant'] = couplant
    context['interfaces'] = interfaces
    context['paths'] = paths
    context['views'] = views
    context['ray_geometry_dict'] = ray_geometry_dict
    context['probe_points'] = probe_points
    context['probe_orientations'] = probe_orientations
    context['frontwall_points'] = frontwall_points
    context['frontwall_orientations'] = frontwall_orientations
    context['backwall_points'] = backwall_points
    context['backwall_orientations'] = backwall_orientations
    context['scatterer_points'] = scatterer_points
    context['scatterer_orientations'] = scatterer_orientations
    context['freq'] = 2e6
    context['element_width'] = 0.5e-3

    '''==================== copy/paste me ====================
    context = make_context()
    block = context['block']
    """:type : arim.Material"""
    couplant = context['couplant']
    """:type : arim.Material"""
    interfaces = context['interfaces']
    """:type : list[arim.Interface]"""
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
    '''

    return context


def test_context():
    context = make_context()
    block = context['block']
    couplant = context['couplant']
    interfaces = context['interfaces']
    paths = context['paths']
    views = context['views']
    ray_geometry_dict = context['ray_geometry_dict']
    probe_points = context['probe_points']
    probe_orientations = context['probe_orientations']
    frontwall_points = context['frontwall_points']
    frontwall_orientations = context['frontwall_orientations']
    backwall_points = context['backwall_points']
    backwall_orientations = context['backwall_orientations']
    scatterer_points = context['scatterer_points']
    scatterer_orientations = context['scatterer_orientations']

    assert interfaces[0].points is probe_points
    assert interfaces[0].orientations is probe_orientations
    assert interfaces[1].points is frontwall_points
    assert interfaces[1].orientations is frontwall_orientations
    assert interfaces[2].points is backwall_points
    assert interfaces[2].orientations is backwall_orientations
    assert interfaces[3].points is scatterer_points
    assert interfaces[3].orientations is scatterer_orientations

    assert paths.keys() == {'L', 'LL', 'LT', 'TL', 'TT', 'T'}
    for path in paths.values():
        assert path.rays is not None

    for pathname, path in paths.items():
        assert path.name == pathname

    for viewname, view in views.items():
        assert view.name == viewname


def test_ray_tracing():
    context = make_context()
    interfaces = context['interfaces']
    """:type : list[arim.Interface]"""
    paths = context['paths']
    """:type : list[arim.Path]"""
    views = context['views']
    """:type : dict[str, arim.View]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    # Actually perform ray-tracing:
    for path in paths.values():
        path.rays = None
    arim.im.ray_tracing(views.values())

    # check all intersection points (as indices)
    expected_indices = {'L': [[[0, 0]], [[200, 288]], [[0, 1]]],
                        'T': [[[0, 0]], [[200, 391]], [[0, 1]]],
                        'LL': [[[0, 0]], [[200, 200]], [[91, 545]], [[0, 1]]],
                        'LT': [[[0, 0]], [[200, 200]], [[91, 698]], [[0, 1]]],
                        'TL': [[[0, 0]], [[200, 200]], [[91, 392]], [[0, 1]]],
                        'TT': [[[0, 0]], [[200, 200]], [[91, 545]], [[0, 1]]], }
    for pathname, path in paths.items():
        np.testing.assert_equal(path.rays.indices, expected_indices[pathname])

    tol = dict(atol=0., rtol=1e-5)

    # check intersection points on frontwall (as coordinates)
    expected_frontwall_points = {
        'L': [[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
               [2.20720721e-03, 0.00000000e+00, 0.00000000e+00]]],
        'T': [[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
               [4.78478478e-03, 0.00000000e+00, 0.00000000e+00]]],
        'LL': [[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                [5.00500501e-06, 0.00000000e+00, 0.00000000e+00]]],
        'LT': [[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                [5.00500501e-06, 0.00000000e+00, 0.00000000e+00]]],
        'TL': [[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                [5.00500501e-06, 0.00000000e+00, 0.00000000e+00]]],
        'TT': [[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                [5.00500501e-06, 0.00000000e+00, 0.00000000e+00]]], }

    for pathname, path in paths.items():
        idx = 1
        coords = interfaces[idx].points.coords.take(path.rays.indices[idx], axis=0)
        np.testing.assert_allclose(coords, expected_frontwall_points[pathname], **tol)

    # check intersection points on backwall (as coordinates)
    expected_backwall_points = {
        'L': None,
        'T': None,
        'LL': [[[1.00100100e-05, 0.00000000e+00, 2.70000000e+01],
                [2.50050050e-02, 0.00000000e+00, 2.70000000e+01]]],
        'LT': [[[1.00100100e-05, 0.00000000e+00, 2.70000000e+01],
                [3.34284284e-02, 0.00000000e+00, 2.70000000e+01]]],
        'TL': [[[1.00100100e-05, 0.00000000e+00, 2.70000000e+01],
                [1.65815816e-02, 0.00000000e+00, 2.70000000e+01]]],
        'TT': [[[1.00100100e-05, 0.00000000e+00, 2.70000000e+01],
                [2.50050050e-02, 0.00000000e+00, 2.70000000e+01]]]
    }
    for pathname, path in paths.items():
        if expected_backwall_points[pathname] is not None:
            idx = 2
            coords = interfaces[idx].points.coords.take(path.rays.indices[idx], axis=0)
            np.testing.assert_allclose(coords, expected_backwall_points[pathname], **tol)

    # check Snell-laws for frontwall
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]
        idx = 1
        incident_angles = ray_geometry.conventional_inc_angle(idx)
        refracted_angles = ray_geometry.conventional_out_angle(idx)

        # The first scatterer is just below the transmitter so the angles should be 0.
        # poor precision because of discretisation of interface
        # TODO: better precision for ray tracing?
        assert np.isclose(incident_angles[0][0], 0., atol=1e-3, rtol=0.)
        assert np.isclose(refracted_angles[0][0], 0., atol=1e-3, rtol=0.)

        c_incident = path.velocities[idx - 1]
        c_refracted = path.velocities[idx]
        expected_refracted_angles = arim.ut.snell_angles(incident_angles, c_incident,
                                                         c_refracted)
        np.testing.assert_allclose(refracted_angles, expected_refracted_angles, rtol=0.,
                                   atol=1e-2)

    # check Snell-laws for backwall (half-skip only)
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]
        if len(paths[pathname].interfaces) <= 3:
            continue

        idx = 2
        incident_angles = ray_geometry.conventional_inc_angle(idx)
        refracted_angles = ray_geometry.conventional_out_angle(idx)

        c_incident = path.velocities[idx - 1]
        c_refracted = path.velocities[idx]
        expected_refracted_angles = arim.ut.snell_angles(incident_angles, c_incident,
                                                         c_refracted)
        np.testing.assert_allclose(refracted_angles, expected_refracted_angles, rtol=0.,
                                   atol=1e-3)

    # check the leg sizes
    for pathname, ray_geometry in ray_geometry_dict.items():
        first_leg_size = ray_geometry.inc_leg_size(1)
        np.testing.assert_allclose(first_leg_size[0][0], 10e-3, rtol=1e-5)


def test_beamspread_direct():
    context = make_context()
    block = context['block']
    """:type : arim.Material"""
    couplant = context['couplant']
    """:type : arim.Material"""
    paths = context['paths']
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""
    frontwall_points = context['frontwall_points']
    frontwall_orientations = context['frontwall_orientations']
    backwall_points = context['backwall_points']
    backwall_orientations = context['backwall_orientations']
    scatterer_points = context['scatterer_points']
    scatterer_orientations = context['scatterer_orientations']

    # hardcoded results
    expected_beamspread = {
        'L': array([[3.2375215, 0.84817938]]),
        'T': array([[4.37280604, 1.38511721]]),
        'LL': array([[0.06586218, 0.06586217]]),
        'LT': array([[0.07616695, 0.07616695]]),
        'TL': array([[0.07617325, 0.07617319]]),
        'TT': array([[0.09358452, 0.0935845]]),
    }
    beamspread = dict()
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        beamspread[pathname] = arim.model.beamspread_for_path_snell(path, ray_geometry)
        np.testing.assert_allclose(beamspread[pathname], expected_beamspread[pathname])

    # For the case L - scat 0:
    first_leg = 10e-3
    second_leg = 20e-3
    c1 = couplant.longitudinal_vel
    c2 = block.longitudinal_vel
    beta = c1 / c2
    beamspread_L = np.sqrt(1 / (first_leg + second_leg / beta))
    np.testing.assert_allclose(beamspread['L'][0][0], beamspread_L, rtol=1e-5)


def test_transmission_reflection_direct():
    context = make_context()
    paths = context['paths']
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    # hardcoded results
    expected_transrefl = {
        'L': array([[1.84037966 + 0.j, 2.24977557 + 0.j]]),
        'T': array([[-1.92953093e-03 + 0.j, -2.06170606e+00 + 0.76785004j]]),
        'LL': array([[-2.18993843 + 0.j, -2.18994043 + 0.j]]),
        'LT': array([[-3.65598421e-07 + 0.j, -2.44743435e-03 + 0.j]]),
        'TL': array([[-1.31331720e-09 + 0.j, -4.36034322e-06 + 0.j]]),
        'TT': array([[0.00192953 - 0.j, 0.00192954 - 0.j]]),
    }
    transrefl = dict()
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        transrefl[pathname] = arim.model.transmission_reflection_for_path(path,
                                                                          ray_geometry)
        np.testing.assert_allclose(transrefl[pathname], expected_transrefl[pathname],
                                   rtol=0., atol=1e-6)


def test_radiation_2d_rectangular_in_fluid():
    # arim.model.radiation_2d_rectangular_in_fluid_for_path()
    context = make_context()
    couplant = context['couplant']
    paths = context['paths']
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    impedance = couplant.longitudinal_vel * couplant.density
    wavelength = context['freq'] / couplant.longitudinal_vel

    # hardcoded results
    expected_radiation = {
        'L': array([[14.23418421 + 14.23418421j, 14.23418421 + 14.23418421j]]),
        'T': array([[14.23418421 + 14.23418421j, 14.23418421 + 14.23418421j]]),
        'LL': array([[14.23418421 + 14.23418421j, 14.23418421 + 14.23418421j]]),
        'LT': array([[14.23418421 + 14.23418421j, 14.23418421 + 14.23418421j]]),
        'TL': array([[14.23418421 + 14.23418421j, 14.23418421 + 14.23418421j]]),
        'TT': array([[14.23418421 + 14.23418421j, 14.23418421 + 14.23418421j]]),
    }
    radiation = dict()
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        radiation[pathname] = arim.model.radiation_2d_rectangular_in_fluid_for_path(
            ray_geometry, context['element_width'], wavelength, impedance)
        np.testing.assert_allclose(radiation[pathname], expected_radiation[pathname],
                                   rtol=0., atol=1e-6)
        # Uncomment the following line to generate hardcoded-values:
        # (use -s flag in pytest to show output)
        # print("'{}': {},".format(pathname, repr(radiation[pathname])))
