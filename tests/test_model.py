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
        arim.path.points_1d_wall_z(numpoints=1000, xmin=-5.e-3, xmax=50.e-3, z=30.e-3,
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
        'LL': [[[200, 273]], [[91, 780]]],
        'LT': [[[200, 278]], [[91, 918]]],
        'TL': [[[200, 291]], [[91, 424]]],
        'TT': [[[200, 353]], [[91, 789]]],
    }
    expected_ray_times = {
        'L': array([[9.92131466e-06, 1.51169800e-05]]),
        'T': array([[1.31465342e-05, 2.32862381e-05]]),
        'LL': array([[1.30858724e-05, 1.67760353e-05]]),
        'LT': array([[1.46984829e-05, 1.87550216e-05]]),
        'TL': array([[1.79237015e-05, 2.30552458e-05]]),
        'TT': array([[1.95363121e-05, 2.67521168e-05]]),
    }
    for pathname, path in paths.items():
        rays = arim.im.Rays(np.asarray(expected_ray_times[pathname]),
                            np.asarray(expected_ray_indices[pathname], np.uint32),
                            path.to_fermat_path())
        path.rays = rays

    # Ray geometry
    ray_geometry_dict = OrderedDict((k, arim.model.RayGeometry.from_path(v))
                                    for (k, v) in paths.items())

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
    """:type : dict[str, arim.Path]"""
    views = context['views']
    """:type : dict[str, arim.View]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    expected_rays = OrderedDict()

    # Actually perform ray-tracing:
    for pathname, path in paths.items():
        expected_rays[path.name] = path.rays
        path.rays = None
    arim.im.ray_tracing(views.values())

    # check all intersection points (as indices)
    for pathname, path in paths.items():
        # print("'{}': {},".format(pathname, path.rays.interior_indices.tolist()))
        np.testing.assert_equal(path.rays.indices, expected_rays[pathname].indices)

    # check times (as indices)
    for pathname, path in paths.items():
        # print("'{}': {},".format(pathname, repr(path.rays.times)))
        np.testing.assert_allclose(path.rays.times, expected_rays[pathname].times)

    tol = dict(atol=0., rtol=1e-5)

    # check intersection points on frontwall (as coordinates)
    expected_frontwall_points = {
        'L': array([[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                     [2.20720721e-03, 0.00000000e+00, 0.00000000e+00]]]),
        'T': array([[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                     [4.78478478e-03, 0.00000000e+00, 0.00000000e+00]]]),
        'LL': array([[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                      [1.83183183e-03, 0.00000000e+00, 0.00000000e+00]]]),
        'LT': array([[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                      [1.95695696e-03, 0.00000000e+00, 0.00000000e+00]]]),
        'TL': array([[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                      [2.28228228e-03, 0.00000000e+00, 0.00000000e+00]]]),
        'TT': array([[[5.00500501e-06, 0.00000000e+00, 0.00000000e+00],
                      [3.83383383e-03, 0.00000000e+00, 0.00000000e+00]]]),
    }

    for pathname, path in paths.items():
        idx = 1
        coords = interfaces[idx].points.coords.take(path.rays.indices[idx], axis=0)
        # print("'{}': {},".format(pathname, repr(coords)))
        np.testing.assert_allclose(coords, expected_frontwall_points[pathname], **tol)

    # check intersection points on backwall (as coordinates)
    expected_backwall_points = {
        'L': None,
        'T': None,
        'LL': array([[[1.00100100e-05, 0.00000000e+00, 3.00000000e-02],
                      [3.79429429e-02, 0.00000000e+00, 3.00000000e-02]]]),
        'LT': array([[[1.00100100e-05, 0.00000000e+00, 3.00000000e-02],
                      [4.55405405e-02, 0.00000000e+00, 3.00000000e-02]]]),
        'TL': array([[[1.00100100e-05, 0.00000000e+00, 3.00000000e-02],
                      [1.83433433e-02, 0.00000000e+00, 3.00000000e-02]]]),
        'TT': array([[[1.00100100e-05, 0.00000000e+00, 3.00000000e-02],
                      [3.84384384e-02, 0.00000000e+00, 3.00000000e-02]]]),
    }
    for pathname, path in paths.items():
        if expected_backwall_points[pathname] is not None:
            idx = 2
            coords = interfaces[idx].points.coords.take(path.rays.indices[idx], axis=0)
            # print("'{}': {},".format(pathname, repr(coords)))
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
                                   atol=1e-2)

    # check the leg sizes
    for pathname, ray_geometry in ray_geometry_dict.items():
        first_leg_size = ray_geometry.inc_leg_size(1)
        second_leg_size = ray_geometry.inc_leg_size(2)
        np.testing.assert_allclose(first_leg_size[0][0], 10e-3, rtol=1e-5)
        if pathname in {'L', 'T'}:
            np.testing.assert_allclose(second_leg_size[0][0], 20e-3, rtol=1e-5)
        else:
            third_leg_size = ray_geometry.inc_leg_size(3)
            np.testing.assert_allclose(second_leg_size[0][0], 30e-3, rtol=1e-5)
            np.testing.assert_allclose(third_leg_size[0][0], 10e-3, rtol=1e-5)


def test_beamspread_2d_direct():
    context = make_context()
    block = context['block']
    """:type : arim.Material"""
    couplant = context['couplant']
    """:type : arim.Material"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    # hardcoded results
    expected_beamspread = {
        'L': array([[3.2375215, 0.84817938]]),
        'T': array([[4.37280604, 1.38511721]]),
        'LL': array([[2.33034458, 1.24259583]]),
        'LT': array([[2.49293429, 1.19394661]]),
        'TL': array([[2.85272935, 0.75913334]]),
        'TT': array([[3.19555665, 1.8960812]]),
    }
    beamspread = dict()
    for pathname, ray_geometry in ray_geometry_dict.items():
        beamspread[pathname] = arim.model.beamspread_2d_for_path(ray_geometry)
        np.testing.assert_allclose(beamspread[pathname], expected_beamspread[pathname])
        # Uncomment the following line to generate hardcoded-values:
        # (use -s flag in pytest to show output)
        # print("'{}': {},".format(pathname, repr(beamspread[pathname])))

    # Path L - scat 0:
    first_leg = 10e-3
    second_leg = 20e-3
    c1 = couplant.longitudinal_vel
    c2 = block.longitudinal_vel
    beta = c1 / c2
    beamspread_L = np.sqrt(1 / (first_leg + second_leg / beta))
    np.testing.assert_allclose(beamspread['L'][0][0], beamspread_L, rtol=1e-5)

    # Path LL - scat 0:
    first_leg = 10e-3
    second_leg = 30e-3
    third_leg = 10.e-3
    c1 = couplant.longitudinal_vel
    c2 = block.longitudinal_vel
    beta = c1 / c2
    beamspread_LL = 1. / np.sqrt(((first_leg + second_leg / beta) *
                                  (1. + third_leg / second_leg)))
    np.testing.assert_allclose(beamspread['LL'][0][0], beamspread_LL, rtol=1e-5)


def test_beamspread_2d_reverse():
    context = make_context()
    block = context['block']
    """:type : arim.Material"""
    couplant = context['couplant']
    """:type : arim.Material"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    # hardcoded results
    expected_beamspread = {
        'L': array([[6.69023357, 4.37716144]]),
        'T': array([[6.35918872, 4.44926218]]),
        'LL': array([[4.81558177, 3.95438967]]),
        'LT': array([[3.62537617, 1.84938364]]),
        'TL': array([[5.89506311, 5.04457277]]),
        'TT': array([[4.64716424, 3.94081983]]),
    }
    beamspread = dict()
    for pathname, ray_geometry in ray_geometry_dict.items():
        beamspread[pathname] = arim.model.beamspread_2d_for_reversed_path(ray_geometry)
        np.testing.assert_allclose(beamspread[pathname], expected_beamspread[pathname])
        # Uncomment the following line to generate hardcoded-values:
        # (use -s flag in pytest to show output)
        # print("'{}': {},".format(pathname, repr(beamspread[pathname])))

    # Reversed path L - scat 0:
    first_leg = 20e-3
    second_leg = 10e-3
    c1 = block.longitudinal_vel
    c2 = couplant.longitudinal_vel
    beta = c1 / c2
    beamspread_L = np.sqrt(1 / (first_leg + second_leg / beta))
    np.testing.assert_allclose(beamspread['L'][0][0], beamspread_L, rtol=1e-5)

    # Path LL - scat 0:
    first_leg = 10e-3
    second_leg = 30e-3
    third_leg = 10.e-3
    c1 = block.longitudinal_vel
    c2 = couplant.longitudinal_vel
    beta = c1 / c2
    beamspread_LL = 1. / np.sqrt(((first_leg + second_leg) *
                                  (1. + third_leg / (beta * second_leg))))
    np.testing.assert_allclose(beamspread['LL'][0][0], beamspread_LL, rtol=1e-5)


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
        'LL': array([[-2.18993849 + 0.j, -5.11477179 + 0.j]]),
        'LT': array([[-3.29842870e-04 + 0.j, -2.72427011e+00 + 0.j]]),
        'TL': array([[-1.18487478e-06 + 0.j, -2.69978593e+00 + 0.j]]),
        'TT': array([[1.92953113e-03 - 0.j, -2.31008884e+00 - 0.05936071j]]),
    }
    transrefl = dict()
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        transrefl[pathname] = arim.model.transmission_reflection_for_path(path,
                                                                          ray_geometry)
        np.testing.assert_allclose(transrefl[pathname], expected_transrefl[pathname],
                                   rtol=0., atol=1e-6)
        # print("'{}': {},".format(pathname, repr(transrefl[pathname])))


def test_radiation_2d_rectangular_in_fluid():
    # arim.model.radiation_2d_rectangular_in_fluid_for_path()
    context = make_context()
    couplant = context['couplant']
    paths = context['paths']
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context['ray_geometry_dict']
    """:type : dict[str, arim.path.RayGeometry]"""

    wavelength = couplant.longitudinal_vel / context['freq']

    # hardcoded results
    expected_radiation = {
        'L': array([[0.47777476 + 0.47777476j, 0.46128071 + 0.46128071j]]),
        'T': array([[0.47777476 + 0.47777476j, 0.41368391 + 0.41368391j]]),
        'LL': array([[0.47777476 + 0.47777476j, 0.46621085 + 0.46621085j]]),
        'LT': array([[0.47777476 + 0.47777476j, 0.46465044 + 0.46465044j]]),
        'TL': array([[0.47777476 + 0.47777476j, 0.46020818 + 0.46020818j]]),
        'TT': array([[0.47777476 + 0.47777476j, 0.43310535 + 0.43310535j]]),
    }
    radiation = dict()
    print()
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        radiation[pathname] = arim.model.radiation_2d_rectangular_in_fluid_for_path(
            ray_geometry, context['element_width'], wavelength)
        np.testing.assert_allclose(radiation[pathname], expected_radiation[pathname],
                                   rtol=0., atol=1e-6)
        # Uncomment the following line to generate hardcoded-values:
        # (use -s flag in pytest to show output)
        # print("'{}': {},".format(pathname, repr(radiation[pathname])))
