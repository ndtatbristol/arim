"""
Hard-code results and hope they do not evolve over time.

"""
from collections import OrderedDict

import numpy as np
import pytest
from numpy import array

import arim
import arim.model
import arim.models.block_in_immersion
import arim.ray
from arim import model


def make_context(couplant_att=None, block_l_att=None, block_t_att=None):
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
    couplant = arim.Material(
        longitudinal_vel=1480.0,
        density=1000.0,
        state_of_matter="liquid",
        longitudinal_att=couplant_att,
    )
    block = arim.Material(
        longitudinal_vel=6320.0,
        transverse_vel=3130.0,
        density=2700.0,
        state_of_matter="solid",
        longitudinal_att=block_l_att,
        transverse_att=block_t_att,
    )

    probe_points = arim.Points([[0.0, 0.0, -10e-3]], "Probe")
    probe_orientations = arim.geometry.default_orientations(probe_points)
    probe_oriented_points = arim.geometry.OrientedPoints(
        probe_points, probe_orientations
    )

    frontwall_oriented_points = arim.geometry.points_1d_wall_z(
        numpoints=1000, xmin=-5.0e-3, xmax=20.0e-3, z=0.0, name="Frontwall"
    )
    frontwall_points, frontwall_orientations = frontwall_oriented_points

    backwall_oriented_points = arim.geometry.points_1d_wall_z(
        numpoints=1000, xmin=-5.0e-3, xmax=50.0e-3, z=30.0e-3, name="Backwall"
    )
    backwall_points, backwall_orientations = backwall_oriented_points

    scatterer_points = arim.Points(
        [[0.0, 0.0, 20e-3], [50e-3, 0.0, 20e-3]], "Scatterers"
    )
    scatterer_orientations = arim.geometry.default_orientations(scatterer_points)
    scatterer_oriented_points = arim.geometry.OrientedPoints(
        scatterer_points, scatterer_orientations
    )

    interfaces = arim.models.block_in_immersion.make_interfaces(
        couplant,
        probe_oriented_points,
        frontwall_oriented_points,
        scatterer_oriented_points,
        [backwall_oriented_points],
    )

    paths = arim.models.block_in_immersion.make_paths(block, couplant, interfaces)
    views = arim.models.block_in_immersion.make_views_from_paths(paths)

    # Do the ray tracing manually
    # Hardcode the result of ray-tracing in order to write tests with lower coupling
    expected_ray_indices = {
        "L": [[[200, 288]]],
        "T": [[[200, 391]]],
        "LL": [[[200, 273]], [[91, 780]]],
        "LT": [[[200, 278]], [[91, 918]]],
        "TL": [[[200, 291]], [[91, 424]]],
        "TT": [[[200, 353]], [[91, 789]]],
    }
    expected_ray_times = {
        "L": array([[9.92131466e-06, 1.51169800e-05]]),
        "T": array([[1.31465342e-05, 2.32862381e-05]]),
        "LL": array([[1.30858724e-05, 1.67760353e-05]]),
        "LT": array([[1.46984829e-05, 1.87550216e-05]]),
        "TL": array([[1.79237015e-05, 2.30552458e-05]]),
        "TT": array([[1.95363121e-05, 2.67521168e-05]]),
    }
    for pathname, path in paths.items():
        rays = arim.ray.Rays(
            np.asarray(expected_ray_times[pathname]),
            np.asarray(expected_ray_indices[pathname], arim.settings.INT),
            path.to_fermat_path(),
        )
        path.rays = rays

    # Ray geometry
    ray_geometry_dict = OrderedDict(
        (k, arim.ray.RayGeometry.from_path(v, use_cache=True))
        for (k, v) in paths.items()
    )

    # Reverse paths
    rev_paths = OrderedDict([(key, path.reverse()) for (key, path) in paths.items()])
    rev_ray_geometry_dict = OrderedDict(
        (k, arim.ray.RayGeometry.from_path(v)) for (k, v) in rev_paths.items()
    )

    exam_obj = arim.BlockInImmersion(
        block,
        couplant,
        [(frontwall_points, frontwall_orientations), (backwall_points, backwall_orientations)],
        [1],
        (scatterer_points, scatterer_orientations),
    )

    context = dict()
    context["block"] = block
    context["couplant"] = couplant
    context["interfaces"] = interfaces
    context["paths"] = paths
    context["rev_paths"] = rev_paths
    context["views"] = views
    context["ray_geometry_dict"] = ray_geometry_dict
    context["rev_ray_geometry_dict"] = rev_ray_geometry_dict
    context["probe_oriented_points"] = probe_oriented_points
    context["frontwall_oriented_points"] = frontwall_oriented_points
    context["backwall_oriented_points"] = backwall_oriented_points
    context["scatterer_oriented_points"] = scatterer_oriented_points
    context["freq"] = 2e6
    context["element_width"] = 0.5e-3
    context["wavelength_in_couplant"] = couplant.longitudinal_vel / context["freq"]
    context["numelements"] = len(probe_points)
    context["numpoints"] = len(scatterer_points)
    context["exam_obj"] = exam_obj

    '''==================== copy/paste me ====================
    context = make_context()
    block = context['block']
    """:type : arim.Material"""
    couplant = context['couplant']
    """:type : arim.Material"""
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
    probe_oriented_points = context['probe_oriented_points']
    frontwall_oriented_points = context['frontwall_oriented_points']
    backwall_oriented_points = context['backwall_oriented_points']
    scatterer_oriented_points = context['scatterer_oriented_points']
    exam_obj = context['exam_obj']
    '''

    return context


def test_context():
    context = make_context()
    block = context["block"]
    couplant = context["couplant"]
    interfaces = context["interfaces"]
    paths = context["paths"]
    views = context["views"]
    ray_geometry_dict = context["ray_geometry_dict"]

    assert block is not None
    assert couplant is not None
    assert interfaces is not None
    assert ray_geometry_dict is not None

    assert paths.keys() == {"L", "LL", "LT", "TL", "TT", "T"}
    for path in paths.values():
        assert path.rays is not None

    for pathname, path in paths.items():
        assert path.name == pathname

    for viewname, view in views.items():
        assert view.name == viewname


def test_ray_tracing():
    context = make_context()
    interfaces = context["interfaces"]
    """:type : dict[str, arim.Interface]"""
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    rev_paths = context["rev_paths"]
    """:type : dict[str, arim.Path]"""
    views = context["views"]
    """:type : dict[str, arim.View]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    rev_ray_geometry_dict = context["rev_ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""

    assert rev_paths is not None

    expected_rays = OrderedDict()

    # Actually perform ray-tracing:
    for pathname, path in paths.items():
        expected_rays[path.name] = path.rays
        path.rays = None
    arim.ray.ray_tracing(views.values())

    # check all intersection points (as indices)
    for pathname, path in paths.items():
        # print("'{}': {},".format(pathname, path.rays.interior_indices.tolist()))
        np.testing.assert_equal(path.rays.indices, expected_rays[pathname].indices)

    # check times (as indices)
    for pathname, path in paths.items():
        # print("'{}': {},".format(pathname, repr(path.rays.times)))
        np.testing.assert_allclose(path.rays.times, expected_rays[pathname].times)

    tol = dict(atol=0.0, rtol=1e-5)

    # check intersection points on frontwall (as coordinates)
    expected_frontwall_points = {
        "L": array(
            [
                [
                    [5.00500501e-06, 0.00000000e00, 0.00000000e00],
                    [2.20720721e-03, 0.00000000e00, 0.00000000e00],
                ]
            ]
        ),
        "T": array(
            [
                [
                    [5.00500501e-06, 0.00000000e00, 0.00000000e00],
                    [4.78478478e-03, 0.00000000e00, 0.00000000e00],
                ]
            ]
        ),
        "LL": array(
            [
                [
                    [5.00500501e-06, 0.00000000e00, 0.00000000e00],
                    [1.83183183e-03, 0.00000000e00, 0.00000000e00],
                ]
            ]
        ),
        "LT": array(
            [
                [
                    [5.00500501e-06, 0.00000000e00, 0.00000000e00],
                    [1.95695696e-03, 0.00000000e00, 0.00000000e00],
                ]
            ]
        ),
        "TL": array(
            [
                [
                    [5.00500501e-06, 0.00000000e00, 0.00000000e00],
                    [2.28228228e-03, 0.00000000e00, 0.00000000e00],
                ]
            ]
        ),
        "TT": array(
            [
                [
                    [5.00500501e-06, 0.00000000e00, 0.00000000e00],
                    [3.83383383e-03, 0.00000000e00, 0.00000000e00],
                ]
            ]
        ),
    }

    for pathname, path in paths.items():
        idx = 1
        coords = interfaces["frontwall_trans"].points.coords.take(
            path.rays.indices[idx], axis=0
        )
        # print("'{}': {},".format(pathname, repr(coords)))
        np.testing.assert_allclose(coords, expected_frontwall_points[pathname], **tol)

    # check intersection points on backwall (as coordinates)
    expected_backwall_points = {
        "L": None,
        "T": None,
        "LL": array(
            [
                [
                    [1.00100100e-05, 0.00000000e00, 3.00000000e-02],
                    [3.79429429e-02, 0.00000000e00, 3.00000000e-02],
                ]
            ]
        ),
        "LT": array(
            [
                [
                    [1.00100100e-05, 0.00000000e00, 3.00000000e-02],
                    [4.55405405e-02, 0.00000000e00, 3.00000000e-02],
                ]
            ]
        ),
        "TL": array(
            [
                [
                    [1.00100100e-05, 0.00000000e00, 3.00000000e-02],
                    [1.83433433e-02, 0.00000000e00, 3.00000000e-02],
                ]
            ]
        ),
        "TT": array(
            [
                [
                    [1.00100100e-05, 0.00000000e00, 3.00000000e-02],
                    [3.84384384e-02, 0.00000000e00, 3.00000000e-02],
                ]
            ]
        ),
    }
    for pathname, path in paths.items():
        if expected_backwall_points[pathname] is not None:
            idx = 2
            coords = interfaces["backwall_refl"].points.coords.take(
                path.rays.indices[idx], axis=0
            )
            # print("'{}': {},".format(pathname, repr(coords)))
            np.testing.assert_allclose(
                coords, expected_backwall_points[pathname], **tol
            )

    # check Snell-laws for frontwall
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]
        idx = 1
        incident_angles = ray_geometry.conventional_inc_angle(idx)
        refracted_angles = ray_geometry.conventional_out_angle(idx)

        # The first scatterer is just below the transmitter so the angles should be 0.
        # poor precision because of discretisation of interface
        # TODO: better precision for ray tracing?
        assert np.isclose(incident_angles[0][0], 0.0, atol=1e-3, rtol=0.0)
        assert np.isclose(refracted_angles[0][0], 0.0, atol=1e-3, rtol=0.0)

        c_incident = path.velocities[idx - 1]
        c_refracted = path.velocities[idx]
        expected_refracted_angles = arim.model.snell_angles(
            incident_angles, c_incident, c_refracted
        )
        np.testing.assert_allclose(
            refracted_angles, expected_refracted_angles, rtol=0.0, atol=1e-2
        )

    # check Snell-laws for backwall (half-skip only)
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]
        if len(paths[pathname].interfaces) <= 3:
            continue

        idx = 2
        incident_angles = ray_geometry.conventional_inc_angle(idx)
        refracted_angles = ray_geometry.conventional_out_angle(idx)

        assert np.isclose(incident_angles[0][0], 0.0, atol=1e-2, rtol=0.0)
        assert np.isclose(refracted_angles[0][0], 0.0, atol=1e-2, rtol=0.0)

        c_incident = path.velocities[idx - 1]
        c_refracted = path.velocities[idx]
        expected_refracted_angles = arim.model.snell_angles(
            incident_angles, c_incident, c_refracted
        )
        np.testing.assert_allclose(
            refracted_angles, expected_refracted_angles, rtol=0.0, atol=1e-2
        )

    # check incident angles for scatterer
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]
        if len(paths[pathname].interfaces) <= 3:
            # direct paths
            incident_angles = ray_geometry.conventional_inc_angle(2)
            assert np.isclose(incident_angles[0][0], np.pi, atol=1e-3, rtol=0.0)
        else:
            # half-skip paths
            incident_angles = ray_geometry.conventional_inc_angle(3)
            assert np.isclose(incident_angles[0][0], 0.0, atol=1e-2, rtol=0.0)

    # check conventional angles are in the right quadrant
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]
        for i in range(path.numinterfaces):
            if i != 0:
                inc_angles = ray_geometry.conventional_inc_angle(i)
                assert np.all(inc_angles >= 0)
                assert np.all(inc_angles <= np.pi)
            if i != (path.numinterfaces - 1):
                out_angles = ray_geometry.conventional_out_angle(i)
                assert np.all(out_angles >= 0)
                assert np.all(out_angles <= np.pi)

    # check the leg sizes
    for pathname, ray_geometry in ray_geometry_dict.items():
        first_leg_size = ray_geometry.inc_leg_size(1)
        second_leg_size = ray_geometry.inc_leg_size(2)
        np.testing.assert_allclose(first_leg_size[0][0], 10e-3, rtol=1e-5)
        if pathname in {"L", "T"}:
            np.testing.assert_allclose(second_leg_size[0][0], 20e-3, rtol=1e-5)
        else:
            third_leg_size = ray_geometry.inc_leg_size(3)
            np.testing.assert_allclose(second_leg_size[0][0], 30e-3, rtol=1e-5)
            np.testing.assert_allclose(third_leg_size[0][0], 10e-3, rtol=1e-5)

    # Check consistency with reverse paths
    for pathname, ray_geometry in ray_geometry_dict.items():
        rev_ray_geometry = rev_ray_geometry_dict[pathname]
        numinterfaces = paths[pathname].numinterfaces

        for i in range(numinterfaces):
            j = numinterfaces - 1 - i

            inc_angles_1 = ray_geometry.conventional_inc_angle(i)
            inc_angles_2 = rev_ray_geometry.conventional_out_angle(j)
            if i == 0:
                assert inc_angles_1 is None
                assert inc_angles_2 is None
            else:
                assert inc_angles_1 is not None
                assert inc_angles_2 is not None
                np.testing.assert_allclose(inc_angles_1, inc_angles_2.T)

            out_angles_1 = ray_geometry.conventional_out_angle(i)
            out_angles_2 = rev_ray_geometry.conventional_inc_angle(j)
            if j == 0:
                assert out_angles_1 is None
                assert out_angles_2 is None
            else:
                assert out_angles_1 is not None
                assert out_angles_2 is not None
                np.testing.assert_allclose(out_angles_1, out_angles_2.T)


def test_caching():
    context = make_context()
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    paths = context["paths"]
    element_width = context["element_width"]
    wavelength = context["wavelength_in_couplant"]

    new_ray_geometry = lambda pathname: make_context()["ray_geometry_dict"][pathname]
    new_path = lambda pathname: make_context()["paths"][pathname]

    # realistic dry run so that all caching mechanisms are called
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]
        _ = model.beamspread_2d_for_path(ray_geometry)
        _ = model.reverse_beamspread_2d_for_path(ray_geometry)
        _ = model.transmission_reflection_for_path(path, ray_geometry)
        _ = model.reverse_transmission_reflection_for_path(path, ray_geometry)
        _ = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, element_width, wavelength
        )

    # Compare partially cached results to fresh ones
    for pathname, ray_geometry in ray_geometry_dict.items():
        path = paths[pathname]

        r1 = model.beamspread_2d_for_path(ray_geometry)
        r2 = model.beamspread_2d_for_path(new_ray_geometry(pathname))
        np.testing.assert_allclose(r1, r2, err_msg=pathname)

        r1 = model.reverse_beamspread_2d_for_path(ray_geometry)
        r2 = model.reverse_beamspread_2d_for_path(new_ray_geometry(pathname))
        np.testing.assert_allclose(r1, r2, err_msg=pathname)

        r1 = model.transmission_reflection_for_path(path, ray_geometry)
        r2 = model.transmission_reflection_for_path(
            new_path(pathname), new_ray_geometry(pathname)
        )
        np.testing.assert_allclose(r1, r2, err_msg=pathname)

        r1 = model.reverse_transmission_reflection_for_path(path, ray_geometry)
        r2 = model.reverse_transmission_reflection_for_path(
            new_path(pathname), new_ray_geometry(pathname)
        )
        np.testing.assert_allclose(r1, r2, err_msg=pathname)

        r1 = model.directivity_2d_rectangular_in_fluid_for_path(
            ray_geometry, element_width, wavelength
        )
        r2 = model.directivity_2d_rectangular_in_fluid_for_path(
            new_ray_geometry(pathname), element_width, wavelength
        )
        np.testing.assert_allclose(r1, r2, err_msg=pathname)


def test_beamspread_2d_direct():
    context = make_context()
    block = context["block"]
    """:type : arim.Material"""
    couplant = context["couplant"]
    """:type : arim.Material"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""

    # hardcoded results
    expected_beamspread = {
        "L": array([[3.2375215, 0.84817938]]),
        "T": array([[4.37280604, 1.38511721]]),
        "LL": array([[2.35172691, 1.24586279]]),
        "LT": array([[2.50582172, 1.1942895]]),
        "TL": array([[2.93422015, 0.7995886]]),
        "TT": array([[3.25137185, 1.90838339]]),
    }
    beamspread = dict()
    for pathname, ray_geometry in ray_geometry_dict.items():
        beamspread[pathname] = model.beamspread_2d_for_path(ray_geometry)
        np.testing.assert_allclose(
            beamspread[pathname], expected_beamspread[pathname], err_msg=pathname
        )
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
    np.testing.assert_allclose(beamspread["L"][0][0], beamspread_L, rtol=1e-5)

    # Path LL - scat 0:
    first_leg = 10e-3
    second_leg = 30e-3
    third_leg = 10.0e-3
    c1 = couplant.longitudinal_vel
    c2 = block.longitudinal_vel
    c3 = block.longitudinal_vel
    beta = c1 / c2
    gamma = c2 / c3
    beamspread_LL = 1.0 / np.sqrt(
        first_leg + second_leg / beta + third_leg / (gamma * beta)
    )
    np.testing.assert_allclose(beamspread["LL"][0][0], beamspread_LL, rtol=1e-5)

    # Path LT - scat 0:
    first_leg = 10e-3
    second_leg = 30e-3
    third_leg = 10.0e-3
    c1 = couplant.longitudinal_vel
    c2 = block.longitudinal_vel
    c3 = block.transverse_vel
    beta = c1 / c2
    gamma = c2 / c3
    beamspread_LT = 1.0 / np.sqrt(
        first_leg + second_leg / beta + third_leg / (gamma * beta)
    )
    np.testing.assert_allclose(beamspread["LT"][0][0], beamspread_LT, rtol=1e-5)


def test_beamspread_2d_reverse():
    context = make_context()
    block = context["block"]
    """:type : arim.Material"""
    couplant = context["couplant"]
    """:type : arim.Material"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""

    rev_paths = OrderedDict([(key, path.reverse()) for (key, path) in paths.items()])
    rev_ray_geometry_dict = OrderedDict(
        (k, arim.ray.RayGeometry.from_path(v)) for (k, v) in rev_paths.items()
    )

    # hardcoded results
    expected_beamspread = {
        "L": array([[6.69023357, 4.37716144]]),
        "T": array([[6.35918872, 4.44926218]]),
        "LL": array([[4.85976767, 3.96478629]]),
        "LT": array([[3.64411785, 1.84991477]]),
        "TL": array([[6.06346095, 5.31340498]]),
        "TT": array([[4.72833395, 3.96638874]]),
    }
    beamspread = dict()
    for pathname, ray_geometry in ray_geometry_dict.items():
        beamspread[pathname] = model.reverse_beamspread_2d_for_path(ray_geometry)
        np.testing.assert_allclose(
            beamspread[pathname], expected_beamspread[pathname], err_msg=pathname
        )
        # Uncomment the following line to generate hardcoded-values:
        # (use -s flag in pytest to show output)
        # print("'{}': {},".format(pathname, repr(beamspread[pathname])))

    # Direct beamspread of reversed paths should give the same results as reversed
    # beamspread of direct paths.
    for pathname, ray_geometry in rev_ray_geometry_dict.items():
        rev_beamspread = model.beamspread_2d_for_path(ray_geometry).T
        # this is a fairly tolerant comparison but it works.
        np.testing.assert_allclose(
            rev_beamspread, expected_beamspread[pathname], rtol=1e-2
        )

    # Reversed path L - scat 0:
    # first_leg is in the solid
    first_leg = 20e-3
    second_leg = 10e-3
    c1 = block.longitudinal_vel
    c2 = couplant.longitudinal_vel
    beta = c1 / c2
    beamspread_L = np.sqrt(1 / (first_leg + second_leg / beta))
    np.testing.assert_allclose(beamspread["L"][0][0], beamspread_L, rtol=1e-5)

    # Path LL - scat 0:
    first_leg = 10e-3
    second_leg = 30e-3
    third_leg = 10.0e-3
    c1 = block.longitudinal_vel
    c2 = block.longitudinal_vel
    c3 = couplant.longitudinal_vel
    beta = c1 / c2
    gamma = c2 / c3
    beamspread_LL = 1.0 / np.sqrt(
        first_leg + second_leg / beta + third_leg / (gamma * beta)
    )
    np.testing.assert_allclose(beamspread["LL"][0][0], beamspread_LL, rtol=1e-5)

    # Path LT - scat 0:
    first_leg = 10e-3
    second_leg = 30e-3
    third_leg = 10.0e-3
    c1 = block.transverse_vel
    c2 = block.longitudinal_vel
    c3 = couplant.longitudinal_vel
    beta = c1 / c2
    gamma = c2 / c3
    beamspread_LT = 1.0 / np.sqrt(
        first_leg + second_leg / beta + third_leg / (gamma * beta)
    )
    np.testing.assert_allclose(beamspread["LT"][0][0], beamspread_LT, rtol=1e-5)


def test_transmission_reflection_direct():
    context = make_context()
    block = context["block"]
    """:type : arim.Material"""
    couplant = context["couplant"]
    """:type : arim.Material"""
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""

    # hardcoded results
    expected_transrefl = {
        "L": array([[1.84037966 + 0.0j, 2.24977557 + 0.0j]]),
        "T": array([[-1.92953093e-03 + 0.0j, -2.06170606e00 + 0.76785004j]]),
        "LL": array([[-1.54661755 + 0.0j, -0.73952250 + 0.0j]]),
        "LT": array([[2.77193223e-04 + 0.0j, 9.17687259e-01 + 0.0j]]),
        "TL": array([[1.18487466e-06 - 0.0j, 1.29264072e00 - 0.0j]]),
        "TT": array([[0.00192953 - 0.0j, -1.39294465 + 0.09773593j]]),
    }
    transrefl = dict()
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        transrefl[pathname] = model.transmission_reflection_for_path(path, ray_geometry)
        np.testing.assert_allclose(
            transrefl[pathname], expected_transrefl[pathname], rtol=0.0, atol=1e-6
        )
        # print("'{}': {},".format(pathname, repr(transrefl[pathname])))

    # For the first scatterer (angle of incidence 0):
    params = (
        complex(0.0),
        couplant.density,
        block.density,
        couplant.longitudinal_vel,
        block.longitudinal_vel,
        block.transverse_vel,
        complex(0.0),
        complex(0.0),
    )
    tol = dict(rtol=0, atol=1e-2)
    _, transrefl_L, transrefl_T = arim.model.fluid_solid(*params)
    refl_LL, refl_LT, _ = arim.model.solid_l_fluid(*params)
    refl_TL, refl_TT, _ = arim.model.solid_t_fluid(*params)
    np.testing.assert_allclose(transrefl["L"][0][0], transrefl_L)
    np.testing.assert_allclose(transrefl["T"][0][0], transrefl_T, **tol)
    np.testing.assert_allclose(transrefl["LL"][0][0], transrefl_L * refl_LL)
    np.testing.assert_allclose(transrefl["LT"][0][0], transrefl_L * refl_LT, **tol)
    np.testing.assert_allclose(transrefl["TL"][0][0], transrefl_T * refl_LL, **tol)
    np.testing.assert_allclose(transrefl["TT"][0][0], transrefl_T * refl_LT, **tol)


def test_transmission_reflection_reverse_hardcode():
    context = make_context()
    block = context["block"]
    """:type : arim.Material"""
    couplant = context["couplant"]
    """:type : arim.Material"""
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    rev_paths = context["rev_paths"]
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    rev_ray_geometry_dict = context["rev_ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""

    # ====================================================================================
    # Check hardcoded results
    # There is no guarantee that they are good, this test will just catch any change.

    # hardcoded results
    expected_rev_transrefl = {
        "L": array([[0.15962002 + 0.0j, 0.07813451 + 0.0j]]),
        "T": array([[0.00033791 + 0.0j, 0.16346328 - 0.06087933j]]),
        "LL": array([[-0.13414141 + 0.0j, -0.04164956 + 0.0j]]),
        "LT": array([[-4.85439698e-05 + 0.0j, -1.50885505e-01 + 0.0j]]),
        "TL": array([[1.02766857e-07 + 0.0j, 3.48642287e-02 + 0.0j]]),
        "TT": array([[-0.00033791 + 0.0j, 0.17068649 - 0.01197621j]]),
    }
    rev_transrefl = dict()
    for pathname, path in paths.items():
        ray_geometry = ray_geometry_dict[pathname]
        rev_transrefl[pathname] = model.reverse_transmission_reflection_for_path(
            path, ray_geometry
        )
        np.testing.assert_allclose(
            rev_transrefl[pathname],
            expected_rev_transrefl[pathname],
            rtol=0.0,
            atol=1e-6,
        )
        # print("'{}': {},".format(pathname, repr(rev_transrefl[pathname])))

    # ====================================================================================
    # Check limit case: angle of incidence 0° (first scatterer)
    params = (
        complex(0.0),
        couplant.density,
        block.density,
        couplant.longitudinal_vel,
        block.longitudinal_vel,
        block.transverse_vel,
        complex(0.0),
        complex(0.0),
    )
    tol = dict(rtol=0, atol=1e-3)
    refl_LL, refl_LT, trans_L = arim.model.solid_l_fluid(*params)
    refl_TL, refl_TT, trans_T = arim.model.solid_t_fluid(*params)
    np.testing.assert_allclose(rev_transrefl["L"][0][0], trans_L, **tol)
    np.testing.assert_allclose(rev_transrefl["T"][0][0], trans_T, **tol)
    np.testing.assert_allclose(rev_transrefl["LL"][0][0], trans_L * refl_LL, **tol)
    np.testing.assert_allclose(rev_transrefl["LT"][0][0], trans_L * refl_TL, **tol)
    np.testing.assert_allclose(rev_transrefl["TL"][0][0], trans_T * refl_LT, **tol)
    np.testing.assert_allclose(rev_transrefl["TT"][0][0], trans_T * refl_TT, **tol)

    # ====================================================================================
    # Check that the direct transmission-reflection coefficients for the reversed path are
    # consistent with the reversed coefficients for the direct path.
    # Unfortunately this is not exact because of angles approximation.

    rev_transrefl2 = dict()
    for pathname, rev_ray_geometry in rev_ray_geometry_dict.items():
        rev_transrefl2[pathname] = model.transmission_reflection_for_path(
            rev_paths[pathname], rev_ray_geometry
        ).T
        np.testing.assert_allclose(
            rev_transrefl2[pathname],
            expected_rev_transrefl[pathname],
            rtol=1e-1,
            atol=1e-3,
            err_msg=pathname,
        )
        # TODO: very poor comparison


def test_transmission_reflection_reverse_stokes():
    """
    Compare function reverse_transmission_reflection_for_path() with the function
    transmission_reflection_for_path() using Stokes relations.

    """
    context = make_context()
    block = context["block"]
    """:type : arim.Material"""
    couplant = context["couplant"]
    """:type : arim.Material"""
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""

    rho_fluid = couplant.density
    rho_solid = block.density
    c_fluid = couplant.longitudinal_vel
    c_l = block.longitudinal_vel
    c_t = block.transverse_vel

    z_l = block.longitudinal_vel * rho_solid
    z_t = block.transverse_vel * rho_solid
    z_fluid = couplant.longitudinal_vel * rho_fluid

    magic_coefficient = -1.0

    # ====================================================================================
    pathname = "LT"

    # Frontwall in direct sense: fluid inc, solid L out
    alpha_fluid = np.asarray(
        ray_geometry_dict[pathname].conventional_inc_angle(1), complex
    )
    alpha_l = arim.model.snell_angles(alpha_fluid, c_fluid, c_l)
    correction_frontwall = (z_fluid / z_l) * (np.cos(alpha_l) / np.cos(alpha_fluid))
    del alpha_fluid, alpha_l

    # Backwall in direct sense: solid L inc, solid T out
    alpha_l = np.asarray(ray_geometry_dict[pathname].conventional_inc_angle(2), complex)
    alpha_t = arim.model.snell_angles(alpha_l, c_l, c_t)
    correction_backwall = (z_l / z_t) * (np.cos(alpha_t) / np.cos(alpha_l))
    del alpha_l, alpha_t
    transrefl_stokes = (
        model.transmission_reflection_for_path(
            paths[pathname], ray_geometry_dict[pathname]
        )
        * correction_backwall
        * correction_frontwall
    )
    transrefl_rev = model.reverse_transmission_reflection_for_path(
        paths[pathname], ray_geometry_dict[pathname]
    )

    np.testing.assert_allclose(
        transrefl_stokes, magic_coefficient * transrefl_rev, err_msg=pathname
    )

    # ====================================================================================
    pathname = "TL"

    # Frontwall in direct sense: fluid inc, solid T out
    alpha_fluid = np.asarray(
        ray_geometry_dict[pathname].conventional_inc_angle(1), complex
    )
    alpha_t = arim.model.snell_angles(alpha_fluid, c_fluid, c_t)
    correction_frontwall = (z_fluid / z_t) * (np.cos(alpha_t) / np.cos(alpha_fluid))
    del alpha_fluid, alpha_t

    # Backwall in direct sense: solid T inc, solid L out
    alpha_t = np.asarray(ray_geometry_dict[pathname].conventional_inc_angle(2), complex)
    alpha_l = arim.model.snell_angles(alpha_t, c_t, c_l)
    correction_backwall = (z_t / z_l) * (np.cos(alpha_l) / np.cos(alpha_t))
    del alpha_l, alpha_t
    transrefl_stokes = (
        model.transmission_reflection_for_path(
            paths[pathname], ray_geometry_dict[pathname]
        )
        * correction_backwall
        * correction_frontwall
    )
    transrefl_rev = model.reverse_transmission_reflection_for_path(
        paths[pathname], ray_geometry_dict[pathname]
    )

    np.testing.assert_allclose(transrefl_stokes, transrefl_rev, err_msg=pathname)

    # ====================================================================================
    pathname = "TT"

    # Frontwall in direct sense: fluid inc, solid T out
    alpha_fluid = np.asarray(
        ray_geometry_dict[pathname].conventional_inc_angle(1), complex
    )
    alpha_l = arim.model.snell_angles(alpha_fluid, c_fluid, c_l)  # noqa
    alpha_t = arim.model.snell_angles(alpha_fluid, c_fluid, c_t)
    correction_frontwall = (z_fluid / z_t) * (np.cos(alpha_t) / np.cos(alpha_fluid))

    # Backwall in direct sense: solid L inc, solid L out
    correction_backwall = 1.0
    transrefl_stokes = (
        model.transmission_reflection_for_path(
            paths[pathname], ray_geometry_dict[pathname]
        )
        * correction_backwall
        * correction_frontwall
    )
    transrefl_rev = model.reverse_transmission_reflection_for_path(
        paths[pathname], ray_geometry_dict[pathname]
    )

    np.testing.assert_allclose(
        transrefl_stokes, magic_coefficient * transrefl_rev, err_msg=pathname
    )


def test_material_attenuation():
    # no att
    context = make_context()
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    rev_paths = context["rev_paths"]
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    rev_ray_geometry_dict = context["rev_ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    frequency = context["freq"]

    for path, ray_geometry, rev_path, rev_ray_geometry in zip(
        paths.values(),
        ray_geometry_dict.values(),
        rev_paths.values(),
        rev_ray_geometry_dict.values(),
    ):
        att = model.material_attenuation_for_path(path, ray_geometry, frequency)
        rev_att = model.material_attenuation_for_path(
            rev_path, rev_ray_geometry, frequency
        )

        np.testing.assert_allclose(att, 1.0)
        np.testing.assert_allclose(rev_att, 1.0)

    # add attenuation
    context = make_context(
        couplant_att=arim.material_attenuation_factory("constant", 7.0)
    )
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    rev_paths = context["rev_paths"]
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    rev_ray_geometry_dict = context["rev_ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""
    frequency = context["freq"]

    for path, ray_geometry, rev_path, rev_ray_geometry in zip(
        paths.values(),
        ray_geometry_dict.values(),
        rev_paths.values(),
        rev_ray_geometry_dict.values(),
    ):
        att = model.material_attenuation_for_path(path, ray_geometry, frequency)
        rev_att = model.material_attenuation_for_path(
            rev_path, rev_ray_geometry, frequency
        )

        np.testing.assert_allclose(att, rev_att.T)
        np.testing.assert_array_equal(att > 0.0, True)
        np.testing.assert_array_equal(att < 1.0, True)


def bak_test_sensitivity():
    numpoints = 30
    numelements = 16
    result = np.zeros(numpoints)

    x = np.exp(1j * 0.3)

    # FMC case
    tx, rx = arim.ut.fmc(numelements)
    numtimetraces = len(tx)
    timetrace_weights = np.ones(numtimetraces)

    model_amplitudes = np.zeros((numpoints, numtimetraces), complex)
    model_amplitudes[0] = x

    # write on result
    model.sensitivity_image(model_amplitudes, timetrace_weights, result)
    np.testing.assert_almost_equal(result[0], numelements * numelements)
    np.testing.assert_allclose(result[1:], 0.0)

    # create a new array
    result2 = model.sensitivity_image(model_amplitudes, timetrace_weights)
    np.testing.assert_almost_equal(result, result2)

    # FMC case
    tx, rx = arim.ut.hmc(numelements)
    numtimetraces = len(tx)
    timetrace_weights = 2.0 * np.ones(numtimetraces)
    timetrace_weights[tx == rx] = 1.0

    model_amplitudes = np.zeros((numpoints, numtimetraces), complex)
    model_amplitudes[0] = x

    model.sensitivity_image(model_amplitudes, timetrace_weights, result)
    np.testing.assert_almost_equal(result[0], numelements * numelements)
    np.testing.assert_allclose(result[1:], 0.0)

    # create a new array
    result2 = model.sensitivity_image(model_amplitudes, timetrace_weights)
    np.testing.assert_almost_equal(result, result2)


def random_uniform_complex(low=0.0, high=1.0, size=None):
    return np.random.uniform(low, high, size) + 1j * np.random.uniform(low, high, size)


def make_point_source_scattering_func(context):
    block = context["block"]
    vl = block.longitudinal_vel
    vt = block.transverse_vel

    return {
        "LL": lambda inc, out: np.full_like(inc, 1.0),
        "LT": lambda inc, out: np.full_like(inc, vl / vt),
        "TL": lambda inc, out: np.full_like(inc, vt / vl),
        "TT": lambda inc, out: np.full_like(inc, 1.0),
    }


def make_point_source_scattering_matrix(context):
    block = context["block"]
    vl = block.longitudinal_vel
    vt = block.transverse_vel

    return {
        "LL": np.ones((2, 2)),
        "LT": np.full((2, 2), vl / vt),
        "TL": np.full((2, 2), vt / vl),
        "TT": np.ones((2, 2)),
    }


def make_random_ray_weights(context):
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""

    numelements = context["numelements"]
    numpoints = context["numpoints"]

    tx_ray_weights_dict = {}
    rx_ray_weights_dict = {}
    tx_ray_weights_debug_dict = None
    rx_ray_weights_debug_dict = None
    scattering_angles_dict = {}
    for pathname, path in paths.items():
        tx_ray_weights_dict[path] = np.asfortranarray(
            random_uniform_complex(size=(numelements, numpoints))
        )
        rx_ray_weights_dict[path] = np.asfortranarray(
            random_uniform_complex(size=(numelements, numpoints))
        )
        ray_geometry = ray_geometry_dict[pathname]
        scattering_angles_dict[path] = np.asfortranarray(
            ray_geometry.signed_inc_angle(-1)
        )

    return model.RayWeights(
        tx_ray_weights_dict,
        rx_ray_weights_dict,
        tx_ray_weights_debug_dict,
        rx_ray_weights_debug_dict,
        scattering_angles_dict,
    )


def test_model_amplitudes_factory():
    context = make_context()
    views = context["views"]

    ray_weights = make_random_ray_weights(context)
    scattering_funcs = make_point_source_scattering_func(context)
    scattering_matrices = make_point_source_scattering_matrix(context)

    # tx, rx = arim.ut.hmc(context['numelements'])
    tx = np.array([0, 0, 0])
    rx = np.array([0, 0, 0])
    numtimetraces = len(tx)
    numpoints = context["numpoints"]

    for viewname, view in views.items():
        # With scattering functions -----------
        amps = model.model_amplitudes_factory(
            tx, rx, view, ray_weights, scattering_funcs
        )
        a_ref = amps[...].copy()
        np.testing.assert_array_equal(a_ref, amps[...])

        assert amps.shape == (numpoints, numtimetraces)
        assert amps[...].shape == (numpoints, numtimetraces)
        assert amps[0].shape == (numtimetraces,)
        assert amps[:1].shape == (1, numtimetraces)
        assert amps[:1, ...].shape == (1, numtimetraces)
        assert amps[slice(0, 1), ...].shape == (1, numtimetraces)

        with pytest.raises(IndexError):
            amps[0, 0]

        for k, i in np.ndindex(numpoints, numtimetraces):
            assert amps[k][i] == a_ref[k, i]
            assert amps[k, ...][i] == a_ref[k, i]

        np.testing.assert_array_equal(amps[[0, 0, 0]], a_ref[[0, 0, 0]])
        np.testing.assert_array_equal(amps[:1, ...], amps[:1])

        with pytest.raises(TypeError):
            amps[0] = 1.0

        # With scattering matrices -----------
        amps = model.model_amplitudes_factory(
            tx, rx, view, ray_weights, scattering_matrices
        )
        np.testing.assert_allclose(a_ref, amps[...])

        assert amps.shape == (numpoints, numtimetraces)
        assert amps[...].shape == (numpoints, numtimetraces)
        assert amps[0].shape == (numtimetraces,)
        assert amps[:1].shape == (1, numtimetraces)
        assert amps[:1, ...].shape == (1, numtimetraces)
        assert amps[slice(0, 1), ...].shape == (1, numtimetraces)

        with pytest.raises(ValueError):
            amps[0, 0]

        for k, i in np.ndindex(numpoints, numtimetraces):
            np.testing.assert_allclose(amps[k][i], a_ref[k, i])
            np.testing.assert_allclose(amps[k, ...][i], a_ref[k, i])

        np.testing.assert_allclose(amps[[0, 0, 0]], a_ref[[0, 0, 0]])
        np.testing.assert_allclose(amps[:1, ...], amps[:1])

        with pytest.raises(TypeError):
            # TypeError: '_ModelAmplitudesWithScatMatrix' object does not support item assignment
            amps[0] = 1.0


def test_sensitivity_tfm():
    context = make_context()
    views = context["views"]

    ray_weights = make_random_ray_weights(context)
    scattering_dict = make_point_source_scattering_func(context)

    # we have only one element, duplicate timetrace to check the sum is actually performed
    tx = np.array([0, 0])
    rx = np.array([0, 0])
    numtimetraces = len(tx)
    numpoints = context["numpoints"]

    timetrace_weights = np.array([3.0, 7.0])

    for viewname, view in views.items():
        amps = model.model_amplitudes_factory(
            tx, rx, view, ray_weights, scattering_dict
        )

        # Sensitivity for uniform TFM
        sensitivity = model.sensitivity_uniform_tfm(amps, timetrace_weights)
        assert sensitivity.shape == (numpoints,)
        sensitivity2 = model.sensitivity_uniform_tfm(
            amps, timetrace_weights, block_size=1
        )
        np.testing.assert_array_equal(sensitivity, sensitivity2)

        for k in range(numpoints):
            expected = 0.0
            for i in range(numtimetraces):
                expected += amps[k][i] * timetrace_weights[i]
            expected /= numtimetraces
            assert np.isclose(sensitivity[k], expected)

        # Sensitivity for model-assited TFM
        sensitivity = model.sensitivity_model_assisted_tfm(amps, timetrace_weights)
        assert sensitivity.shape == (numpoints,)
        sensitivity2 = model.sensitivity_model_assisted_tfm(
            amps, timetrace_weights, block_size=1
        )
        np.testing.assert_array_equal(sensitivity, sensitivity2)

        for k in range(numpoints):
            expected = 0.0
            for i in range(numtimetraces):
                expected += np.abs(amps[k][i] * amps[k][i]) * timetrace_weights[i]
            expected /= numtimetraces
            assert np.isclose(sensitivity[k], expected)


def test_directivity_2d_rectangular_in_fluid():
    theta = 0.0
    element_width = 1e-3
    wavelength = 0.5e-3
    directivity = arim.model.directivity_2d_rectangular_in_fluid(
        theta, element_width, wavelength
    )

    assert np.isclose(directivity, 1.0)

    # From the NDT library (2016/03/22):
    # >>> fn_calc_directivity_main(0.7, 1., 0.3, 'wooh')
    matlab_res = 0.931080327325574
    assert np.isclose(
        arim.model.directivity_2d_rectangular_in_fluid(0.3, 0.7, 1.0), matlab_res
    )


def test_fluid_solid_real():
    """
    Test fluid_solid() below critical angles (real only).
    The conservation of energy should be respected.

    Stay below critical angles.
    """
    alpha_fluid = np.deg2rad([0, 5, 10])

    # water:
    c_fluid = 1480.0
    rho_fluid = 1000.0

    # aluminium :
    c_l = 6320.0
    c_t = 3130.0
    rho_solid = 2700.0

    with np.errstate(invalid="raise"):
        alpha_l = arim.model.snell_angles(alpha_fluid, c_fluid, c_l)
        alpha_t = arim.model.snell_angles(alpha_fluid, c_fluid, c_t)

        reflection, transmission_l, transmission_t = arim.model.fluid_solid(
            alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_l, alpha_t
        )
    assert reflection.dtype.kind == "f"
    assert transmission_l.dtype.kind == "f"
    assert transmission_t.dtype.kind == "f"

    # Compute the energy incoming and outcoming: the principle of conservation of energy
    # must be respected.
    # Reference: Schmerr, §6.3.2
    # Caveat about inhomogeneous waves: Schmerr, §6.2.5

    # incident pressure
    pres_i = 10000.0

    # cross section areas:
    area_fluid = np.cos(alpha_fluid)
    area_l = np.cos(alpha_l)
    area_t = np.cos(alpha_t)

    # Conservation of energy
    inc_energy = 0.5 * pres_i**2 / (rho_fluid * c_fluid) * area_fluid
    energy_refl = 0.5 * (reflection * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    energy_l = 0.5 * (transmission_l * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (transmission_t * pres_i) ** 2 / (rho_solid * c_t) * area_t

    np.testing.assert_allclose(inc_energy, energy_refl + energy_l + energy_t)


def test_fluid_solid_complex():
    """
    Test fluid_solid() below and above critical angles (complex).
    The conservation of energy should be respected for all cases.
    """

    alpha_fluid = np.asarray(np.deg2rad(np.arange(0.0, 85.0, 10.0)), dtype=complex)

    # water:
    c_fluid = 1480.0
    rho_fluid = 1000.0

    # aluminium :
    c_l = 6320.0
    c_t = 3130.0
    rho_solid = 2700.0

    with np.errstate(invalid="raise"):
        alpha_l = arim.model.snell_angles(alpha_fluid, c_fluid, c_l)
        alpha_t = arim.model.snell_angles(alpha_fluid, c_fluid, c_t)

        reflection, transmission_l, transmission_t = arim.model.fluid_solid(
            alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_l, alpha_t
        )

    # Compute the energy incoming and outcoming: the principle of conservation of energy
    # must be respected.
    # Reference: Schmerr, §6.3.2
    # Caveat about inhomogeneous waves: Schmerr, §6.2.5

    # incident pressure
    pres_i = 10000.0

    # cross section areas:
    area_fluid = np.cos(alpha_fluid.real)
    area_l = np.cos(alpha_l.real)
    area_t = np.cos(alpha_t.real)

    inc_energy = 0.5 * pres_i**2 / (rho_fluid * c_fluid) * area_fluid
    energy_refl = (
        0.5 * (np.abs(reflection) * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    )
    energy_l = 0.5 * (np.abs(transmission_l) * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (np.abs(transmission_t) * pres_i) ** 2 / (rho_solid * c_t) * area_t

    np.testing.assert_allclose(inc_energy, energy_refl + energy_l + energy_t)


def test_solid_l_fluid():
    """
    Test solid_l_fluid() below critical angles (real only).
    The conservation of energy should be respected.
    """
    alpha_l = np.deg2rad(np.arange(0.0, 85.0, 10.0))

    # water:
    c_fluid = 1480.0
    rho_fluid = 1000.0

    # aluminium :
    c_l = 6320.0
    c_t = 3130.0
    rho_solid = 2700.0

    with np.errstate(invalid="raise"):
        alpha_fluid = arim.model.snell_angles(alpha_l, c_l, c_fluid)
        alpha_t = arim.model.snell_angles(alpha_l, c_l, c_t)

        reflection_l, reflection_t, transmission = arim.model.solid_l_fluid(
            alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid, alpha_t
        )

    # Compute the energy incoming and outcoming: the principle of conservation of energy
    # must be respected.
    # Reference: Schmerr, §6.3.2
    # Caveat about inhomogeneous waves: Schmerr, §6.2.5

    # incident pressure
    pres_i = 10000.0

    # cross section areas:
    area_fluid = np.cos(alpha_fluid)
    area_l = np.cos(alpha_l)
    area_t = np.cos(alpha_t)

    inc_energy = 0.5 * pres_i**2 / (rho_solid * c_l) * area_l
    energy_trans = (
        0.5 * (transmission * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    )
    energy_l = 0.5 * (reflection_l * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (reflection_t * pres_i) ** 2 / (rho_solid * c_t) * area_t

    # Check the conservation of energy:
    np.testing.assert_allclose(inc_energy, energy_trans + energy_l + energy_t)


def test_solid_t_fluid_complex():
    """
    Test solid_t_fluid() below and above critical angles (complex).
    The conservation of energy should be respected for all cases.
    """
    alpha_t = np.asarray(np.deg2rad([0, 5, 10, 20, 30, 40]), dtype=complex)

    # water:
    c_fluid = 1480.0
    rho_fluid = 1000.0

    # aluminium :
    c_l = 6320.0
    c_t = 3130.0
    rho_solid = 2700.0

    with np.errstate(invalid="raise"):
        alpha_fluid = arim.model.snell_angles(alpha_t, c_t, c_fluid)
        alpha_l = arim.model.snell_angles(alpha_t, c_t, c_l)

        reflection_l, reflection_t, transmission = arim.model.solid_t_fluid(
            alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid, alpha_l
        )

    # Compute the energy incoming and outcoming: the principle of conservation of energy
    # must be respected.
    # Reference: Schmerr, §6.3.2
    # Caveat about inhomogeneous waves: Schmerr, §6.2.5

    # incident pressure
    pres_i = 10000.0

    # cross section areas:
    area_fluid = np.cos(alpha_fluid.real)
    area_l = np.cos(alpha_l.real)
    area_t = np.cos(alpha_t.real)

    inc_energy = 0.5 * pres_i**2 / (rho_solid * c_t) * area_t
    energy_trans = (
        0.5 * (np.abs(transmission) * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    )
    energy_l = 0.5 * (np.abs(reflection_l) * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (np.abs(reflection_t) * pres_i) ** 2 / (rho_solid * c_t) * area_t

    # Check equality of complex values (this equality has NO physical meaning)
    np.testing.assert_allclose(inc_energy, energy_trans + energy_l + energy_t)


def test_snell_angles():
    """
    Test snell_angles() with both real and complex angles
    """
    incidents_angles = np.deg2rad([0, 10, 20, 30])

    # water:
    c = 1480.0

    # aluminium :
    c_l = 6320
    c_t = 3130

    with np.errstate(invalid="ignore"):
        alpha_l = arim.model.snell_angles(incidents_angles, c, c_l)
        alpha_t = arim.model.snell_angles(incidents_angles, c, c_t)

    assert alpha_l.shape == incidents_angles.shape
    assert alpha_t.shape == incidents_angles.shape

    # Normal incident = normal refraction
    assert np.isclose(alpha_l[0], 0.0)
    assert np.isclose(alpha_t[0], 0.0)

    # 10°: transmitted L and T
    assert np.isfinite(alpha_l[1])
    assert np.isfinite(alpha_t[1])

    assert np.isclose(np.sin(alpha_l[1]) / np.sin(incidents_angles[1]), c_l / c)
    assert np.isclose(np.sin(alpha_t[1]) / np.sin(incidents_angles[1]), c_t / c)

    # 20°: total reflection for L, T is transmitted
    assert np.isnan(alpha_l[2])
    assert np.isfinite(alpha_t[2])

    # 30°: total reflection for L and T
    assert np.isnan(alpha_l[3])
    assert np.isnan(alpha_t[3])


def test_stokes_relation():
    """
    Test fluid_solid(), solid_t_fluid() and solid_l_fluid() by checking consistency with Stokes relations.

    Stokes relations link transmission coefficients solid -> fluid and fluid -> solid.
    Warning: Schmerr defines this coefficient for stress/pressure ratios such as in the solid the stress has the opposite sign of the
    pressure such as defined by Krautkrämer. Therefore we change the sign in the Stokes relations.

    References
    ----------
    Schmerr §6.3.3, equation (6.150a)
    """
    alpha_fluid = np.asarray(np.deg2rad(np.arange(0.0, 85.0, 10.0)), dtype=complex)
    alpha_fluid = np.asarray(np.deg2rad([0, 5, 10]), dtype=float)

    # water:
    c_fluid = 1480.0
    rho_fluid = 1000.0

    # aluminium :
    c_l = 6320.0
    c_t = 3130.0
    rho_solid = 2700.0

    alpha_l = arim.model.snell_angles(alpha_fluid, c_fluid, c_l)
    alpha_t = arim.model.snell_angles(alpha_fluid, c_fluid, c_t)

    # Transmission fluid->solid
    _, transmission_l_fs, transmission_t_fs = arim.model.fluid_solid(
        alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_l, alpha_t
    )
    refl_tl, refl_tt, transmission_t_sf = arim.model.solid_t_fluid(
        alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid, alpha_l
    )
    refl_ll, refl_lt, transmission_l_sf = arim.model.solid_l_fluid(
        alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid, alpha_t
    )

    # TODO: there is a magic coefficient here. Rose vs Krautkrämer discrepancy?
    magic_coefficient = -1.0
    transmission_t_sf *= magic_coefficient

    transmission_l_sf_stokes = (
        rho_fluid
        * c_fluid
        * np.cos(alpha_l)
        * transmission_l_fs
        / (rho_solid * c_l * np.cos(alpha_fluid))
    )
    transmission_t_sf_stokes = (
        rho_fluid
        * c_fluid
        * np.cos(alpha_t)
        * transmission_t_fs
        / (rho_solid * c_t * np.cos(alpha_fluid))
    )

    np.testing.assert_allclose(transmission_l_sf_stokes, transmission_l_sf)
    np.testing.assert_allclose(transmission_t_sf_stokes, transmission_t_sf)

    # Extend Stokes relation given by Schmerr to the reflection in solid against fluid.
    # Compare TL and LT
    corr = (c_t / c_l) * (np.cos(alpha_l) / np.cos(alpha_t))
    refl_lt2 = refl_tl * corr * magic_coefficient
    np.testing.assert_allclose(refl_lt2, refl_lt)

    refl_tl2 = refl_lt / corr * magic_coefficient
    np.testing.assert_allclose(refl_tl2, refl_tl)


def test_make_toneburst():
    dt = 50e-9
    num_samples = 70
    f0 = 2e6

    # Test 1: unwrapped, 5 cycles
    num_cycles = 5
    toneburst = arim.model.make_toneburst(num_cycles, f0, dt, num_samples)
    toneburst_complex = arim.model.make_toneburst(
        num_cycles, f0, dt, num_samples, analytical=True
    )
    # ensure we don'ray accidently change the tested function by hardcoding a result
    toneburst_ref = [
        -0.0,
        -0.003189670321154915,
        -0.004854168560396212,
        0.010850129632629638,
        0.05003499896758611,
        0.09549150281252627,
        0.10963449321242304,
        0.056021074460159935,
        -0.07171870434248846,
        -0.23227715582293904,
        -0.3454915028125263,
        -0.3287111632233889,
        -0.14480682837737863,
        0.16421016599756882,
        0.4803058311515585,
        0.6545084971874735,
        0.5767398385520084,
        0.23729829003245898,
        -0.25299591991478754,
        -0.6993825011625243,
        -0.9045084971874737,
        -0.7589819954073612,
        -0.2981668647423177,
        0.30416282581455123,
        0.8058273240537925,
        1.0,
        0.8058273240537925,
        0.30416282581455123,
        -0.2981668647423177,
        -0.7589819954073612,
        -0.904508497187474,
        -0.6993825011625244,
        -0.2529959199147875,
        0.23729829003245886,
        0.5767398385520082,
        0.6545084971874737,
        0.4803058311515585,
        0.1642101659975688,
        -0.14480682837737874,
        -0.32871116322338906,
        -0.3454915028125264,
        -0.2322771558229394,
        -0.07171870434248843,
        0.056021074460160004,
        0.10963449321242318,
        0.09549150281252633,
        0.05003499896758629,
        0.010850129632629638,
        -0.004854168560396229,
        -0.00318967032115496,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_toneburst = np.argmax(toneburst_ref)

    assert len(toneburst) == num_samples
    assert toneburst.dtype == float
    assert toneburst_complex.dtype == complex
    np.testing.assert_allclose(toneburst_complex.real, toneburst)
    assert np.count_nonzero(np.isclose(toneburst, 1.0)) == 1, "1.0 does not appear"
    np.testing.assert_allclose(toneburst, toneburst_ref)

    # Test 2: wrapped, 5 cycles
    num_cycles = 5
    toneburst = arim.model.make_toneburst(num_cycles, f0, dt, num_samples, wrap=True)
    toneburst_complex = arim.model.make_toneburst(
        num_cycles, f0, dt, num_samples, analytical=True, wrap=True
    )

    assert len(toneburst) == num_samples
    assert toneburst.dtype == float
    assert toneburst_complex.dtype == complex
    np.testing.assert_allclose(toneburst_complex.real, toneburst)
    np.testing.assert_allclose(toneburst[0], 1.0)
    np.testing.assert_allclose(
        toneburst[:10], toneburst_ref[max_toneburst : 10 + max_toneburst]
    )
    np.testing.assert_allclose(
        toneburst[-10:], toneburst_ref[-10 + max_toneburst : max_toneburst]
    )

    # num_samples = None
    toneburst = arim.model.make_toneburst(num_cycles, f0, dt, num_samples)
    toneburst_complex = arim.model.make_toneburst(
        num_cycles, f0, dt, num_samples, analytical=True
    )
    toneburst2 = arim.model.make_toneburst(num_cycles, f0, dt)
    toneburst_complex2 = arim.model.make_toneburst(num_cycles, f0, dt, analytical=True)
    len_pulse = len(toneburst2)
    np.testing.assert_allclose(toneburst2, toneburst[:len_pulse])
    np.testing.assert_allclose(toneburst_complex2, toneburst_complex[:len_pulse])


def test_make_toneburst2():
    num_cycles = 5
    dt = 50e-9
    f0 = 2e6

    # basic checks for analytical signal
    toneburst_time, toneburst, t0_idx = arim.model.make_toneburst2(
        num_cycles, f0, dt, analytical=True
    )
    assert toneburst_time.samples[t0_idx] == 0.0
    assert toneburst[t0_idx] == complex(1.0)

    # basic checks for real signal
    toneburst_time, toneburst, t0_idx = arim.model.make_toneburst2(
        num_cycles, f0, dt, analytical=False
    )

    assert toneburst_time.samples[t0_idx] == 0.0
    assert toneburst[t0_idx] == 1.0

    # check consistency between toneburst and make_toneburst2
    toneburst = arim.model.make_toneburst(num_cycles, f0, dt, analytical=False)

    _, toneburst2, _ = arim.model.make_toneburst2(
        num_cycles, f0, dt, num_before=0, analytical=False
    )
    n = len(toneburst)
    np.testing.assert_allclose(toneburst2[:n], toneburst)
