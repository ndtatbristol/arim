import numpy as np
import pytest
from matplotlib import pyplot as plt

import arim
import arim.path as p
from arim import geometry as g
import arim.plot as aplt


def test_filter_unique_views():
    unique_views = p.filter_unique_views([('AB', 'CD'), ('DC', 'BA'), ('X', 'YZ'),
                                          ('ZY', 'X')])
    assert unique_views == [('AB', 'CD'), ('X', 'YZ')]


def test_make_viewnames():
    L = 'L'
    T = 'T'
    LL = 'LL'

    viewnames = p.make_viewnames(['L', 'T'], unique_only=False)
    assert viewnames == [(L, L), (L, T), (T, L), (T, T)]

    viewnames = p.make_viewnames(['L', 'T'], unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T)]

    viewnames = p.make_viewnames(['L', 'T'], unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T)]

    # legacy IMAGING_MODES
    legacy_imaging_views = ["L-L", "L-T", "T-T",
                            "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L",
                            "TT-T",
                            "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                            "LT-LT", "LT-TL", "LT-TT",
                            "TL-LT", "TL-TT",
                            "TT-TT"]
    legacy_imaging_views = [tuple(view.split('-')) for view in legacy_imaging_views]

    viewnames = p.make_viewnames(['L', 'T', 'LL', 'LT', 'TL', 'TT'])
    assert viewnames == legacy_imaging_views

    viewnames = p.make_viewnames(p.DIRECT_PATHS + p.SKIP_PATHS)
    assert viewnames == legacy_imaging_views

    viewnames = p.make_viewnames(p.DIRECT_PATHS + p.SKIP_PATHS + p.DOUBLE_SKIP_PATHS)
    assert viewnames[:21] == legacy_imaging_views
    assert len(viewnames) == 105

    viewnames = p.make_viewnames(p.DIRECT_PATHS + p.SKIP_PATHS + p.DOUBLE_SKIP_PATHS,
                                 unique_only=False)
    assert len(viewnames) == 14 * 14

    viewnames = p.make_viewnames(['L', 'T', 'LL'], unique_only=True)
    assert viewnames == [(L, L), (L, T), (T, T), (LL, L), (LL, T), (LL, LL)]


def test_points_1d_wall_z():
    args = dict(xmin=10, xmax=20, numpoints=6, y=1, z=2, name='toto')
    points, orientations = p.points_1d_wall_z(**args)
    assert points.shape == (6,)
    assert orientations.shape == (6, 3)

    np.testing.assert_allclose(points.x, [10, 12, 14, 16, 18, 20])
    np.testing.assert_allclose(points.y, args['y'])
    np.testing.assert_allclose(points.z, args['z'])
    for i in range(6):
        np.testing.assert_allclose(orientations[i], np.eye(3))


def test_path_in_immersion():
    xmin = -20e-3
    xmax = 100e-3

    couplant = arim.Material(longitudinal_vel=1480, transverse_vel=None, density=1000.,
                             state_of_matter='liquid', metadata={'long_name': 'Water'})
    block = arim.Material(longitudinal_vel=6320., transverse_vel=3130., density=2700.,
                          state_of_matter='solid', metadata={'long_name': 'Aluminium'})

    probe_points, probe_orientations = p.points_1d_wall_z(0e-3, 15e-3,
                                                          z=0., numpoints=16,
                                                          name='Frontwall')

    frontwall_points, frontwall_orientations = p.points_1d_wall_z(xmin, xmax,
                                                                  z=0., numpoints=20,
                                                                  name='Frontwall')
    backwall_points, backwall_orientations = p.points_1d_wall_z(xmin, xmax,
                                                                z=40.18e-3, numpoints=21,
                                                                name='Backwall')

    grid = arim.geometry.Grid(xmin, xmax,
                              ymin=0., ymax=0.,
                              zmin=0., zmax=20e-3,
                              pixel_size=5e-3)
    grid_points, grid_orientation = p.points_from_grid(grid)

    interfaces = p.interfaces_for_block_in_immersion(couplant, probe_points,
                                                     probe_orientations,
                                                     frontwall_points,
                                                     frontwall_orientations,
                                                     backwall_points,
                                                     backwall_orientations,
                                                     grid_points, grid_orientation)
    assert interfaces['probe'].points is probe_points
    assert interfaces['probe'].orientations is probe_orientations
    assert interfaces['frontwall_trans'].points is frontwall_points
    assert interfaces['frontwall_trans'].orientations is frontwall_orientations
    assert interfaces['frontwall_refl'].points is frontwall_points
    assert interfaces['frontwall_refl'].orientations is frontwall_orientations
    assert interfaces['backwall_refl'].points is backwall_points
    assert interfaces['backwall_refl'].orientations is backwall_orientations
    assert interfaces['grid'].points is grid_points
    assert interfaces['grid'].orientations is grid_orientation

    # ------------------------------------------------------------------------------------
    # 6 paths, 21 views

    paths = p.paths_for_block_in_immersion(block, couplant, interfaces)
    assert len(paths) == 6
    assert paths['L'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                           frontwall_points, block.longitudinal_vel,
                                           grid_points)
    assert paths['TL'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                            frontwall_points, block.transverse_vel,
                                            backwall_points,
                                            block.longitudinal_vel, grid_points)

    for path_key, path in paths.items():
        assert path_key == path.name

    # Make views
    views = p.views_for_block_in_immersion(paths)
    assert len(views) == 21

    view = views['LT-LT']
    assert view.tx_path is paths['LT']
    assert view.rx_path is paths['TL']

    # ------------------------------------------------------------------------------------
    # 14 paths, 105 views
    paths = p.paths_for_block_in_immersion(block, couplant, interfaces,
                                           max_number_of_reflection=2)
    assert len(paths) == 14
    assert paths['L'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                           frontwall_points, block.longitudinal_vel,
                                           grid_points)
    assert paths['TL'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                            frontwall_points, block.transverse_vel,
                                            backwall_points,
                                            block.longitudinal_vel, grid_points)
    assert paths['TTL'].to_fermat_path() == (probe_points, couplant.longitudinal_vel,
                                             frontwall_points, block.transverse_vel,
                                             backwall_points, block.transverse_vel,
                                             frontwall_points, block.longitudinal_vel,
                                             grid_points)

    for path_key, path in paths.items():
        assert path_key == path.name

    # Make views
    views = p.views_for_block_in_immersion(paths)
    assert len(views) == 105

    view = views['LT-LT']
    assert view.tx_path is paths['LT']
    assert view.rx_path is paths['TL']


def make_incoming_angles_results():
    no_tilt_ref_angles = np.array([78.69006752597979, 76.86597769360367,
                                   74.35775354279127, 70.70995378081126,
                                   64.98310652189996, 55.00797980144132,
                                   35.537677791974374, 0.0,
                                   35.537677791974374,
                                   55.007979801441344, 64.98310652189998,
                                   70.70995378081126, 74.35775354279127,
                                   76.86597769360367, 78.69006752597979])
    no_tilt_ref_angles_flipped = 180 - no_tilt_ref_angles
    angles_dict = {}
    # Key order: dest_points_are_above, are_normals_zplus, use_tilt
    angles_dict[True, False, False] = no_tilt_ref_angles
    angles_dict[False, False, False] = no_tilt_ref_angles_flipped
    angles_dict[True, True, False] = no_tilt_ref_angles_flipped
    angles_dict[False, True, False] = no_tilt_ref_angles

    return angles_dict


def make_outgoing_angles_results():
    no_tilt_ref_angles = np.array([78.69006752597979, 76.86597769360367,
                                   74.35775354279127, 70.70995378081126,
                                   64.98310652189996, 55.00797980144132,
                                   35.537677791974374, 0.0,
                                   35.537677791974374,
                                   55.007979801441344, 64.98310652189998,
                                   70.70995378081126, 74.35775354279127,
                                   76.86597769360367, 78.69006752597979])
    no_tilt_ref_angles_flipped = 180 - no_tilt_ref_angles
    angles_dict = {}
    # Key order: dest_points_are_above, are_normals_zplus, use_tilt
    angles_dict[True, False, False] = no_tilt_ref_angles_flipped
    angles_dict[False, False, False] = no_tilt_ref_angles
    angles_dict[True, True, False] = no_tilt_ref_angles
    angles_dict[False, True, False] = no_tilt_ref_angles_flipped

    return angles_dict


INCOMING_ANGLES = make_incoming_angles_results()
OUTGOING_ANGLES = make_outgoing_angles_results()


class TestLegacyRayGeometry:
    """
    Test legacy interface Rays.get_incoming_angles() and Rays.get_outgoing_angles()

    Source point: O(0., 0., 0.)

    Dest points 'above': line y = 0 and z = +1.
    Dest points 'below': line y = 0 and z = -1.

    The incoming angle is a function of the polar angle of the source point of the leg in the
    coordinate system of the interface point. This angle is the polar angle when the normal
    is not flipped, 180° minus the polar angle when the normal is flipped.

    """

    @staticmethod
    def make_ray_and_path(dest_points_are_above, are_normals_zplus, use_tilt):
        src_points, src_basis = arim.path.points_1d_wall_z(0., 0., 0., 1, name='source')
        if use_tilt:
            src_basis = src_basis.rotate(g.rotation_matrix_y(np.pi / 6))
        source_interface = arim.Interface(src_points, src_basis,
                                          are_normals_on_out_rays_side=are_normals_zplus)

        if dest_points_are_above:
            z = 1.
        else:
            z = -1.

        xmin = -5.
        xmax = 5.
        numpoints = 15

        dst_points, dst_basis = arim.path.points_1d_wall_z(xmin, xmax, z, numpoints,
                                                           name='dest')
        if use_tilt:
            dst_basis = dst_basis.rotate(g.rotation_matrix_y(np.pi / 6))

        dest_interface = arim.Interface(dst_points, dst_basis,
                                        are_normals_on_inc_rays_side=are_normals_zplus)

        material = arim.Material(np.nan, metadata=dict(long_name='Dummy'))

        interfaces = [source_interface, dest_interface]

        # The i-th ray starts from the source and ends at the i-th destination point.
        shape = [len(source_interface.points), len(dest_interface.points)]
        ray_indices = np.zeros((0, *shape), np.uint)
        times = np.empty(shape, float)
        times.fill(np.nan)

        path = arim.Path(interfaces, [material], ['L'])
        ray = arim.ray.Rays(times, ray_indices, path.to_fermat_path())
        path.rays = ray
        ray_geometry = arim.path.RayGeometry.from_path(path)
        return path, ray_geometry

    @pytest.mark.parametrize("dest_points_are_above, are_normals_zplus, use_tilt",
                             [  # (True, False, True),
                                 (True, False, False),
                                 (False, False, False),
                                 (True, True, False),
                                 (False, True, False)])
    def test_incoming_angles(self, show_plots, dest_points_are_above,
                             are_normals_zplus, use_tilt):
        path, ray_geometry = self.make_ray_and_path(dest_points_are_above,
                                                    are_normals_zplus,
                                                    use_tilt)

        num_src_points = len(path.interfaces[0].points)
        num_dst_points = len(path.interfaces[1].points)

        with pytest.warns(DeprecationWarning):
            all_incoming_angles = ray_geometry.inc_angles_list
        assert len(all_incoming_angles) == len(path.interfaces)
        assert all_incoming_angles[0] is None

        assert all_incoming_angles[1].shape == (num_src_points, num_dst_points)
        angles = np.rad2deg(all_incoming_angles[1][0, ...])

        expected_angles = INCOMING_ANGLES[
            dest_points_are_above, are_normals_zplus, use_tilt]

        if show_plots:
            fig, ax = plt.subplots()
            ax.plot(angles, label='actual')
            ax.plot(expected_angles, '--', label='expected')
            ax.set_xlabel('dest point index')
            ax.set_ylabel('incomming angle (deg)')
            ax.set_title(
                "test_incoming_angles\ndest_points_are_above={}, are_normals_zplus={}, use_tilt={}".format(
                    dest_points_are_above, are_normals_zplus, use_tilt))
            ax.legend()
            plt.show()

        np.testing.assert_allclose(angles, expected_angles)

    @pytest.mark.parametrize("dest_points_are_above, are_normals_zplus, use_tilt",
                             [(True, False, False),
                              (False, False, False),
                              (True, True, False),
                              (False, True, False)])
    def test_outgoing_angles(self, show_plots, dest_points_are_above,
                             are_normals_zplus, use_tilt):
        path, ray_geometry = self.make_ray_and_path(dest_points_are_above,
                                                    are_normals_zplus,
                                                    use_tilt)

        num_src_points = len(path.interfaces[0].points)
        num_dst_points = len(path.interfaces[1].points)

        with pytest.warns(DeprecationWarning):
            all_outgoing_angles = ray_geometry.out_angles_list
        assert len(all_outgoing_angles) == len(path.interfaces)
        assert all_outgoing_angles[1] is None

        assert all_outgoing_angles[0].shape == (num_src_points, num_dst_points)
        angles = np.rad2deg(all_outgoing_angles[0][0, ...])

        expected_angles = OUTGOING_ANGLES[
            dest_points_are_above, are_normals_zplus, use_tilt]

        if show_plots:
            fig, ax = plt.subplots()
            ax.plot(angles, label='actual')
            ax.plot(expected_angles, '--', label='expected')
            ax.set_xlabel('dest point index')
            ax.set_ylabel('outgoing angle (deg)')
            ax.set_title(
                "test_outgoing_angles\ndest_points_are_above={}, are_normals_zplus={}, use_tilt={}".format(
                    dest_points_are_above, are_normals_zplus, use_tilt))
            ax.legend()
            plt.show()

        np.testing.assert_allclose(angles, expected_angles)


RAY_GEOMETRY_CASES = ("are_normals_zplus", [True, False])


class TestRayGeometry:
    """
    There are two interfaces: the first one (source) has a unique single point in Omega.
    The second one has 12 points in a circle of centre Omega and radius 5 mm. These points
    are spaced by 30°.

    The coordinate system of the source point is obtained by rotation of 30° around GCS-y
    axis. Therefore its normal points towards the first destination points.

    Use --show-plots in CLI to visualise the set-up. The arrows are the normals Oz+.


    """
    circle_num = 12
    circle_theta = arim.ut.wrap_phase(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    circle_radius = 5e-2

    def make_case(self, are_normals_zplus):
        """
        Source point: omega
        Dest points: 12 points along a circle of centre omega and radius 5 (points are
        spaced by  30°)

        Parameters
        ----------
        are_normals_zplus

        Returns
        -------

        """
        omega = np.array((3., 0., 5.)) * 1e-2
        src_points = g.Points(omega.reshape([1, 3]), name='source')
        src_basis = arim.path.default_orientations(src_points)
        src_basis = src_basis.rotate(g.rotation_matrix_y(np.pi / 6))
        source_interface = arim.Interface(src_points, src_basis,
                                          are_normals_on_out_rays_side=are_normals_zplus)

        # circle:
        dst_points = g.Points(np.zeros([len(self.circle_theta), 3]), name='dest')
        dst_points.x[...] = omega[0]
        dst_points.y[...] = omega[1]
        dst_points.z[...] = omega[2]
        dst_points.x[...] += self.circle_radius * np.sin(self.circle_theta)
        dst_points.z[...] += self.circle_radius * np.cos(self.circle_theta)

        dst_basis = arim.path.default_orientations(dst_points)

        dest_interface = arim.Interface(dst_points, dst_basis,
                                        are_normals_on_inc_rays_side=are_normals_zplus)

        material = arim.Material(np.nan, metadata=dict(long_name='Dummy'))

        interfaces = [source_interface, dest_interface]

        # The i-th ray starts from the source and ends at the i-th destination point.
        shape = [len(source_interface.points), len(dest_interface.points)]
        ray_indices = np.zeros((0, *shape), np.uint)
        times = np.empty(shape, float)
        times.fill(np.nan)

        path = arim.Path(interfaces, [material], ['L'])
        ray = arim.ray.Rays(times, ray_indices, path.to_fermat_path())
        path.rays = ray
        ray_geometry = arim.path.RayGeometry.from_path(path)
        return path, ray_geometry

    def test_ray_geometry_cache(self):
        path, ray_geometry = self.make_case(are_normals_zplus=False)

        assert len(ray_geometry._cache) == 0

        # Case: final result
        leg_points = ray_geometry.leg_points(0, is_final=True)
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_intermediate_results()
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_all_results()
        assert len(ray_geometry._cache) == 0

        # Case: intermediate result
        leg_points = ray_geometry.leg_points(0, is_final=False)
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_intermediate_results()
        assert len(ray_geometry._cache) == 0
        ray_geometry.clear_all_results()
        assert len(ray_geometry._cache) == 0

        # Case: intermediate result promoted to final result
        leg_points = ray_geometry.leg_points(0, is_final=False)
        assert len(ray_geometry._cache) == 1
        # Promotion to final result:
        leg_points = ray_geometry.leg_points(0, is_final=True)
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_intermediate_results()
        assert len(ray_geometry._cache) == 1
        ray_geometry.clear_all_results()
        assert len(ray_geometry._cache) == 0

    def test_plot_cases(self, show_plots):
        if show_plots:
            kwargs = dict(are_normals_zplus=False)
            path, ray_geometry = self.make_case(**kwargs)
            ax = aplt.plot_interfaces(path.interfaces, show_grid=True,
                                      show_orientations=True,
                                      markers=['.', '.'])
            for idx, (x, y, z) in enumerate(path.interfaces[1].points):
                ax.text(x, z, str(idx))
            plt.title('TestRayGeometry.test_plot_cases\nThe arrows are the (Oz+) axes.')
            plt.show()

    def test_precompute(self):
        path, ray_geometry = self.make_case(True)
        with ray_geometry.precompute():
            r1 = ray_geometry.inc_angle(1)
            r2 = ray_geometry.signed_inc_angle(1)
            assert len(ray_geometry._cache) > 2, "no intermediate results is stored"

        assert len(ray_geometry._cache) == 2, "intermediate results aren'ray flushed"

        # Check that caching is effective:
        assert ray_geometry.inc_angle(1) is r1, "caching is not effective"
        assert ray_geometry.signed_inc_angle(1) is r2, "caching is not effective"

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_cache(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        r = ray_geometry.leg_points(1)
        r2 = ray_geometry.leg_points(1)
        assert r is r2

        # there are two interfaces so interfaces[-1] is interfaces[1]
        r3 = ray_geometry.leg_points(-1)
        assert r is r3

        # RayGeometry without cache:
        ray_geometry = arim.path.RayGeometry.from_path(path, use_cache=False)
        r = ray_geometry.leg_points(1)
        r2 = ray_geometry.leg_points(1)
        assert r is not r2

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_leg_points(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        shape = len(path.interfaces[0].points), len(path.interfaces[1].points)
        ray_geometry = arim.path.RayGeometry.from_path(path)

        leg_points = ray_geometry.leg_points(0)
        for point in leg_points:
            np.testing.assert_allclose(point, path.interfaces[0].points[0])

        leg_points = ray_geometry.leg_points(1)
        for i, point in enumerate(leg_points):
            np.testing.assert_allclose(point, path.interfaces[1].points[i])

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_radius(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_radius(0) is None
        radius = ray_geometry.inc_leg_radius(1)
        assert radius.shape == (1, self.circle_num)
        np.testing.assert_allclose(radius, self.circle_radius)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_size(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_size(0) is None
        radius = ray_geometry.inc_leg_radius(1)
        size = ray_geometry.inc_leg_size(1)
        np.testing.assert_allclose(size, radius)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_azimuth(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_azimuth(0) is None
        azimuth = ray_geometry.inc_leg_azimuth(1)
        assert azimuth.shape == (1, self.circle_num)

        # Azimuth is 0 if the source point is in x > 0 (from the dest point),
        # Pi if the source point is in x < 0.
        azimuth = np.squeeze(azimuth)
        for i, theta in enumerate(np.unwrap(self.circle_theta)):
            if np.isclose(theta, 0.) or np.isclose(theta, np.pi):
                # Depends on x=+eps or x=-eps
                assert np.isclose(azimuth[i], 0.) or np.isclose(azimuth[i], np.pi)
            elif 0 < theta < np.pi:
                assert np.isclose(azimuth[i], np.pi)
            else:
                assert np.isclose(azimuth[i], 0.)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_leg_polar(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_leg_azimuth(0) is None
        polar = ray_geometry.inc_leg_polar(1)
        assert polar.shape == (1, self.circle_num)
        expected_polar = np.abs(arim.ut.wrap_phase(self.circle_theta + np.pi))

        np.testing.assert_allclose(np.rad2deg(np.squeeze(polar)),
                                   np.rad2deg(expected_polar))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_inc_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.inc_angle(0) is None

        polar = ray_geometry.inc_leg_polar(1)
        angles = ray_geometry.inc_angle(1)
        np.testing.assert_allclose(np.rad2deg(angles),
                                   np.rad2deg(polar))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_signed_inc_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.signed_inc_angle(0) is None
        angles = ray_geometry.signed_inc_angle(1)
        assert angles.shape == (1, self.circle_num)

        expected_angles = arim.ut.wrap_phase(self.circle_theta + np.pi)
        angles = arim.ut.wrap_phase(np.squeeze(angles))
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_conventional_inc_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.conventional_inc_angle(0) is None
        angles = ray_geometry.conventional_inc_angle(1)
        assert angles.shape == (1, self.circle_num)

        # wrap in [-pi, pi] then takes the abs value
        expected_angles = np.abs(arim.ut.wrap_phase(self.circle_theta + np.pi))
        if not are_normals_zplus:
            expected_angles = np.pi - expected_angles

        angles = np.squeeze(angles)
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_leg_radius(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_leg_radius(1) is None
        radius = ray_geometry.out_leg_radius(0)
        assert radius.shape == (1, self.circle_num)
        np.testing.assert_allclose(radius, self.circle_radius)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_leg_azimuth(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_leg_azimuth(1) is None
        azimuth = ray_geometry.out_leg_azimuth(0)
        assert azimuth.shape == (1, self.circle_num)

        # Azimuth is 0 if the source point is in x > 0 (from the dest point),
        # Pi if the source point is in x < 0.
        azimuth = np.squeeze(azimuth)
        for i, theta in enumerate(np.unwrap(self.circle_theta)):
            if np.isclose(theta, np.pi / 6) or np.isclose(theta, 7 * np.pi / 6):
                # Depends on x=+eps or x=-eps
                assert np.isclose(azimuth[i], 0.) or np.isclose(azimuth[i], np.pi)
            elif np.pi / 6 < theta < 7 * np.pi / 6:
                assert np.isclose(azimuth[i], 0.)
            else:
                assert np.isclose(azimuth[i], np.pi)

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_leg_polar(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_leg_azimuth(1) is None
        polar = ray_geometry.out_leg_polar(0)
        assert polar.shape == (1, self.circle_num)
        expected_polar = np.abs(arim.ut.wrap_phase(self.circle_theta - np.pi / 6))

        np.testing.assert_allclose(np.rad2deg(np.squeeze(polar)),
                                   np.rad2deg(expected_polar))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_out_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.out_angle(1) is None

        polar = ray_geometry.out_leg_polar(0)
        angles = ray_geometry.out_angle(0)
        np.testing.assert_allclose(np.rad2deg(angles),
                                   np.rad2deg(polar))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_signed_out_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.signed_out_angle(1) is None
        angles = ray_geometry.signed_out_angle(0)
        assert angles.shape == (1, self.circle_num)

        expected_angles = np.unwrap(arim.ut.wrap_phase(self.circle_theta - np.pi / 6))
        angles = np.unwrap(arim.ut.wrap_phase(np.squeeze(angles)))

        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))

    @pytest.mark.parametrize(*RAY_GEOMETRY_CASES)
    def test_conventional_out_angle(self, are_normals_zplus):
        path, ray_geometry = self.make_case(are_normals_zplus)

        assert ray_geometry.conventional_out_angle(1) is None
        angles = ray_geometry.conventional_out_angle(0)
        assert angles.shape == (1, self.circle_num)

        # wrap in [-pi, pi] then takes the abs value
        expected_angles = np.abs(arim.ut.wrap_phase(self.circle_theta - np.pi / 6))
        if not are_normals_zplus:
            expected_angles = np.pi - expected_angles

        # angles = np.unwrap(arim.ut.wrap_phase(np.squeeze(angles)))
        angles = np.squeeze(angles)
        np.testing.assert_allclose(np.rad2deg(angles), np.rad2deg(expected_angles))
