import pytest
import arim
import numpy as np
import matplotlib.pyplot as plt


class TestRayGeometry:
    """
    Source point: O(0., 0., 0.)

    Dest points 'above': line y = 0 and z = +1.
    Dest points 'below': line y = 0 and z = -1.

    The incoming angle is a function of the polar angle of the source point of the leg in the
    coordinate system of the interface point. This angle is the polar angle when the normal
    is not flipped, 180° minus the polar angle when the normal is flipped.

    """
    INCOMING_ANGLES = np.array([78.69006752597979, 76.86597769360367,
                                74.35775354279127, 70.70995378081126,
                                64.98310652189996, 55.00797980144132,
                                35.537677791974374, 0.0,
                                35.537677791974374,
                                55.007979801441344, 64.98310652189998,
                                70.70995378081126, 74.35775354279127,
                                76.86597769360367, 78.69006752597979])

    def make_ray_and_path(self, dest_points_are_above, normals_are_flipped):
        source_interface = arim.Interface(
            *arim.path.points_1d_wall_z(0., 0., 0., 1, name='source'),
            are_normals_on_out_rays_side=normals_are_flipped)

        if dest_points_are_above:
            z = 1.
        else:
            z = -1.

        xmin = -5.
        xmax = 5.
        numpoints = 15

        dest_interface = arim.Interface(
            *arim.path.points_1d_wall_z(xmin, xmax, z, numpoints, name='dest'),
            are_normals_on_inc_rays_side=normals_are_flipped)

        material = arim.Material(np.nan, metadata=dict(long_name='Dummy'))

        interfaces = [source_interface, dest_interface]

        # The i-th ray starts from the source and ends at the i-th destination point.
        shape = [len(source_interface.points), len(dest_interface.points)]
        ray_indices = np.zeros((0, *shape), np.uint)
        times = np.empty(shape, float)
        times.fill(np.nan)

        path = arim.Path(interfaces, [material], ['L'])
        ray = arim.im.Rays(times, ray_indices, path.to_fermat_path())
        return path, ray

    @pytest.mark.parametrize("dest_points_are_above, normals_are_flipped",
                             [(True, False),
                              (False, False),
                              (True, True),
                              (False, True)])
    def test_incoming_angles(self, show_plots, dest_points_are_above,
                             normals_are_flipped):
        path, ray = self.make_ray_and_path(dest_points_are_above, normals_are_flipped)

        num_src_points = len(path.interfaces[0].points)
        num_dst_points = len(path.interfaces[1].points)

        all_incoming_angles = list(ray.get_incoming_angles(path.interfaces))
        assert len(all_incoming_angles) == len(path.interfaces)
        assert all_incoming_angles[0] is None

        assert all_incoming_angles[1].shape == (num_src_points, num_dst_points)
        angles = np.rad2deg(all_incoming_angles[1][0, ...])

        # For the case dest_points_are_above=True and normals_are_flipped=False
        if dest_points_are_above:
            expected_angles = self.INCOMING_ANGLES
        else:
            expected_angles = 180 - self.INCOMING_ANGLES
        if normals_are_flipped:
            expected_angles = 180 - expected_angles
        else:
            expected_angles = expected_angles

        if show_plots:
            fig, ax = plt.subplots()
            ax.plot(angles, label='actual')
            ax.plot(expected_angles, '--', label='expected')
            ax.set_xlabel('dest point index')
            ax.set_ylabel('incomming angle (deg)')
            ax.set_title(
                "test_incoming_angles\ndest_points_are_above={}, normals_are_flipped={}".format(
                    dest_points_are_above, normals_are_flipped))
            ax.legend()
            plt.show()

        np.testing.assert_allclose(angles, expected_angles)

    @pytest.mark.parametrize("dest_points_are_above, normals_are_flipped",
                             [(True, False),
                              (False, False),
                              (True, True),
                              (False, True)])
    def test_outgoing_angles(self, show_plots, dest_points_are_above,
                             normals_are_flipped):
        path, ray = self.make_ray_and_path(dest_points_are_above, normals_are_flipped)

        num_src_points = len(path.interfaces[0].points)
        num_dst_points = len(path.interfaces[1].points)

        all_outgoing_angles = list(ray.get_outgoing_angles(path.interfaces))
        assert len(all_outgoing_angles) == len(path.interfaces)
        assert all_outgoing_angles[1] is None

        assert all_outgoing_angles[0].shape == (num_src_points, num_dst_points)
        angles = np.rad2deg(all_outgoing_angles[0][0, ...])

        # For the case dest_points_are_above=True and normals_are_flipped=False
        if dest_points_are_above:
            expected_angles = 180. - self.INCOMING_ANGLES
        else:
            expected_angles = self.INCOMING_ANGLES
        if normals_are_flipped:
            expected_angles = 180. - expected_angles
        else:
            expected_angles = expected_angles

        if show_plots:
            fig, ax = plt.subplots()
            ax.plot(angles, label='actual')
            ax.plot(expected_angles, '--', label='expected')
            ax.set_xlabel('dest point index')
            ax.set_ylabel('outgoing angle (deg)')
            ax.set_title(
                "test_outgoing_angles\ndest_points_are_above={}, normals_are_flipped={}".format(
                    dest_points_are_above, normals_are_flipped))
            ax.legend()
            plt.show()

        np.testing.assert_allclose(angles, expected_angles)
