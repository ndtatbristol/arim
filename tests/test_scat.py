import numpy as np
import pytest

from arim import ut, scat


def test_scattering_angles_grid():
    n = 10
    theta = scat.make_angles(n)
    inc_theta, out_theta = scat.make_angles_grid(n)
    for i in range(n):
        for j in range(n):
            assert inc_theta[i, j] == theta[j]
            assert out_theta[i, j] == theta[i]


def test_sdh_2d_scat():
    out_theta = np.array(
        [-3.141592653589793, -2.722713633111154, -2.303834612632515, -1.884955592153876,
         -1.466076571675237, -1.047197551196598, -0.628318530717959, -0.209439510239319,
         0.209439510239319, 0.628318530717959, 1.047197551196597, 1.466076571675236,
         1.884955592153876, 2.303834612632516, 2.722713633111154, ])
    inc_theta = 0.
    matlab_res = dict()
    matlab_res['LL'] = np.array([
        -0.206384032591909 + 0.336645038756022j, -0.194171819277630 + 0.313226502544485j,
        -0.155687913654758 + 0.264243478643578j, -0.090375683177214 + 0.226391506237526j,
        -0.005253862284530 + 0.211028560232004j, 0.085889202419455 + 0.204053854945626j,
        0.165030960663520 + 0.193967940239943j, 0.212013086087838 + 0.184664953622806j,
        0.212013086087838 + 0.184664953622806j, 0.165030960663520 + 0.193967940239943j,
        0.085889202419455 + 0.204053854945626j, -0.005253862284530 + 0.211028560232004j,
        -0.090375683177214 + 0.226391506237526j, -0.155687913654758 + 0.264243478643578j,
        -0.194171819277630 + 0.313226502544484j,
    ])
    matlab_res['LT'] = np.array([
        -0.000000000000000 + 0.000000000000000j, 0.173514558396338 - 0.235915394468874j,
        0.363162600270786 - 0.165777746007565j, 0.503786047970495 + 0.061137988770260j,
        0.546299366197900 + 0.133217223565162j, 0.506680725919996 + 0.029760392507310j,
        0.380878711540161 - 0.059504104563334j, 0.145732609209624 - 0.037889115351498j,
        -0.145732609209624 + 0.037889115351498j, -0.380878711540162 + 0.059504104563334j,
        -0.506680725919996 - 0.029760392507310j, -0.546299366197900 - 0.133217223565162j,
        -0.503786047970495 - 0.061137988770261j, -0.363162600270786 + 0.165777746007565j,
        -0.173514558396338 + 0.235915394468874j,
    ])
    matlab_res['TL'] = np.array([
        0.000000000000000 - 0.000000000000000j, -0.043378639599085 + 0.058978848617218j,
        -0.090790650067696 + 0.041444436501891j, -0.125946511992624 - 0.015284497192565j,
        -0.136574841549475 - 0.033304305891291j, -0.126670181479999 - 0.007440098126828j,
        -0.095219677885040 + 0.014876026140834j, -0.036433152302406 + 0.009472278837875j,
        0.036433152302406 - 0.009472278837875j, 0.095219677885040 - 0.014876026140834j,
        0.126670181479999 + 0.007440098126828j, 0.136574841549475 + 0.033304305891291j,
        0.125946511992624 + 0.015284497192565j, 0.090790650067697 - 0.041444436501891j,
        0.043378639599085 - 0.058978848617218j,
    ])
    matlab_res['TT'] = np.array([
        -0.262017703125609 + 0.771353787922999j, -0.376441609988753 + 0.188374651320542j,
        -0.429903377994878 - 0.562524535327520j, -0.243229145424068 - 0.367845549589069j,
        -0.126223795958403 + 0.187399980358998j, -0.136416167137459 + 0.230680597518463j,
        -0.019016110619602 - 0.071187589718864j, 0.181910208019545 - 0.257602560262746j,
        0.181910208019545 - 0.257602560262746j, -0.019016110619602 - 0.071187589718863j,
        -0.136416167137459 + 0.230680597518463j, -0.126223795958403 + 0.187399980358999j,
        -0.243229145424068 - 0.367845549589067j, -0.429903377994878 - 0.562524535327520j,
        -0.376441609988753 + 0.188374651320541j,
    ])
    freq = 2.e6
    v_l = 6000
    v_t = 3000
    hole_radius = 5e-4

    result = scat.sdh_2d_scat(inc_theta, out_theta, freq, hole_radius, v_l, v_t)
    # There is an unexplained -1 multiplicative factor.
    correction = dict()
    correction['LL'] = 1.
    correction['LT'] = -1.
    correction['TL'] = 1.
    correction['TT'] = 1.

    assert len(result) == 4
    assert result['LL'].shape == out_theta.shape
    assert result['LT'].shape == out_theta.shape
    assert result['TL'].shape == out_theta.shape
    assert result['TT'].shape == out_theta.shape

    corr_matlab_res = {key: (correction[key] * val).conjugate()
                       for key, val in matlab_res.items()}

    # thetadeg = np.rad2deg(theta)
    # import matplotlib.pyplot as plt
    # for key in corr_matlab_res:
    #     fig, ax = plt.subplots()
    #     ax.plot(thetadeg, np.abs(result[key]), label='ours')
    #     ax.plot(thetadeg, np.abs(corr_matlab_res[key]), '--', label='matlab')
    #     ax.legend()
    #     ax.set_title('test_elastic_scattering_2d_cylinder - ' + key)
    #     plt.show()

    args = dict(rtol=1e-5)
    np.testing.assert_allclose(result['LL'], corr_matlab_res['LL'], **args)
    np.testing.assert_allclose(result['LT'], corr_matlab_res['LT'], **args)
    np.testing.assert_allclose(result['TL'], corr_matlab_res['TL'], **args)
    np.testing.assert_allclose(result['TT'], corr_matlab_res['TT'], **args)


def test_sdh_2d_scat2():
    shape = (4, 5, 7)
    out_theta = np.random.uniform(low=-np.pi, high=np.pi, size=shape)
    inc_theta = 0.

    freq = 2.e6
    v_l = 6000
    v_t = 3000
    hole_radius = 5e-4
    scat_params = dict(radius=hole_radius, longitudinal_vel=v_l,
                       transverse_vel=v_t)
    scat_params2 = dict(frequency=freq, radius=hole_radius, longitudinal_vel=v_l,
                        transverse_vel=v_t)
    scat_obj = scat.Sdh2dScat(**scat_params)

    result = scat.sdh_2d_scat(inc_theta, out_theta, **scat_params2)

    scat_funcs = scat.scat_2d_cylinder_funcs(**scat_params2)
    for key, scat_func in scat_funcs.items():
        np.testing.assert_allclose(scat_func(inc_theta, out_theta), result[key],
                                   err_msg=key)

    assert set(result.keys()) == {'LL', 'LT', 'TL', 'TT'}
    for key, val in result.items():
        assert val.shape == shape

    to_compute = {'LL', 'TL'}
    result2 = scat.sdh_2d_scat(inc_theta, out_theta,
                               to_compute=to_compute,
                               **scat_params2)
    assert set(result2.keys()) == to_compute
    for key, val in result2.items():
        assert val.shape == shape
        assert np.allclose(val, result[key])


def _scattering_function(inc_theta, out_theta):
    inc_theta = ut.wrap_phase(inc_theta)
    out_theta = ut.wrap_phase(out_theta)
    return (inc_theta + np.pi) / np.pi * 10 + (out_theta + np.pi) / np.pi * 100


def test_make_scattering_matrix():
    numpoints = 5
    inc_theta, out_theta = scat.make_angles_grid(numpoints)
    scattering_matrix = scat.func_to_matrix(_scattering_function, numpoints)

    assert inc_theta.shape == (numpoints, numpoints)
    assert out_theta.shape == (numpoints, numpoints)

    theta = scat.make_angles(numpoints)
    for i in range(numpoints):
        for j in range(numpoints):
            assert np.allclose(scattering_matrix[i, j],
                               _scattering_function(theta[j], theta[i]))

    assert scattering_matrix.shape == (numpoints, numpoints)

    idx = (1, 3)
    np.testing.assert_allclose(scattering_matrix[idx],
                               _scattering_function(inc_theta[idx], out_theta[idx]))

    np.testing.assert_allclose(_scattering_function(np.pi, np.pi),
                               _scattering_function(-np.pi, -np.pi))

    x = 0.1
    np.testing.assert_allclose(_scattering_function(np.pi + x, np.pi),
                               _scattering_function(-np.pi + x, -np.pi))


def test_scattering_interpolate_matrix():
    numpoints = 5
    dtheta = 2 * np.pi / numpoints

    inc_theta, out_theta = scat.make_angles_grid(numpoints)
    scattering_matrix = scat.func_to_matrix(_scattering_function, numpoints)
    scat_fn = scat.interpolate_matrix(scattering_matrix)
    # raise Exception(scattering_matrix)

    np.testing.assert_allclose(scat_fn(inc_theta, out_theta),
                               _scattering_function(inc_theta, out_theta))

    # add multiple of 2pi
    np.testing.assert_allclose(scat_fn(inc_theta + 10 * np.pi, out_theta - 6 * np.pi),
                               _scattering_function(inc_theta, out_theta))

    # remove last column because edge effect
    x = (inc_theta + dtheta / 4)[:, :-1]
    y = out_theta[:, :-1]
    np.testing.assert_allclose(scat_fn(x, y), _scattering_function(x, y))

    # remove last line because edge effect
    x = inc_theta[:-1]
    y = (out_theta + dtheta / 4)[:-1]
    np.testing.assert_allclose(scat_fn(x, y), _scattering_function(x, y))

    x = (inc_theta[:-1, :-1] + dtheta / 3)
    y = (out_theta + dtheta / 4)[:-1, :-1]
    np.testing.assert_allclose(scat_fn(x, y), _scattering_function(x, y))


def test_rotate_scattering_matrix():
    # n = 10
    # scat_matrix = np.random.uniform(size=(n, n)) + 1j * np.random.uniform(size=(n, n))
    # scat_func = ut.interpolate_matrix(scat_matrix)
    n = 72
    inc_angles, out_angles = scat.make_angles_grid(n)
    scat_matrix = (
        np.exp(-(inc_angles - np.pi / 6) ** 2 - (out_angles + np.pi / 4) ** 2)
        + 1j * np.exp(
            -(inc_angles + np.pi / 2) ** 2 - (out_angles - np.pi / 10) ** 2))
    scat_func = scat.interpolate_matrix(scat_matrix)

    # rotation of 0°
    rotated_scat_matrix = scat.rotate_matrix(scat_matrix, 0.)
    np.testing.assert_allclose(scat_matrix, rotated_scat_matrix)

    # rotation of 360°
    rotated_scat_matrix = scat.rotate_matrix(scat_matrix, 2 * np.pi)
    np.testing.assert_allclose(scat_matrix, rotated_scat_matrix, rtol=1e-5)

    # rotation of pi/ 6
    # Ensure that S'(theta_1, theta_2) = S(theta_1 - phi, theta_2 - phi)
    # No interpolation is involded here, this should be perfectly equal
    phi = np.pi / 6
    rotated_scat_matrix = scat.rotate_matrix(scat_matrix, phi)
    rotated_scat_scat_func = scat.interpolate_matrix(rotated_scat_matrix)
    theta_1 = np.linspace(0, 2 * np.pi, n)
    theta_2 = np.linspace(0, np.pi, n)
    np.testing.assert_allclose(rotated_scat_scat_func(theta_1, theta_2),
                               scat_func(theta_1 - phi, theta_2 - phi))

    # rotation of pi/ 5
    # Ensure that S'(theta_1, theta_2) = S(theta_1 - phi, theta_2 - phi)
    # Because of interpolation, this is not exactly equal.
    phi = np.pi / 5
    rotated_scat_matrix = scat.rotate_matrix(scat_matrix, phi)
    rotated_scat_scat_func = scat.interpolate_matrix(rotated_scat_matrix)
    theta_1 = np.linspace(0, 2 * np.pi, 15)
    theta_2 = np.linspace(0, np.pi, 15)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.real(rotated_scat_scat_func(theta_1, theta_2)))
    # plt.plot(np.real(scat_func(theta_1 - phi, theta_2 - phi)))
    # plt.show()
    # plt.figure()
    # plt.plot(np.imag(rotated_scat_scat_func(theta_1, theta_2)))
    # plt.plot(np.imag(scat_func(theta_1 - phi, theta_2 - phi)))
    # plt.show()

    np.testing.assert_allclose(rotated_scat_scat_func(theta_1, theta_2),
                               scat_func(theta_1 - phi, theta_2 - phi), atol=5e-3)

    # unrotate
    np.testing.assert_allclose(
        scat.rotate_matrix(rotated_scat_matrix, -phi),
        scat_matrix)


@pytest.fixture(params=['sdh', 'point'])
def scat_obj(request):
    if request.param == 'sdh':
        hole_radius = 5e-4
        return scat.Sdh2dScat(hole_radius, TestScattering.v_L, TestScattering.v_T)
    elif request.param == 'point':
        return scat.PointSourceScat(TestScattering.v_L, TestScattering.v_T)
    else:
        raise Exception('this fixture does not behave well')


class TestScattering:
    v_L = 6300.
    v_T = 3100.
    density = 2700.

    def test_scattering(self, scat_obj):
        numangles = 7
        n, m = 9, 11
        phi_in = scat.make_angles(n)
        phi_out = scat.make_angles(m)

        phi_in_array, phi_out_array = np.meshgrid(phi_in, phi_out, indexing='xy')
        assert phi_in_array.shape == phi_out_array.shape == (m, n)

        scat_keys = ('LL', 'LT', 'TL', 'TT')

        freq = 2e6

        # test Scattering.__call__, Scattering.__str__
        repr(scat_obj)
        str(scat_obj)

        # test Scattering.__call__ with 0d array angles
        val_dict = scat_obj(0.1, 0.2, freq)
        assert set(val_dict.keys()) == set(scat_keys)
        for val in val_dict.values():
            assert np.ndim(val) == 0

        # test Scattering.__call__ with 1d array angles
        val_dict = scat_obj(phi_in, phi_in, freq)
        for val in val_dict.values():
            assert val.shape == phi_in.shape

        # test Scattering.__call__ with 2d array angles
        reference_dict = scat_obj(phi_in_array, phi_out_array, freq)
        for val in reference_dict.values():
            assert val.shape == phi_in_array.shape

        # test broadcasting works well
        for idx in np.ndindex(*phi_in_array.shape):
            val_dict = scat_obj(phi_in_array[idx], phi_out_array[idx], freq)
            for scat_key in scat_keys:
                assert val_dict[scat_key] == reference_dict[scat_key][idx]

        # computing the values for one scat_key
        for scat_key in scat_keys:
            val_dict = scat_obj(phi_in_array, phi_out_array, freq, to_compute=[scat_key])
            assert scat_key in val_dict
            np.testing.assert_allclose(val_dict[scat_key], reference_dict[scat_key],
                                       err_msg='different output for the scat_key')

        # test Scattering.as_single_freq_matrices
        matrices_singlef = scat_obj.as_single_freq_matrices(freq, numangles)
        for scat_key in scat_keys:
            mat = matrices_singlef[scat_key]
            assert mat.shape == (numangles, numangles)
            assert mat[0, 0] == reference_dict[scat_key][0, 0], \
                'different values for phi_in = phi_out = -pi ({})'.format(scat_key)

        # test Scattering.as_multi_freq_matrices (use 2 frequencies)
        matrices_multif = scat_obj.as_multi_freq_matrices([freq, 2 * freq], numangles)
        for scat_key in scat_keys:
            mat = matrices_multif[scat_key]
            assert mat.shape == (2, numangles, numangles)
            np.testing.assert_allclose(mat[0], matrices_singlef[scat_key],
                                       err_msg='different output for the same frequency')
            assert mat[0, 0, 0] == reference_dict[scat_key][0, 0], \
                'different values for phi_in = phi_out = -pi ({})'.format(scat_key)

    def test_reciprocity(self, scat_obj, show_plots):
        numangles = 20
        freq = 2e6
        matrices = scat_obj.as_single_freq_matrices(freq, numangles)
        LT = matrices['LT']
        TL = matrices['TL']

        lhs_func = lambda x: self.v_T ** 2 * x
        rhs_func = lambda x: -self.v_L ** 2 * x

        if show_plots:
            import matplotlib.pyplot as plt
            plt.figure()
            idx = 1
            lhs = lhs_func(LT[idx, :])
            rhs = rhs_func(TL[:, idx])
            plt.subplot(211)
            plt.plot(lhs.real)
            plt.plot(rhs.real)
            plt.title(repr(scat_obj))
            plt.subplot(212)
            plt.plot(lhs.imag)
            plt.plot(rhs.imag)
            plt.show()

        lhs = lhs_func(LT)
        rhs = rhs_func(TL.T)
        max_error = np.max(np.abs(lhs - rhs))
        median_error = np.median(np.abs(lhs - rhs))
        np.testing.assert_allclose(
            lhs, rhs, err_msg='no reciprocity - maxerror = {}, median error = {}'.format(
                max_error, median_error), rtol=1e-7, atol=1e-8)
