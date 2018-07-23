import warnings

import numpy as np
import pytest

import arim
from arim import ut, scat, _scat_crack
import tests.helpers


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
    matlab_res['LT'] *= -1  # trust Lopez-Sanchez instead of Brind, different from matlab implementation

    assert len(result) == 4
    assert result['LL'].shape == out_theta.shape
    assert result['LT'].shape == out_theta.shape
    assert result['TL'].shape == out_theta.shape
    assert result['TT'].shape == out_theta.shape

    args = dict(rtol=1e-5)
    np.testing.assert_allclose(result['LL'], matlab_res['LL'], **args)
    np.testing.assert_allclose(result['LT'], matlab_res['LT'], **args)
    np.testing.assert_allclose(result['TL'], matlab_res['TL'], **args)
    np.testing.assert_allclose(result['TT'], matlab_res['TT'], **args)


def _scattering_function(inc_theta, out_theta):
    inc_theta = ut.wrap_phase(inc_theta)
    out_theta = ut.wrap_phase(out_theta)
    return (inc_theta + np.pi) / np.pi * 10 + (out_theta + np.pi) / np.pi * 100


def test_make_scattering_matrix():
    numpoints = 5
    inc_theta, out_theta = scat.make_angles_grid(numpoints)
    scattering_matrix = _scattering_function(inc_theta, out_theta)


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


def test_interpolate_matrix():
    numpoints = 5
    dtheta = 2 * np.pi / numpoints

    inc_theta, out_theta = scat.make_angles_grid(numpoints)
    scattering_matrix = _scattering_function(inc_theta, out_theta)

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


def test_rotate_matrix():
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

    # test rotate_matrices
    matrices = dict(LL=scat_matrix, TT=np.ones((10, 10)))
    rotated_matrices = scat.rotate_matrices(matrices, np.pi / 6)
    assert 'LL' in rotated_matrices
    assert 'TT' in rotated_matrices


def make_scat_data_single_freq():
    # create realistic data
    hole_radius = 5e-4
    scat_sdh = scat.SdhScat(hole_radius, TestScattering.v_L, TestScattering.v_T)
    numangles = 80
    frequency = 2e6
    matrices = scat_sdh.as_single_freq_matrices(frequency, numangles)
    matrices2 = {scat_key: mat[np.newaxis] for scat_key, mat in matrices.items()}
    return scat.ScatFromData.from_dict(frequency, matrices2)


def test_scat_factory():
    material = arim.Material(6300., 3120., 2700., 'solid', metadata={'long_name': 'Aluminium'})

    fname = tests.helpers.get_data_filename("scat/scat_matlab.mat")
    scat_obj = scat.scat_factory('file', material, fname)
    assert isinstance(scat_obj, scat.ScatFromData)
    scat_obj(0., 0., 2e6)

    scat_obj = scat.scat_factory('crack_centre', material, crack_length=2.0e-3)
    assert isinstance(scat_obj, scat.CrackCentreScat)
    scat_obj(0., 0., 2e6)

    scat_obj = scat.scat_factory('sdh', material, radius=0.5e-3)
    assert isinstance(scat_obj, scat.SdhScat)
    scat_obj(0., 0., 2e6)

    scat_obj = scat.scat_factory('point', material)
    assert isinstance(scat_obj, scat.PointSourceScat)
    scat_obj(0., 0., 2e6)

    scat_obj = scat.scat_factory('crack_tip', material)
    assert isinstance(scat_obj, scat.CrackTipScat)
    scat_obj(0., 0., 2e6)


def make_scat_data_multi_freq():
    # create realistic data
    hole_radius = 5e-4
    scat_sdh = scat.SdhScat(hole_radius, TestScattering.v_L, TestScattering.v_T)
    frequencies = [1e6, 2e6, 3e6]
    numangles = 80
    matrices = scat_sdh.as_multi_freq_matrices(frequencies, numangles)
    return scat.ScatFromData.from_dict(frequencies, matrices)


# TODO: add test 'crack_tip' once properly implemented
@pytest.fixture(params=['sdh', 'point', 'data_singlefreq', 'data_multifreq',
                        'crack_centre'])
def scat_obj(request):
    if request.param == 'sdh':
        hole_radius = 5e-4
        return scat.SdhScat(hole_radius, TestScattering.v_L, TestScattering.v_T)
    elif request.param == 'point':
        return scat.PointSourceScat(TestScattering.v_L, TestScattering.v_T)
    elif request.param == 'data_singlefreq':
        return make_scat_data_single_freq()
    elif request.param == 'data_multifreq':
        return make_scat_data_multi_freq()
    elif request.param == 'crack_centre':
        crack_length = 2.e-3
        return scat.CrackCentreScat(crack_length, TestScattering.v_L, TestScattering.v_T,
                                    TestScattering.density)
    elif request.param == 'crack_tip':
        return scat.CrackTipScat(TestScattering.v_L, TestScattering.v_T)
    else:
        raise Exception('this fixture does not behave well')


@pytest.fixture(params=['data_singlefreq', 'data_multifreq'])
def scat_data_obj(request):
    if request.param == 'data_singlefreq':
        return make_scat_data_single_freq()
    elif request.param == 'data_multifreq':
        return make_scat_data_multi_freq()
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

        # test Scattering.__call__ with broadcast
        val_dict = scat_obj(0., phi_out, freq)
        val_dict2 = scat_obj(np.zeros_like(phi_out), phi_out, freq)
        for scat_key in scat_keys:
            np.testing.assert_allclose(val_dict[scat_key], val_dict2[scat_key])

        val_dict = scat_obj(0., phi_out_array, freq)
        val_dict2 = scat_obj(np.zeros_like(phi_out_array), phi_out_array, freq)
        for scat_key in scat_keys:
            np.testing.assert_allclose(val_dict[scat_key], val_dict2[scat_key])

        # test Scattering.__call__ with 2d array angles
        reference_dict = scat_obj(phi_in_array, phi_out_array, freq)
        for val in reference_dict.values():
            assert val.shape == phi_in_array.shape

        # test broadcasting works well
        for idx in np.ndindex(*phi_in_array.shape):
            val_dict = scat_obj(phi_in_array[idx], phi_out_array[idx], freq)
            for scat_key in scat_keys:
                if np.isnan(reference_dict[scat_key][idx]):
                    continue
                np.testing.assert_allclose(val_dict[scat_key], reference_dict[scat_key][idx])

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
        with warnings.catch_warnings():
            if isinstance(scat_obj, scat.ScatFromData):
                # ignore a legitimate warning (frequency extrapolation)
                warnings.filterwarnings("ignore", category=arim.exceptions.ArimWarning)
            matrices_multif = scat_obj.as_multi_freq_matrices([freq, 2 * freq], numangles)
        for scat_key in scat_keys:
            mat = matrices_multif[scat_key]
            assert mat.shape == (2, numangles, numangles)
            np.testing.assert_allclose(mat[0], matrices_singlef[scat_key],
                                       err_msg='different output for the same frequency')
            assert mat[0, 0, 0] == reference_dict[scat_key][0, 0], \
                'different values for phi_in = phi_out = -pi ({})'.format(scat_key)

        # test Scattering.as_angles_funcs and Scattering.as_freq_angles_funcs
        angles_funcs = scat_obj.as_angles_funcs(freq)
        freq_angles_funcs = scat_obj.as_freq_angles_funcs()
        for scat_key in scat_keys:
            x = angles_funcs[scat_key](2., 3.)
            y = freq_angles_funcs[scat_key](2., 3., freq)
            assert x == y

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
            plt.title(scat_obj.__class__.__name__)
            plt.subplot(212)
            plt.plot(lhs.imag)
            plt.plot(rhs.imag)
            plt.show()

        lhs = lhs_func(LT)
        rhs = rhs_func(TL.T)
        max_error = np.nanmax(np.abs(lhs - rhs))
        median_error = np.nanmedian(np.abs(lhs - rhs))
        np.testing.assert_allclose(
            lhs, rhs, err_msg='no reciprocity - maxerror = {}, median error = {}'.format(
                max_error, median_error), rtol=1e-7, atol=1e-8)

    def test_scat_data(self, scat_data_obj):
        scat_keys = scat.SCAT_KEYS

        scat_obj = scat_data_obj  # alias

        assert scat_obj.frequencies.ndim == 1
        assert scat_obj.numfreq == scat_obj.frequencies.shape[0]

        numangles = scat_obj.numangles

        # asking values at the known values must be the known values
        matrices = scat_obj.as_multi_freq_matrices(scat_obj.frequencies, numangles)
        for scat_key in scat_keys:
            np.testing.assert_allclose(matrices[scat_key],
                                       scat_obj.orig_matrices[scat_key], atol=1e-14)

        # another way to ask the known values
        frequency = scat_obj.frequencies[0]
        phi_in, phi_out = scat.make_angles_grid(numangles)
        matrices = scat_obj(phi_in, phi_out, frequency)
        np.testing.assert_allclose(matrices[scat_key],
                                   scat_obj.orig_matrices[scat_key][0], atol=1e-14)

        # check angle interpolation
        phi = phi_in[0]
        a, b = 0.25, 0.75
        matrices = scat_obj([phi[0], phi[1], a * phi[0] + b * phi[1]],
                            [phi[2], phi[2], phi[2]], frequency)
        for scat_key in scat_keys:
            val = matrices[scat_key]
            np.testing.assert_allclose(a * val[0] + b * val[1], val[2])

        # check frequency interpolation
        if scat_obj.numfreq == 1:
            # This should be the same value all the time
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=arim.exceptions.ArimWarning)

                funcs = scat_obj.as_freq_angles_funcs()
                for scat_key, func in funcs.items():
                    assert func(2., 3., frequency) == func(2., 3., frequency * 2.)
                    assert func(2., 3., frequency) == func(2., 3., frequency / 2.)
        else:
            # This should be a linear interpolation
            funcs = scat_obj.as_freq_angles_funcs()
            for scat_key, func in funcs.items():
                x = (a * func(2., 3., scat_obj.frequencies[0]) +
                     b * func(2., 3., scat_obj.frequencies[1]))
                y = func(2., 3.,
                         a * scat_obj.frequencies[0] + b * scat_obj.frequencies[1])
                np.testing.assert_allclose(x, y)

            # extrapolation: there should be not bound error
            func(2., 3., np.min(scat_obj.frequencies) / 2)
            func(2., 3., np.max(scat_obj.frequencies) * 2)

    def test_scat_data2(self):
        n = 10
        # create obj with incomplete matrices
        scat.ScatFromData([2e6], scat_matrix_TT=np.ones((1, n, n)))

        # no data
        with pytest.raises(ValueError):
            scat.ScatFromData([2e6])

        # wrong shapes
        with pytest.raises(ValueError):
            scat.ScatFromData([2e6, 4e6], np.ones((n, n)))
        with pytest.raises(ValueError):
            scat.ScatFromData([2e6, 4e6], np.ones((2, n, n + 1)))
        # ok:
        scat.ScatFromData([2e6, 4e6], np.ones((2, n, n)))


def test_crack_2d_scat():
    """
    Compare the results of the python impletation to the original matlab one

    October 2017
    git hash of ndt-library : eb8192a2b54613d0c2f0f79be09ca166bb8d935c

    """
    np.testing.assert_allclose(_scat_crack.basis_function(0.2),
                               0.997779793163178)
    np.testing.assert_allclose(_scat_crack.basis_function(0.02),
                               0.999977777979798)
    np.testing.assert_allclose(_scat_crack.basis_function(-5e-4),
                               0.999999986111111)
    np.testing.assert_allclose(_scat_crack.basis_function(-5), 0.193049319258009)

    np.testing.assert_allclose(_scat_crack.sigma(3, 0), 3.)
    np.testing.assert_allclose(_scat_crack.sigma(-3, 0), 3.)
    np.testing.assert_allclose(_scat_crack.sigma(0, 3), -3j)
    np.testing.assert_allclose(_scat_crack.sigma(0, -3), -3j)

    np.testing.assert_allclose(_scat_crack.F(100., 20., .1, 10.), -
    6.201991005820587e-05 - 6.201834393840944e-04j)
    np.testing.assert_allclose(_scat_crack.F(100., 20., .1, 100.),
                               -1.134154772479002e-07)

    np.testing.assert_allclose(_scat_crack.P(0.2),
                               0.916975649032288 + 0.387691081895481j)
    np.testing.assert_allclose(_scat_crack.P(0.02),
                               0.999155676046222 + 0.039987556345679j)
    np.testing.assert_allclose(_scat_crack.P(-5e-4),
                               0.999999472222269 - 0.000999999805556j)
    np.testing.assert_allclose(_scat_crack.P(-5),
                               -0.031270551028216 + 0.020274600339756j)

    A_x_matlab = np.array([
        [
            11.611147572274902 - 0.411760282576216j,
            -4.167624006645408 - 0.387858246147355j,
            -0.975637927872947 - 0.321211994499892j,
            -0.294251905702566 - 0.225789065105809j],
        [
            -4.167624006645408 - 0.387858246147355j,
            11.611147572274902 - 0.411760282576216j,
            -4.167624006645408 - 0.387858246147355j,
            -0.975637927872947 - 0.321211994499892j],
        [
            -0.975637927872947 - 0.321211994499892j,
            -4.167624006645408 - 0.387858246147355j,
            11.611147572274902 - 0.411760282576216j,
            -4.167624006645408 - 0.387858246147355j],
        [
            -0.294251905702566 - 0.225789065105809j,
            -0.975637927872947 - 0.321211994499892j,
            -4.167624006645408 - 0.387858246147355j,
            11.611147572274902 - 0.411760282576216j],

    ])
    xi = 0.5
    xi2 = 1e4
    h_nodes = 5e-5
    num_nodes = 4
    A_x = _scat_crack.A_x(xi, xi2, h_nodes, num_nodes)
    np.testing.assert_allclose(A_x, A_x_matlab, rtol=1e-5)

    A_z_matlab = np.array([
        [
            10.831364548525023 - 0.852510531964943j,
            -4.600889012837459 - 0.823760906489351j,
            -1.153402642894143 - 0.741329549593963j,
            -0.321727351393225 - 0.616174938007808j,
        ],
        [
            -4.600889012837459 - 0.823760906489351j,
            10.831364548525023 - 0.852510531964943j,
            -4.600889012837459 - 0.823760906489351j,
            -1.153402642894143 - 0.741329549593963j,
        ],
        [
            -1.153402642894143 - 0.741329549593963j,
            -4.600889012837459 - 0.823760906489351j,
            10.831364548525023 - 0.852510531964943j,
            -4.600889012837459 - 0.823760906489351j,
        ],
        [
            -0.321727351393225 - 0.616174938007808j,
            -1.153402642894143 - 0.741329549593963j,
            -4.600889012837459 - 0.823760906489351j,
            10.831364548525023 - 0.852510531964943j
        ]
    ])
    A_z = _scat_crack.A_z(xi, xi2, h_nodes, num_nodes)
    np.testing.assert_allclose(A_z, A_z_matlab, rtol=1e-5)

    inc_theta = np.linspace(-np.pi, np.pi, 3, endpoint=False)
    out_theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    inc_theta_arr, out_theta_arr = np.meshgrid(inc_theta, out_theta, indexing='xy')

    frequency = 5e6
    v_L = 6300.
    v_T = 3100.
    density = 2700.
    crack_length = 2.0e-3
    scat_vals = scat.crack_2d_scat(inc_theta_arr, out_theta_arr, frequency, crack_length,
                                   v_L, v_T, density)
    scat_vals2 = scat.crack_2d_scat(inc_theta_arr, out_theta_arr, frequency, crack_length,
                                    v_L, v_T, density, assume_safe_for_opt=True)
    for scat_key in scat.SCAT_KEYS:
        np.testing.assert_allclose(scat_vals[scat_key], scat_vals2[scat_key],
                                   err_msg="different values depending on 'assume_safe_for_opt'")

    S_LL_matlab = np.array([
        [-1.0699 + 1.2424j, 0.12384 - 0.098549j, -0.082018 - 0.0029527j,
         -0.082018 - 0.0029527j, 0.12384 - 0.098549j],
        [0.13852 - 0.1516j, 0.027951 + 0.00065141j, -0.06977 + 0.11084j,
         -0.29307 + 0.25468j, -0.47499 + 0.45202j],
        [0.13852 - 0.1516j, -0.47499 + 0.45202j, -0.29307 + 0.25468j,
         -0.06977 + 0.11084j, 0.027951 + 0.00065141j],
    ])
    S_LT_matlab = np.array([
        [2.8488e-16 - 3.0033e-16j, -0.1543 + 0.48485j, 0.13084 + 0.29801j,
         -0.13084 - 0.29801j, 0.1543 - 0.48485j],
        [-0.21021 + 0.42328j, 0.15 - 0.039526j, 0.27275 - 0.12168j, 1.2591 - 1.024j,
         0.44211 - 0.093854j],
        [0.21021 - 0.42328j, -0.44211 + 0.093854j, -1.2591 + 1.024j,
         -0.27275 + 0.12168j, -0.15 + 0.039526j],
    ])
    S_TL_matlab = np.array([
        [6.8978e-17 - 7.2717e-17j, 0.040729 - 0.069972j, 0.032289 + 0.010542j,
         -0.032289 - 0.010542j, -0.040729 + 0.069972j],
        [-0.0055569 - 0.038185j, -0.028158 + 0.0042409j, 0.013384 + 0.020158j,
         -0.008903 + 0.053993j, 0.10481 - 0.040638j],
        [0.0055569 + 0.038185j, -0.10481 + 0.040638j, 0.008903 - 0.053993j,
         -0.013384 - 0.020158j, 0.028158 - 0.0042409j],
    ])
    S_TT_matlab = np.array([
        [-2.0549 + 2.6393j, -0.030879 + 0.045921j, 0.15293 - 0.16031j,
         0.15293 - 0.16031j, -0.030879 + 0.045921j],
        [0.0041305 - 0.089707j, 0.11045 - 0.16748j, -0.00979 - 0.13942j,
         -0.13575 - 0.037302j, -1.3597 + 1.5069j],
        [0.0041305 - 0.089707j, -1.3597 + 1.5069j, -0.13575 - 0.037302j,
         -0.00979 - 0.13942j, 0.11045 - 0.16748j],
    ])

    # add a bit of tolerance due to the error in numerical integration
    tol = dict(rtol=1e-3)
    # Tranpose because the matlab data uses the convention [phi_in, phi_out] instead
    # of [phi_out, phi_in]
    # Minus sign for different polarisation of shear wave.
    np.testing.assert_allclose(scat_vals['LL'].T, S_LL_matlab, err_msg='LL', **tol)
    np.testing.assert_allclose(scat_vals['LT'].T, S_LT_matlab, err_msg='LT', **tol)
    np.testing.assert_allclose(-scat_vals['TL'].T, S_TL_matlab, err_msg='TL', **tol)
    np.testing.assert_allclose(-scat_vals['TT'].T, S_TT_matlab, err_msg='TT', **tol)

    # check the Scattering2d object give the same answers
    scat_obj = scat.CrackCentreScat(crack_length, v_L, v_T, density)
    scat_vals2 = scat_obj(inc_theta_arr, out_theta_arr, frequency)
    for scat_key in scat.SCAT_KEYS:
        np.testing.assert_allclose(scat_vals[scat_key], scat_vals2[scat_key])

    scat_vals = scat.crack_2d_scat(inc_theta_arr, out_theta_arr, frequency, crack_length,
                                   v_L, v_T, density, to_compute=('LL', 'LT'))
    np.testing.assert_allclose(scat_vals['LL'].T, S_LL_matlab, err_msg='LL', **tol)
    np.testing.assert_allclose(scat_vals['LT'].T, S_LT_matlab, err_msg='LT', **tol)
    np.testing.assert_allclose(-scat_vals['TL'].T, 0., err_msg='TL', **tol)
    np.testing.assert_allclose(-scat_vals['TT'].T, 0., err_msg='TT', **tol)

    scat_vals = scat.crack_2d_scat(inc_theta_arr, out_theta_arr, frequency, crack_length,
                                   v_L, v_T, density, to_compute=('TL', 'TT'))
    np.testing.assert_allclose(scat_vals['LL'].T, 0., err_msg='LL', **tol)
    np.testing.assert_allclose(scat_vals['LT'].T, 0., err_msg='LT', **tol)
    np.testing.assert_allclose(-scat_vals['TL'].T, S_TL_matlab, err_msg='TL', **tol)
    np.testing.assert_allclose(-scat_vals['TT'].T, S_TT_matlab, err_msg='TT', **tol)
