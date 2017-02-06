import numpy as np
import pytest

from arim import ut


def test_decibel():
    db = ut.decibel(0.01, reference=1.)
    assert np.allclose(db, -40.)

    arr = np.array([0.01, 0.1, 1., 10.])
    db = ut.decibel(arr)
    assert np.allclose(db, [-60., -40., -20., 0.])

    arr = np.array([0.01, 0.1, 1., 10.])
    db = ut.decibel(arr, reference=1.)
    assert np.allclose(db, [-40., -20., 0., 20.])

    arr = np.array([0.01, 0.1, 1., 10.])
    db, ref = ut.decibel(arr, return_reference=True)
    assert np.allclose(db, [-60., -40., -20., 0.])
    assert np.isclose(ref, 10.)

    arr = np.array([0.01, 0.1, 1., 10., np.nan])
    with np.errstate(all='raise'):
        db = ut.decibel(arr)
    assert np.isnan(db[-1])
    assert np.allclose(db[:-1], [-60., -40., -20., 0.])

    # Check argument neginf_values:
    arr = np.array([0., 0.01, 0.1, 1., 10., np.nan])
    with np.errstate(all='raise'):
        db = ut.decibel(arr, neginf_value=-666.)
    assert np.isnan(db[-1])
    assert np.allclose(db[:-1], [-666., -60., -40., -20., 0.])

    arr = np.array([0., 0.01, 0.1, 1., 10., np.nan])
    with np.errstate(all='raise'):
        db = ut.decibel(arr, neginf_value=None)
    assert np.isnan(db[-1])
    assert np.isneginf(db[0])
    assert np.allclose(db[1:-1], [-60., -40., -20., 0.])


def test_fmc():
    numelements = 3
    tx2 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]

    tx, rx = ut.fmc(numelements)

    shape = (numelements * numelements,)

    assert tx.shape == shape
    assert rx.shape == shape
    assert np.all(tx == tx2)
    assert np.all(rx == rx2)


def test_hmc():
    numelements = 3
    tx2 = [0, 0, 0, 1, 1, 2]
    rx2 = [0, 1, 2, 1, 2, 2]

    tx, rx = ut.hmc(numelements)

    shape = (numelements * (numelements + 1) / 2,)

    assert tx.shape == shape
    assert rx.shape == shape
    assert np.all(tx == tx2)
    assert np.all(rx == rx2)


def test_infer_capture_method():
    # Valid HMC
    tx = [0, 0, 0, 1, 1, 2]
    rx = [0, 1, 2, 1, 2, 2]
    assert ut.infer_capture_method(tx, rx) == 'hmc'

    # HMC with duplicate signals
    tx = [0, 0, 0, 1, 1, 2, 1]
    rx = [0, 1, 2, 1, 2, 2, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # HMC with missing signals
    tx = [0, 0, 0, 2, 1]
    rx = [0, 1, 2, 2, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # Valid HMC
    tx = [0, 1, 2, 1, 2, 2]
    rx = [0, 0, 0, 1, 1, 2]
    assert ut.infer_capture_method(tx, rx) == 'hmc'

    # Something weird
    tx = [0, 1, 2, 1, 2, 2]
    rx = [0, 0, 0, 1]
    with pytest.raises(Exception):
        ut.infer_capture_method(tx, rx)

    # Valid FMC
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert ut.infer_capture_method(tx, rx) == 'fmc'

    # FMC with duplicate signals
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # FMC with missing signals
    tx = [0, 0, 0, 1, 1, 1, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # Negative values
    tx = [0, -1]
    rx = [0, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'

    # Weird
    tx = [0, 5]
    rx = [0, 1]
    assert ut.infer_capture_method(tx, rx) == 'unsupported'


def test_instantaneous_phase_shift():
    t = np.arange(300)
    f0 = 20
    theta = np.pi / 3
    sig = 12. * np.exp(1j * (2. * np.pi * f0 * t + theta))

    theta_computed = ut.instantaneous_phase_shift(sig, t, f0)
    np.testing.assert_allclose(theta_computed, theta)

    with pytest.warns(ut.UtWarning):
        theta_computed = ut.instantaneous_phase_shift(sig.real, t, f0)


def test_wrap_phase():
    res_phases = [
        # unwrapped, wrapped
        (np.pi, -np.pi),
        (-np.pi, -np.pi),
        (4.5 * np.pi, 0.5 * np.pi),
        (3.5 * np.pi, -0.5 * np.pi),
        (-4.5 * np.pi, -0.5 * np.pi),
    ]
    unwrapped, wrapped = zip(*res_phases)
    np.testing.assert_allclose(ut.wrap_phase(unwrapped), wrapped)


def test_directivity_directivity_finite_width_2d():
    theta = 0.
    element_width = 1e-3
    wavelength = 0.5e-3
    directivity = ut.directivity_finite_width_2d(theta, element_width, wavelength)

    assert np.isclose(directivity, 1.0)

    # From the NDT library (2016/03/22):
    # >>> fn_calc_directivity_main(0.7, 1., 0.3, 'wooh')
    matlab_res = 0.931080327325574
    assert np.isclose(ut.directivity_finite_width_2d(0.3, 0.7, 1.),
                      0.931080327325574)


def test_fluid_solid_real():
    """
    Test fluid_solid() below critical angles (real only).
    The conservation of energy should be respected.
    """
    alpha_fluid = np.deg2rad([0, 5, 10])

    # water:
    c_fluid = 1480.
    rho_fluid = 1000.

    # aluminium :
    c_l = 6320.
    c_t = 3130.
    rho_solid = 2700.

    with np.errstate(invalid='raise'):
        alpha_l = ut.snell_angles(alpha_fluid, c_fluid, c_l)
        alpha_t = ut.snell_angles(alpha_fluid, c_fluid, c_t)

        reflection, transmission_l, transmission_t = ut.fluid_solid(alpha_fluid,
                                                                    rho_fluid,
                                                                    rho_solid,
                                                                    c_fluid, c_l,
                                                                    c_t,
                                                                    alpha_l, alpha_t)

    # Compute the energy incoming and outcoming: the principle of conservation of energy must be respected.
    # Reference: Schmerr, §6.3.2

    # incident pressure
    pres_i = 10000.

    # cross section areas:
    area_fluid = np.cos(alpha_fluid)
    area_l = np.cos(alpha_l)
    area_t = np.cos(alpha_t)

    # Conservation of energy
    inc_energy = 0.5 * pres_i ** 2 / (rho_fluid * c_fluid) * area_fluid
    energy_refl = 0.5 * (reflection * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    energy_l = 0.5 * (transmission_l * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (transmission_t * pres_i) ** 2 / (rho_solid * c_t) * area_t

    # Check equality of complex values (this equality has NO physical meaning)
    np.testing.assert_allclose(inc_energy, energy_refl + energy_l + energy_t)


def test_fluid_solid_complex():
    """
    Test fluid_solid() below and above critical angles (complex).
    The conservation of energy should be respected for all cases.
    """

    alpha_fluid = np.asarray(np.deg2rad(np.arange(0., 85., 10.)), dtype=np.complex)

    # water:
    c_fluid = 1480.
    rho_fluid = 1000.

    # aluminium :
    c_l = 6320.
    c_t = 3130.
    rho_solid = 2700.

    with np.errstate(invalid='raise'):
        alpha_l = ut.snell_angles(alpha_fluid, c_fluid, c_l)
        alpha_t = ut.snell_angles(alpha_fluid, c_fluid, c_t)

        reflection, transmission_l, transmission_t = ut.fluid_solid(alpha_fluid,
                                                                    rho_fluid,
                                                                    rho_solid,
                                                                    c_fluid, c_l,
                                                                    c_t,
                                                                    alpha_l, alpha_t)

    # Compute the energy incoming and outcoming: the principle of conservation of energy must be respected.
    # Reference: Schmerr, §6.3.2

    # incident pressure
    pres_i = 10000.

    # cross section areas:
    area_fluid = np.cos(alpha_fluid)
    area_l = np.cos(alpha_l)
    area_t = np.cos(alpha_t)

    critical_l = np.arcsin(c_fluid / c_l)
    critical_t = np.arcsin(c_fluid / c_t)
    case1 = alpha_fluid < critical_l
    case2 = np.logical_and(critical_l < alpha_fluid, alpha_fluid < critical_t)
    case3 = critical_t < alpha_fluid
    assert np.any(case1)
    assert np.any(case2)
    assert np.any(case3)

    inc_energy = 0.5 * pres_i ** 2 / (rho_fluid * c_fluid) * area_fluid
    energy_refl = 0.5 * (reflection * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    energy_l = 0.5 * (transmission_l * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (transmission_t * pres_i) ** 2 / (rho_solid * c_t) * area_t

    # Check equality of complex values (this equality has NO physical meaning)
    np.testing.assert_allclose(inc_energy, energy_refl + energy_l + energy_t)

    # Check the conservation of energy for all cases:

    # CASE 1 - Below the two critical angle
    np.testing.assert_allclose(np.abs(inc_energy[case1]),
                               np.abs(energy_refl[case1]) + np.abs(
                                   energy_l[case1]) + np.abs(energy_t[case1]))

    # CASE 2 - L reflected, T transmitted
    np.testing.assert_allclose(np.abs(inc_energy[case2]),
                               np.abs(energy_refl[case2]) + np.abs(energy_t[case2]))

    # CASE 3 - L and T reflected
    np.testing.assert_allclose(np.abs(inc_energy[case3]), np.abs(energy_refl[case3]))


def test_solid_l_fluid():
    """
    Test solid_l_fluid() below critical angles (real only).
    The conservation of energy should be respected.
    """
    alpha_l = np.deg2rad(np.arange(0., 85., 10.))

    # water:
    c_fluid = 1480.
    rho_fluid = 1000.

    # aluminium :
    c_l = 6320.
    c_t = 3130.
    rho_solid = 2700.

    with np.errstate(invalid='raise'):
        alpha_fluid = ut.snell_angles(alpha_l, c_l, c_fluid)
        alpha_t = ut.snell_angles(alpha_l, c_l, c_t)

        reflection_l, reflection_t, transmission = ut.solid_l_fluid(alpha_l,
                                                                    rho_fluid,
                                                                    rho_solid,
                                                                    c_fluid, c_l,
                                                                    c_t,
                                                                    alpha_fluid,
                                                                    alpha_t)

    # Compute the energy incoming and outcoming: the principle of conservation of energy must be respected.
    # Reference: Schmerr, §6.3.2

    # incident pressure
    pres_i = 10000.

    # cross section areas:
    area_fluid = np.cos(alpha_fluid)
    area_l = np.cos(alpha_l)
    area_t = np.cos(alpha_t)

    inc_energy = 0.5 * pres_i ** 2 / (rho_solid * c_l) * area_l
    energy_trans = 0.5 * (transmission * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    energy_l = 0.5 * (reflection_l * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (reflection_t * pres_i) ** 2 / (rho_solid * c_t) * area_t

    # Check the conservation of energy:
    np.testing.assert_allclose(inc_energy, energy_trans + energy_l + energy_t)


def test_solid_t_fluid_complex():
    """
    Test solid_t_fluid() below and above critical angles (complex).
    The conservation of energy should be respected for all cases.
    """
    alpha_t = np.asarray(np.deg2rad([0, 5, 10, 20, 30, 40]), dtype=np.complex)

    # water:
    c_fluid = 1480.
    rho_fluid = 1000.

    # aluminium :
    c_l = 6320.
    c_t = 3130.
    rho_solid = 2700.

    with np.errstate(invalid='raise'):
        alpha_fluid = ut.snell_angles(alpha_t, c_t, c_fluid)
        alpha_l = ut.snell_angles(alpha_t, c_t, c_l)

        reflection_l, reflection_t, transmission = ut.solid_t_fluid(alpha_t,
                                                                    rho_fluid,
                                                                    rho_solid,
                                                                    c_fluid, c_l,
                                                                    c_t,
                                                                    alpha_fluid,
                                                                    alpha_l)

    # Compute the energy incoming and outcoming: the principle of conservation of energy must be respected.
    # Reference: Schmerr, §6.3.2

    # incident pressure
    pres_i = 10000.

    # cross section areas:
    area_fluid = np.cos(alpha_fluid)
    area_l = np.cos(alpha_l)
    area_t = np.cos(alpha_t)

    critical = np.arcsin(c_t / c_l)
    case1 = alpha_t < critical
    case2 = np.logical_not(case1)
    assert np.any(case1)
    assert np.any(case2)

    inc_energy = 0.5 * pres_i ** 2 / (rho_solid * c_t) * area_t
    energy_trans = 0.5 * (transmission * pres_i) ** 2 / (rho_fluid * c_fluid) * area_fluid
    energy_l = 0.5 * (reflection_l * pres_i) ** 2 / (rho_solid * c_l) * area_l
    energy_t = 0.5 * (reflection_t * pres_i) ** 2 / (rho_solid * c_t) * area_t

    # Check equality of complex values (this equality has NO physical meaning)
    np.testing.assert_allclose(inc_energy, energy_trans + energy_l + energy_t)

    # Check the conservation of energy for all cases:

    # CASE 1 - L is reflected
    np.testing.assert_allclose(np.abs(inc_energy[case1]),
                               np.abs(energy_trans[case1]) + np.abs(
                                   energy_l[case1]) + np.abs(energy_t[case1]))

    # CASE 2 - L not reflected
    np.testing.assert_allclose(np.abs(inc_energy[case2]),
                               np.abs(energy_trans[case2]) + np.abs(energy_t[case2]))


def test_snell_angles():
    """
    Test snell_angles() with both real and complex angles
    """
    incidents_angles = np.deg2rad([0, 10, 20, 30])

    # water:
    c = 1480.

    # aluminium :
    c_l = 6320
    c_t = 3130

    with np.errstate(invalid='ignore'):
        alpha_l = ut.snell_angles(incidents_angles, c, c_l)
        alpha_t = ut.snell_angles(incidents_angles, c, c_t)

    assert alpha_l.shape == incidents_angles.shape
    assert alpha_t.shape == incidents_angles.shape

    # Normal incident = normal refraction
    assert np.isclose(alpha_l[0], 0.)
    assert np.isclose(alpha_t[0], 0.)

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
    alpha_fluid = np.asarray(np.deg2rad(np.arange(0., 85., 10.)), dtype=np.complex)
    alpha_fluid = np.asarray(np.deg2rad([0, 5, 10]), dtype=np.float)

    # water:
    c_fluid = 1480.
    rho_fluid = 1000.

    # aluminium :
    c_l = 6320.
    c_t = 3130.
    rho_solid = 2700.

    alpha_l = ut.snell_angles(alpha_fluid, c_fluid, c_l)
    alpha_t = ut.snell_angles(alpha_fluid, c_fluid, c_t)

    # Transmission fluid->solid
    _, transmission_l_fs, transmission_t_fs = ut.fluid_solid(alpha_fluid, rho_fluid,
                                                             rho_solid, c_fluid, c_l,
                                                             c_t,
                                                             alpha_l, alpha_t)
    _, _, transmission_t_sf = ut.solid_t_fluid(alpha_t, rho_fluid, rho_solid,
                                               c_fluid, c_l, c_t, alpha_fluid,
                                               alpha_l)
    _, _, transmission_l_sf = ut.solid_l_fluid(alpha_l, rho_fluid, rho_solid,
                                               c_fluid, c_l, c_t, alpha_fluid,
                                               alpha_t)

    transmission_t_sf *= -1  # HACK. WHY?

    transmission_l_sf_stokes = rho_fluid * c_fluid * np.cos(
        alpha_l) * transmission_l_fs / (
                                   rho_solid * c_l * np.cos(alpha_fluid))
    transmission_t_sf_stokes = rho_fluid * c_fluid * np.cos(
        alpha_t) * transmission_t_fs / (
                                   rho_solid * c_t * np.cos(alpha_fluid))

    np.testing.assert_allclose(transmission_l_sf_stokes, transmission_l_sf)
    np.testing.assert_allclose(transmission_t_sf_stokes, transmission_t_sf)
