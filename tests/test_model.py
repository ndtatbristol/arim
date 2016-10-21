import numpy as np
from arim import model


def test_directivity_directivity_finite_width_2d():
    theta = 0.
    element_width = 1e-3
    wavelength = 0.5e-3
    directivity = model.directivity_finite_width_2d(theta, element_width, wavelength)

    assert np.isclose(directivity, 1.0)

    # From the NDT library (2016/03/22):
    # >>> fn_calc_directivity_main(0.7, 1., 0.3, 'wooh')
    matlab_res = 0.931080327325574
    assert np.isclose(model.directivity_finite_width_2d(0.3, 0.7, 1.), 0.931080327325574)


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
        alpha_l = model.snell_angles(alpha_fluid, c_fluid, c_l)
        alpha_t = model.snell_angles(alpha_fluid, c_fluid, c_t)

        reflection, transmission_l, transmission_t = model.fluid_solid(alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l,
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
        alpha_l = model.snell_angles(alpha_fluid, c_fluid, c_l)
        alpha_t = model.snell_angles(alpha_fluid, c_fluid, c_t)

        reflection, transmission_l, transmission_t = model.fluid_solid(alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l,
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
                               np.abs(energy_refl[case1]) + np.abs(energy_l[case1]) + np.abs(energy_t[case1]))

    # CASE 2 - L reflected, T transmitted
    np.testing.assert_allclose(np.abs(inc_energy[case2]), np.abs(energy_refl[case2]) + np.abs(energy_t[case2]))

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
        alpha_fluid = model.snell_angles(alpha_l, c_l, c_fluid)
        alpha_t = model.snell_angles(alpha_l, c_l, c_t)

        reflection_l, reflection_t, transmission = model.solid_l_fluid(alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t,
                                                                       alpha_fluid, alpha_t)

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
        alpha_fluid = model.snell_angles(alpha_t, c_t, c_fluid)
        alpha_l = model.snell_angles(alpha_t, c_t, c_l)

        reflection_l, reflection_t, transmission = model.solid_t_fluid(alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t,
                                                                       alpha_fluid, alpha_l)

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
                               np.abs(energy_trans[case1]) + np.abs(energy_l[case1]) + np.abs(energy_t[case1]))

    # CASE 2 - L not reflected
    np.testing.assert_allclose(np.abs(inc_energy[case2]), np.abs(energy_trans[case2]) + np.abs(energy_t[case2]))


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
        alpha_l = model.snell_angles(incidents_angles, c, c_l)
        alpha_t = model.snell_angles(incidents_angles, c, c_t)

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

    alpha_l = model.snell_angles(alpha_fluid, c_fluid, c_l)
    alpha_t = model.snell_angles(alpha_fluid, c_fluid, c_t)

    # Transmission fluid->solid
    _, transmission_l_fs, transmission_t_fs = model.fluid_solid(alpha_fluid, rho_fluid, rho_solid, c_fluid, c_l, c_t,
                                                                alpha_l, alpha_t)
    _, _, transmission_t_sf = model.solid_t_fluid(alpha_t, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid,
                                                  alpha_l)
    _, _, transmission_l_sf = model.solid_l_fluid(alpha_l, rho_fluid, rho_solid, c_fluid, c_l, c_t, alpha_fluid,
                                                  alpha_t)

    transmission_t_sf *= -1  # HACK. WHY?

    transmission_l_sf_stokes = rho_fluid * c_fluid * np.cos(alpha_l) * transmission_l_fs / (
        rho_solid * c_l * np.cos(alpha_fluid))
    transmission_t_sf_stokes = rho_fluid * c_fluid * np.cos(alpha_t) * transmission_t_fs / (
        rho_solid * c_t * np.cos(alpha_fluid))

    np.testing.assert_allclose(transmission_l_sf_stokes, transmission_l_sf)
    np.testing.assert_allclose(transmission_t_sf_stokes, transmission_t_sf)
