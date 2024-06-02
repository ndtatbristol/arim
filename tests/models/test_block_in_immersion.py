import collections
import math

import numpy as np
import pytest

import arim
import arim.models.block_in_immersion as bim
import arim.ray
import arim.scat
from tests.test_model import make_context


def test_ray_weights():
    context = make_context()
    paths = context["paths"]
    """:type : dict[str, arim.Path]"""
    ray_geometry_dict = context["ray_geometry_dict"]
    """:type : dict[str, arim.path.RayGeometry]"""

    model_options = dict(
        frequency=context["freq"],
        probe_element_width=context["element_width"],
        use_beamspread=True,
        use_directivity=True,
        use_transrefl=True,
        use_attenuation=True,
    )

    for pathname, path in paths.items():
        # Direct
        ray_geometry = ray_geometry_dict[pathname]
        ray_geometry2 = arim.ray.RayGeometry.from_path(path, use_cache=False)
        weights, weights_dict = bim.tx_ray_weights(path, ray_geometry, **model_options)
        weights2, _ = bim.tx_ray_weights(path, ray_geometry2, **model_options)
        assert "beamspread" in weights_dict
        assert "directivity" in weights_dict
        assert "transrefl" in weights_dict
        assert "attenuation" in weights_dict
        np.testing.assert_allclose(weights, weights2)

        # Reverse
        ray_geometry = ray_geometry_dict[pathname]
        ray_geometry2 = arim.ray.RayGeometry.from_path(path, use_cache=False)
        weights, weights_dict = bim.rx_ray_weights(path, ray_geometry, **model_options)
        weights2, _ = bim.rx_ray_weights(path, ray_geometry2, **model_options)
        assert "beamspread" in weights_dict
        assert "directivity" in weights_dict
        assert "transrefl" in weights_dict
        assert "attenuation" in weights_dict
        np.testing.assert_allclose(weights, weights2)


def test_ray_weights_for_views():
    context = make_context()
    views = context["views"]
    paths = context["paths"]
    paths_set = set(paths.values())

    ray_weights_cache = bim.ray_weights_for_views(
        views, frequency=context["freq"], probe_element_width=context["element_width"]
    )

    assert ray_weights_cache.tx_ray_weights_debug_dict is None
    assert ray_weights_cache.rx_ray_weights_debug_dict is None
    assert len(paths) >= len(ray_weights_cache.tx_ray_weights_dict) > 3
    assert len(paths) >= len(ray_weights_cache.rx_ray_weights_dict) > 3
    assert set(ray_weights_cache.rx_ray_weights_dict.keys()) == paths_set
    nbytes_without_debug = ray_weights_cache.nbytes
    assert nbytes_without_debug > 0

    ray_weights_cache = bim.ray_weights_for_views(
        views,
        frequency=context["freq"],
        probe_element_width=context["element_width"],
        save_debug=True,
    )

    assert (
        ray_weights_cache.tx_ray_weights_debug_dict.keys()
        == ray_weights_cache.tx_ray_weights_dict.keys()
    )
    assert (
        ray_weights_cache.rx_ray_weights_debug_dict.keys()
        == ray_weights_cache.rx_ray_weights_dict.keys()
    )
    assert len(paths) >= len(ray_weights_cache.tx_ray_weights_dict) > 3
    assert len(paths) >= len(ray_weights_cache.rx_ray_weights_dict) > 3
    assert set(ray_weights_cache.rx_ray_weights_dict.keys()) == paths_set
    nbytes_with_debug = ray_weights_cache.nbytes
    assert nbytes_with_debug > nbytes_without_debug


def test_path_in_immersion():
    xmin = -20e-3
    xmax = 100e-3

    couplant = arim.Material(
        longitudinal_vel=1480,
        transverse_vel=None,
        density=1000.0,
        state_of_matter="liquid",
        metadata={"long_name": "Water"},
    )
    block = arim.Material(
        longitudinal_vel=6320.0,
        transverse_vel=3130.0,
        density=2700.0,
        state_of_matter="solid",
        metadata={"long_name": "Aluminium"},
    )

    probe_points, probe_orientations = arim.geometry.points_1d_wall_z(
        0e-3, 15e-3, z=0.0, numpoints=16, name="Frontwall"
    )

    frontwall_points, frontwall_orientations = arim.geometry.points_1d_wall_z(
        xmin, xmax, z=0.0, numpoints=20, name="Frontwall"
    )
    backwall_points, backwall_orientations = arim.geometry.points_1d_wall_z(
        xmin, xmax, z=40.18e-3, numpoints=21, name="Backwall"
    )

    grid = arim.geometry.Grid(
        xmin, xmax, ymin=0.0, ymax=0.0, zmin=0.0, zmax=20e-3, pixel_size=5e-3
    )
    grid_points, grid_orientation = grid.to_oriented_points()

    interfaces = arim.models.block_in_immersion.make_interfaces(
        couplant,
        (probe_points, probe_orientations),
        (frontwall_points, frontwall_orientations),
        (grid_points, grid_orientation),
        [(backwall_points, backwall_orientations), (frontwall_points, frontwall_orientations)],
    )
    assert interfaces["probe"].points is probe_points
    assert interfaces["probe"].orientations is probe_orientations
    assert interfaces["frontwall_trans"].points is frontwall_points
    assert interfaces["frontwall_trans"].orientations is frontwall_orientations
    assert interfaces["frontwall_refl"].points is frontwall_points
    assert interfaces["frontwall_refl"].orientations is frontwall_orientations
    assert interfaces["backwall_refl"].points is backwall_points
    assert interfaces["backwall_refl"].orientations is backwall_orientations
    assert interfaces["grid"].points is grid_points
    assert interfaces["grid"].orientations is grid_orientation

    # ------------------------------------------------------------------------------------
    # 6 paths, 21 views

    paths = arim.models.block_in_immersion.make_paths(block, couplant, interfaces)
    assert len(paths) == 6
    assert paths["L"].to_fermat_path() == (
        probe_points,
        couplant.longitudinal_vel,
        frontwall_points,
        block.longitudinal_vel,
        grid_points,
    )
    assert paths["TL"].to_fermat_path() == (
        probe_points,
        couplant.longitudinal_vel,
        frontwall_points,
        block.transverse_vel,
        backwall_points,
        block.longitudinal_vel,
        grid_points,
    )

    for path_key, path in paths.items():
        assert path_key == path.name

    # Make views
    views = arim.models.block_in_immersion.make_views_from_paths(
        paths, tfm_unique_only=True
    )
    assert len(views) == 21

    view = views["LT-LT"]
    assert view.tx_path is paths["LT"]
    assert view.rx_path is paths["TL"]

    # ------------------------------------------------------------------------------------
    # 14 paths, 105 views
    paths = arim.models.block_in_immersion.make_paths(
        block, couplant, interfaces, max_number_of_reflection=2
    )
    assert len(paths) == 14
    assert paths["L"].to_fermat_path() == (
        probe_points,
        couplant.longitudinal_vel,
        frontwall_points,
        block.longitudinal_vel,
        grid_points,
    )
    assert paths["TL"].to_fermat_path() == (
        probe_points,
        couplant.longitudinal_vel,
        frontwall_points,
        block.transverse_vel,
        backwall_points,
        block.longitudinal_vel,
        grid_points,
    )
    assert paths["TTL"].to_fermat_path() == (
        probe_points,
        couplant.longitudinal_vel,
        frontwall_points,
        block.transverse_vel,
        backwall_points,
        block.transverse_vel,
        frontwall_points,
        block.longitudinal_vel,
        grid_points,
    )

    for path_key, path in paths.items():
        assert path_key == path.name

    # Make views
    views = arim.models.block_in_immersion.make_views_from_paths(
        paths, tfm_unique_only=True
    )
    assert len(views) == 105

    view = views["LT-LT"]
    assert view.tx_path is paths["LT"]
    assert view.rx_path is paths["TL"]


def test_make_views():
    context = make_context()
    probe_oriented_points = context["probe_oriented_points"]
    scatterer_oriented_points = context["scatterer_oriented_points"]
    exam_obj = context["exam_obj"]

    views = bim.make_views(exam_obj, probe_oriented_points, scatterer_oriented_points)

    assert list(views.keys()) == list(context["views"].keys())


SCATTERERS_SPECS = [
    dict(kind="sdh", radius=0.5e-3),
    dict(kind="point"),
    dict(kind="crack_centre", crack_length=2e-3),
]


@pytest.mark.parametrize("scat_specs", SCATTERERS_SPECS)
def test_model(scat_specs, show_plots):
    couplant = arim.Material(
        longitudinal_vel=1480.0,
        density=1000.0,
        state_of_matter="liquid",
        longitudinal_att=arim.material_attenuation_factory("constant", 1.0),
    )
    block = arim.Material(
        longitudinal_vel=6320.0,
        transverse_vel=3130.0,
        density=2700.0,
        state_of_matter="solid",
        longitudinal_att=arim.material_attenuation_factory("constant", 2.0),
        transverse_att=arim.material_attenuation_factory("constant", 3.0),
    )

    probe = arim.Probe.make_matrix_probe(5, 1e-3, 1, np.nan, 5e6)
    probe_element_width = 0.8e-3
    probe.set_reference_element("first")
    probe.reset_position()
    probe.translate([0.0, 0.0, -5e-3])
    probe.rotate(arim.geometry.rotation_matrix_y(np.deg2rad(10)))

    probe_p = probe.to_oriented_points()
    frontwall = arim.geometry.points_1d_wall_z(
        numpoints=1000, xmin=-5.0e-3, xmax=20.0e-3, z=0.0, name="Frontwall"
    )
    backwall = arim.geometry.points_1d_wall_z(
        numpoints=1000, xmin=-5.0e-3, xmax=20.0e-3, z=30.0e-3, name="Backwall", is_block_above=False
    )
    scatterer_p = arim.geometry.default_oriented_points(
        arim.Points([[19e-3, 0.0, 20e-3]])
    )

    # import arim.plot as aplt
    # all_points = [probe_p, frontwall, backwall, scatterer_p]
    # aplt.plot_interfaces(all_points, markers=['o', 'o', 'o', 'd'],
    #                      show_orientations=True)
    # aplt.plt.show()

    exam_obj = arim.BlockInImmersion(block, couplant, [backwall, frontwall], [0, 1], scatterer_p)
    scat_obj = arim.scat.scat_factory(material=block, **scat_specs)
    scat_funcs = scat_obj.as_angles_funcs(probe.frequency)

    # compute only a subset of the FMC: first row and first column
    tx = np.zeros(probe.numelements * 2, np.int_)
    tx[: probe.numelements] = np.arange(probe.numelements)
    rx = np.zeros(probe.numelements * 2, np.int_)
    rx[probe.numelements :] = np.arange(probe.numelements)

    # Compute model
    views = bim.make_views(exam_obj, probe_p, scatterer_p, max_number_of_reflection=2)
    arim.ray.ray_tracing(views.values())
    ray_weights = bim.ray_weights_for_views(views, probe.frequency, probe_element_width)
    lti_coefficients = collections.OrderedDict()
    for viewname, view in views.items():
        amp_obj = arim.model.model_amplitudes_factory(
            tx, rx, view, ray_weights, scat_funcs
        )
        lti_coefficients[viewname] = amp_obj[0]

    # Test reciprocity
    for i, viewname in enumerate(views):
        viewname_r = arim.ut.reciprocal_viewname(viewname)
        lhs = lti_coefficients[viewname][: probe.numelements]  # (tx=k, rx=0) for all k
        rhs = lti_coefficients[viewname_r][probe.numelements :]  # (tx=0, rx=k) for all k

        max_err = np.max(np.abs(lhs - rhs))
        err_msg = f"view {viewname} (#{i}) - max_err={max_err}"

        tol = dict(rtol=1e-7, atol=1e-8)

        try:
            np.testing.assert_allclose(lhs, rhs, err_msg=err_msg, **tol)
        except AssertionError as e:
            if show_plots:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(nrows=2, sharex=True)
                ax = axes[0]
                ax.plot(lhs.real, label="tx=k, rx=0")
                ax.plot(rhs.real, label="tx=0, rx=k")
                ax.set_title(
                    scat_obj.__class__.__name__ + f"\n {viewname} and {viewname_r}"
                )
                ax.set_ylabel("real")
                ax.legend()
                ax = axes[1]
                ax.plot(lhs.imag, label="tx=k, rx=0")
                ax.plot(rhs.imag, label="tx=0, rx=k")
                ax.legend()
                ax.set_xlabel("element index k")
                ax.set_ylabel("imag")
                plt.show()
            raise e


@pytest.mark.parametrize("use_multifreq", [False, True])
def test_fulltime_model(use_multifreq, show_plots):
    # Setup
    couplant = arim.Material(
        longitudinal_vel=1480.0, density=1000.0, state_of_matter="liquid"
    )
    block = arim.Material(
        longitudinal_vel=6320.0,
        transverse_vel=3130.0,
        density=2700.0,
        state_of_matter="solid",
        longitudinal_att=arim.material_attenuation_factory("constant", 2.0),
        transverse_att=arim.material_attenuation_factory("constant", 20.0),
    )

    probe = arim.Probe.make_matrix_probe(20, 1e-3, 1, np.nan, 5e6)
    probe_element_width = 0.8e-3
    probe.set_reference_element("first")
    probe.reset_position()
    probe.translate([0.0, 0.0, -5e-3])
    probe.rotate(arim.geometry.rotation_matrix_y(np.deg2rad(10)))

    probe_p = probe.to_oriented_points()
    frontwall = arim.geometry.points_1d_wall_z(
        numpoints=1000, xmin=0.0e-3, xmax=40.0e-3, z=0.0, name="Frontwall"
    )
    backwall = arim.geometry.points_1d_wall_z(
        numpoints=1000, xmin=0.0e-3, xmax=40.0e-3, z=30.0e-3, name="Backwall"
    )
    scatterer_p = arim.geometry.default_oriented_points(
        arim.Points([[35e-3, 0.0, 20e-3]])
    )

    # if show_plots:
    #     import arim.plot as aplt
    #     all_points = [probe_p, frontwall, backwall, scatterer_p]
    #     aplt.plot_interfaces(
    #         all_points, markers=["o", "o", "o", "d"], show_orientations=True
    #     )
    #     aplt.plt.show()

    exam_obj = arim.BlockInImmersion(block, couplant, [backwall, frontwall], [0], scatterer_p)
    scat_obj = arim.scat.scat_factory(material=block, kind="sdh", radius=0.5e-3)
    scat_angle = 0.0

    tx_list, rx_list = arim.ut.fmc(probe.numelements)

    # Toneburst
    dt = 0.25 / probe.frequency  # to adjust so that the whole toneburst is sampled
    toneburst_time, toneburst, toneburst_t0_idx = arim.model.make_toneburst2(
        5, probe.frequency, dt, num_before=1
    )
    toneburst_f = np.fft.rfft(toneburst)
    toneburst_freq = np.fft.rfftfreq(len(toneburst_time), dt)

    # Allocate a long enough time vector for the timetraces
    views = bim.make_views(
        exam_obj,
        probe_p,
        scatterer_p,
        max_number_of_reflection=0,
        tfm_unique_only=False,
    )
    arim.ray.ray_tracing(views.values())
    max_delay = max(
        view.tx_path.rays.times.max() + view.rx_path.rays.times.max()
        for view in views.values()
    )
    timetraces_time = arim.Time(
        0.0, dt, math.ceil(max_delay / dt) + len(toneburst_time)
    )
    timetraces = None

    # Run model
    if use_multifreq:
        model_freq_array = toneburst_freq
    else:
        model_freq_array = probe.frequency

    transfer_function_iterator = bim.scat_unshifted_transfer_functions(
        views,
        tx_list,
        rx_list,
        model_freq_array,
        scat_obj,
        probe_element_width=probe_element_width,
        use_directivity=True,
        use_beamspread=True,
        use_transrefl=True,
        use_attenuation=True,
        scat_angle=scat_angle,
        numangles_for_scat_precomp=120,
    )

    for unshifted_transfer_func, delays in transfer_function_iterator:
        timetraces = arim.model.transfer_func_to_timetraces(
            unshifted_transfer_func,
            delays,
            timetraces_time,
            toneburst_time,
            toneburst_freq,
            toneburst_f,
            toneburst_t0_idx,
            timetraces=timetraces,
        )
    frame = arim.Frame(timetraces, timetraces_time, tx_list, rx_list, probe, exam_obj)
    if show_plots:
        import matplotlib.pyplot as plt

        import arim.plot as aplt

        aplt.plot_bscan_pulse_echo(frame)
        plt.title(f"test_fulltime_model - Bscan - use_multifreq={use_multifreq}")

        tx = 0
        rx = probe.numelements - 1
        plt.figure()
        plt.plot(np.real(frame.get_timetrace(tx, rx)), label=f"tx={tx}, rx={rx}")
        plt.plot(np.real(frame.get_timetrace(rx, tx)), label=f"tx={rx}, rx={tx}")
        plt.title(f"test_fulltime_model - use_multifreq={use_multifreq}")
        plt.legend()
        plt.show()