import pytest

import arim
import arim.models.block_in_contact as bic


def test_make_views():
    xmin = -5e-3
    xmax = 5e-3
    zmin = 0.0
    zmax = 7e-3
    grid = arim.Grid(xmin, xmax, 0.0, 0.0, zmin, zmax, pixel_size=1e-3)
    grid_p = grid.to_oriented_points()
    probe = arim.probes["ima_50_MHz_128_1d"]
    probe_p = probe.to_oriented_points()
    frontwall = arim.geometry.points_1d_wall_z(xmin, xmax, zmin, 11, name="Frontwall")
    backwall = arim.geometry.points_1d_wall_z(xmin, xmax, zmax, 10, name="Backwall")
    block_material = arim.Material(6300.0, 3120.0)

    # General ExaminationObject
    examination_object = arim.ExaminationObject(block_material)
    views = bic.make_views(
        examination_object,
        probe_p,
        grid_p,
        tfm_unique_only=False,
    )
    assert len(views) == 4

    with pytest.raises(ValueError):
        # Undefined backwall
        views = bic.make_views(
            examination_object, probe_p, grid_p, walls_for_imaging=["Backwall", "Frontwall"]
        )

    # BlockInContact with a backwall
    examination_object = arim.BlockInContact(block_material, walls={"Backwall":backwall})
    views = bic.make_views(
        examination_object,
        probe_p,
        grid_p,
        walls_for_imaging=["Backwall"],
        tfm_unique_only=False,
    )
    assert len(views) == 36

    with pytest.raises(ValueError):
        # Undefined frontwall
        views = bic.make_views(
            examination_object, probe_p, grid_p, walls_for_imaging=["Backwall", "Frontwall"]
        )

    # BlockInContact with a backwall and a frontwall
    examination_object = arim.BlockInContact(block_material, walls={"Backwall":backwall, "Frontwall":frontwall}, )
    views = bic.make_views(
        examination_object,
        probe_p,
        grid_p,
        walls_for_imaging=["Backwall", "Frontwall"],
        tfm_unique_only=False,
    )
    assert len(views) == 196