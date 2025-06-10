# Version 0.10, 06.2025

- Documentation moved to numpy style, added hosting on [Github Pages](https://ndtatbristol.github.io/arim/)
- Changes to `arim.geometry` - generally move towards arbitrary 2D geometry:
	- Make any rectilinear geometry (``arim.geometry.make_contiguous_geometry(...)``)
	- Support for generating curved surfaces (``arim.geometry.combine_oriented_points(...)``), although simulation not validated
	- Path and view names include wall names, to ensure any view is unique
	- Models (``block_in_immersion`` and ``block_in_contact``) functions ``make_views(...)`` now takes argument `walls_for_imaging` instead of `max_number_of_reflection`, as frontwall/backwall is no longer assumed.
	- Added new object ``arim.geometry.MaskedGrid``, to optionally turn off specified pixels in a grid. May improve performance if pixels are known to not be needed (e.g. based on geometry).
	
- New example `multiview_tfm_non_line_of_sight.py`
- Updated BRAIN `exp_data` properties in ``arim.io.load_expdata(...)``: optional `location` field is sometimes present, and is useful to have access to in frame


# Version 0.9, 06.2024

- Add TFM median (``arim.im.tfm.tfm_for_view(..., aggregation="median")``)
- Add `arim.datasets` to automatically fetch test datasets
- Add unfinished forward model for contact inspection
- [Better example scripts, bundled with test data](https://github.com/ndtatbristol/arim/tree/v0.9/examples)
- replace "scanline" (coming from old specs of MFMC) with the more understandable "timetrace"

Internal changes:

- Add MIT licence
- Add continuous integration using GitHub Action
- [Improve CONTRIBUTING guide](https://github.com/ndtatbristol/arim/blob/master/CONTRIBUTING.md)


# Version 0.8, 05.2019

First release, featuring notably:

- a 2D forward model for immersion inspection of small defects,
- multi-view Total Focusing Method imaging for contact and immersion inspections.

# Version 0.7

Legacy version, not production ready. Requires Cython and a C/C++ compiler to compile from source.
