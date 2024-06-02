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
