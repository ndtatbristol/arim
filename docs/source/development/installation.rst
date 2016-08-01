.. _developer_installation:

======================
Developer installation
======================

Requirements
============

All requirements for :ref:`reqs_user_install`. In particular Python 3.5 and conda.

The following Python libraries are required:

- sphinx: documentation generator
- sphinx-rtd-theme: theme for sphinx
- pytest: test runner
- setuptools (conda package): required for packaging

Other software:

- git: version control system
- `conda <http://conda.pydata.org/docs/>`_: package management system


Installation
============

First clone the git directory::

  git clone https://github.com/nbud/arim

Create a new virtual environment::

  conda create -n arim-dev python=3.5

Activate the environment::

  # On Windows:
  activate arim-dev

  # On Unix:
  source activate arim-dev

Install dependencies::

  conda install numpy scipy matplotlib h5py numba sphinx sphinx_rtd_theme

Then install arim in editable mode::

  cd arim
  # check you are in the directory containing file setup.py
  pip install -e .

Then :doc:`build the library <building>` and finally :ref:`check all tests pass <run_tests>`.

*Optional:* if you intend to build conda packages, install in root environment the package conda-build::

  source deactivate
  conda install conda-build
