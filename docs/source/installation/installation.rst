.. _user_install:

============
Installation
============

This section described how to install arim for the end-user. For developer installation,
see: :ref:`developer_installation`. Using conda is recommended because it ensures all dependencies are
present.

.. _reqs_user_install:

Requirements for user installation
==================================

Dependencies for arim itself:

- Python 3
- `numpy <http://www.numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `numba <http://numba.pydata.org/>`_: efficient computation
- `matplotlib <http://matplotlib.org/>`_: plotting library used in :mod:`arim.plot` and in scripts
- `pyyaml <http://pyyaml.org/>`_: used for reading configuration files

Dependencies for the example scripts (not used by arim *per se*):

- `pandas <http://pyyaml.org/>`_: data analysis tools

Optional dependency:

- `h5py <http://www.h5py.org/>`_: for reading MATLAB v7 datafile (used in :mod:`arim.io`)

Installation
============

Installation from a wheel file (recommended)
--------------------------------------------

Use case: general case.

Install `Anaconda distribution <https://www.anaconda.com/download/>`_ (Python 3 version).

Get a wheel package of arim (``.whl`` file) from the developpement team.

Start an Anaconda Prompt (in Windows, it should be in the Start menu) and type in::

  pip install <arim-wheel-file>

Example::

  pip install arim-0.3-py3-none-any.whl


Installation from a wheel file in a virtual environment
-------------------------------------------------------

Use cases:

- several versions of arim are needed on the same machine (create one environment per version),
- and/or the user prefers to keep conda root environment free from arim,
- and/or space is tight (install conda without the whole Anaconda distribution).

Install `Anaconda distribution <https://www.anaconda.com/download/>`_ (Python 3 version) or conda (`conda installation guide <http://conda.pydata.org/docs/download.html>`_).

In an Anaconda prompt, create a new virtual environment with the desired dependencies::

  conda create --name arim python numpy scipy numba matplotlib numba pyyaml pandas h5py

Activate the virtual environment::

  # on Windows:
  activate arim 

  # on Unix:
  source activate arim

Then install arim::

  pip install <arim-wheel-file>

Example::

  pip install arim-0.3-py3-none-any.whl

.. seealso::

  `conda documentation <https://conda.io/docs/>`_

.. _source_install:

Source installation from git
----------------------------

Use cases:

- tracking the latest changes in arim,
- and/or developer installation


Install all requirements, in a virtual environment if desired.
Clone the `github repository <https://github.com/nbud/arim>`_. The newly created directory is referred below
as your *local git repository*. It contains:

- ``arim/setup.py``: file for installing arim
- ``arim/arim``: directory of the code of arim
- ``arim/examples``: directory of example scripts
- ``arim/tests``: directory of unit tests for arim
- ``arim/docs``: directory of the present documentation (must be built first, :ref:`build_doc`)
- other elements.


Option 1: normal installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Anaconda Prompt, change to the top arim directory (the one with ``setup.py``) and type::

  python setup.py install

The content of your local git repository will be *copied* into the ``site-packages`` directory, which is the main
location where Python stores the non-standard libraries For an Anaconda installation on Windows with default settings,
this directory is::

  C:\ProgramData\Anaconda3\Lib\site-packages

When running ``import arim`` in Python, the files from the ``site-packages`` directory will be imported. Consequently,
updating your local git repository *will not change* the installed files. The local git repository can be safely deleted
if needed.


Option 2: developer installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Anaconda Prompt, change to the top arim directory (the one with ``setup.py``) and type::

  python setup.py develop

The content of the your local git repository becomes the place where Python looks up arim files during an import. These
files are *not copied* into the ``site-packages`` directory.  When running ``import arim`` in Python, the files from the
local git repository are imported.

.. seealso::

   :ref:`developer_installation`


Update arim
===========

Re-run the installation procedure with the updated wheel or source files.

Check arim is working
=====================

If arim was installed in a virtual environment, activate it first::

  # on Windows:
  activate arim 

  # on Unix-like:
  source activate arim

Start Python::

  python

Start arim::

  >>> import arim
  >>> arim.__version__
  '0.3'
  >>> exit()

Check also that arim executable is working by typing in a terminal::

  arim --version

Upgrade arim
============

In arim virtual environment (if any)::

  pip install <arim-wheel-file> --upgrade


Uninstall arim
==============

In arim virtual environment (if any)::

  pip uninstall arim

Remove the virtual environment (if any) with::

  conda env remove -n arim

