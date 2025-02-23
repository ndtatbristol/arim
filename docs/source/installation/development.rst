.. _dev_install:

============
Contributing
============

This section describes how to install arim for an end-user. For developer installation,
see the `developer install guide <installation/developer>`_ or
`Contributing.md <https://github.com/ndtatbristol/arim/blob/master/CONTRIBUTING.md>`_ in arim's git repository.

.. _reqs_user_install:

Requirements
============

Dependencies for arim itself:

- Python 3
- `numpy <https://www.numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `numba <https://numba.pydata.org/>`_: efficient computation
- `matplotlib <https://matplotlib.org/>`_: plotting library used in :mod:`arim.plot` and in scripts
- `pyyaml <https://pyyaml.org/>`_: used for reading configuration files

Dependencies for the example scripts (not used by arim *per se*):

- `pandas <https://pandas.pydata.org/>`_: data analysis tools
- `pooch <https://www.fatiando.org/pooch/latest/>`_: for downloading data files

Optional dependency:

- `h5py <https://www.h5py.org/>`_: for reading MATLAB v7 datafile (used in :mod:`arim.io`)

.. _stable_intall:

Installation (stable)
=====================

.. tab-set::

    .. tab-item:: Anaconda

        Make sure you have a working `Anaconda <https://www.anaconda.com/download/>`_ installation with Python v3.9 or greater.

        Anaconda will already have many requirements already installed, however you may have to install ...

        From the `arim releases page <https://github.com/ndtatbristol/arim/releases>`_, download the most recent wheel package
        (``*.whl`` file).

        Open the Anaconda Prompt, navigate to the directory where the wheel file was downloaded, and install this file ::

            cd <download-directory>
            pip install <arim-wheel-file>

        For the most recent version, this is ::

            cd <download-directory>
            pip install <arim-wheel-file>

        arim will now be installed in your environment. ::

            >>> import arim
            >>> print(arim.__version__)

    .. tab-item:: Conda / mamba environment

        Make sure you have a working `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
        or `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ installation.

        Open the Anaconda Prompt or Miniforge Prompt, and create a new environment with the required packages installed ::

            conda create -n <env-name> numpy scipy numba matplotlib pyyaml pandas pooch h5py

        Activate this new environment ::

            conda activate <env-name>

        From the `arim releases page <https://github.com/ndtatbristol/arim/releases>`_, download the most recent wheel
        package (``*.whl`` file).

        In the prompt, navigate to the directory where the wheel file was downloaded, and install this file ::

            cd <download-directory>
            pip install <arim-wheel-file>

        For the most recent version, this is ::

            cd <download-directory>
            pip install <arim-wheel-file>

        arim will now be installed in your environment. ::

            >>> import arim
            >>> print(arim.__version__)

Installation (latest)
=====================

.. tab-set::

    .. tab-item:: Github

        Make sure you have a working installation of Anaconda, Conda or Mamba, and that the requirements are installed.
        See above

        From the `main arim repository <https://github.com/ndtatbristol/arim>`_, click the green ``Code`` button, and
        ``Download ZIP`` to download the latest version.

        After it has downloaded, extract the contents of the zip file.

        Open your Anaconda Prompt or Miniforge Prompt


Stable release (recommended)
------------------------------------------------

Use case: general case.

Install `Anaconda distribution <https://www.anaconda.com/download/>`_ (Python 3 version).

Go to the `Release page of arim <https://github.com/ndtatbristol/arim/releases>`_.

Download the wheel package (``.whl`` file) corresponding to the latest release.

Start an Anaconda Prompt (in Windows, it should be in the Start menu) and type in::

  pip install <arim-wheel-file>

Example::

  pip install arim-1.0-py3-none-any.whl


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

  pip install arim-1.0-py3-none-any.whl

.. seealso::

  `conda documentation <https://conda.io/docs/>`_

.. _source_install:

Source installation from git
----------------------------

Use cases:

- tracking the latest changes in arim,
- and/or developer installation

Install all requirements, in a virtual environment if desired.
Clone the `github repository <https://github.com/ndtatbristol/arim>`_. The newly created directory is referred below
as your *local git repository*. It contains:

- ``arim/setup.py``: file for installing arim
- ``arim/arim``: directory of the code of arim
- ``arim/examples``: directory of example scripts
- ``arim/tests``: directory of unit tests for arim
- ``arim/docs``: directory of the present documentation (must be built first)
- other elements.


Option 1: normal installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Anaconda Prompt, change to the top arim directory (the one with ``setup.py``) and type::

  pip install .

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

  pip install . -e

The content of the your local git repository becomes the place where Python looks up arim files during an import. These
files are *not copied* into the ``site-packages`` directory.  When running ``import arim`` in Python, the files from the
local git repository are imported.

.. seealso::

   `CONTRIBUTING guide <https://github.com/ndtatbristol/arim/blob/master/CONTRIBUTING.md>`_


Update arim
===========

Re-run the installation procedure with the updated wheel or source files.

Check arim is working
=====================

Start Python::

  python

Start arim::

  >>> import arim
  >>> arim.__version__
  '1.0'
  >>> exit()


Upgrade arim
============

To upgrade, repeat the installation procedure with the updated package.

In arim virtual environment (if any)::

  pip install <arim-wheel-file> --upgrade


Uninstall arim
==============

In arim virtual environment (if any)::

  pip uninstall arim

Remove the virtual environment (if any) with::

  conda env remove -n arim

