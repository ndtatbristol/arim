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

Dependencies for the example scripts (not used by arim *per se*):

- `pyyaml <http://pyyaml.org/>`_: used for reading configuration files.
- `pandas <http://pyyaml.org/>`_: data analysis tools

Optional dependency:

- `h5py <http://www.h5py.org/>`_: for reading MATLAB v7 datafile (used in :mod:`arim.io`)

Dependencies for building from source:



Binary installation
===================

This is the easiest way to install arim. It requires a binary version of arim provided by the developement team (wheel file).

Installation with Anaconda (recommended)
----------------------------------------

Use case: general usage.

Install `Anaconda distribution <https://www.anaconda.com/download/>`_ (Python 3 version).

Get a wheel package of arim (``whl`` file) from the developpement team.

Start an Anaconda Prompt (in Windows, it should be in the Start menu) and type in::

  pip install <arim-wheel-file>

Example::

  pip install arim-0.3-py3-none-any.whl


Installation in a virtual environment
-------------------------------------

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

Build and install from source
==============================

Use case:

- no wheel is provided for the platform and/or the Python version that the user uses
- and/or developer installation

Compiler requirements
---------------------

A C++ compiler with OpenMP 2.0 or newer is required.

For Windows and Python 3.5 or Python 3.6, the development team recommends using Visual C++ 2015 build tools.
They can be obtain as a `standalone <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_
or by installing Visual Studio 2015.

.. seealso::

  `Python documentation: Windows compilers <https://wiki.python.org/moin/WindowsCompilers>`_


Additional Python dependency
----------------------------

`Cython <http://cython.org/>`_: static compiler

Installation
------------

Install all requirements, in a virtual environment if desired. Get arim source code.

In a prompt, build and install::

  python setup.py install

Alternatively, for an editable inplace installation (useful for development), type::

  python setup.py develop

.. seealso::

   :ref:`developer_installation`


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

