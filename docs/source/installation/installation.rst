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

Strict dependencies:

- Python 3.5
- `numpy <http://www.numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `numba <http://numba.pydata.org/>`_: efficient computation

Optional dependencies:

- `matplotlib <http://matplotlib.org/>`_: plotting library (used in :mod:`arim.plot`)
- `h5py <http://www.h5py.org/>`_: for reading MATLAB v7 file (used in :mod:`arim.io`)


Installation
============

This is the **recommended method** to install arim.

Get a wheel package of arim (``whl`` file).

Install conda (`conda installation guide <http://conda.pydata.org/docs/download.html>`_).

Create a new virtual environment::

  conda create --name arim python=3.5 numpy scipy matplotlib hdf5 numba

Activate the virtual environment::


  # on Windows:
  activate arim 

  # on Unix:
  source activate arim

Then install arim::

  pip install <arim-wheel-file>

Example::

  pip install arim-0.3-py3-none-any.whl

**Remark:** if you prefer not to use conda (for example: `WinPython <http://winpython.github.io/>`_ users), ensure all
dependencies are satisfied then install the wheel using the ``pip`` command above. WinPython users can make use of the
embedded graphical control panel (menu Packages -> Add packages).


Upgrade arim
============

In arim virtual environment::

  pip install <arim-wheel-file> --upgrade


Uninstall arim
==============

  conda env remove -n arim

Check arim is working
=====================

Activate the virtual environment was installed (assumed to be named ``arim`` here)::

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