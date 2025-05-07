.. _user_install:

============
Installation
============

.. toctree::
    :maxdepth: 2
    :hidden:

    development

This section describes how to install arim for an end-user. For developer installation,
see the `developer install guide <development>`_ or
`Contributing.md <https://github.com/ndtatbristol/arim/blob/master/CONTRIBUTING.md>`_ in arim's git repository.

.. _reqs_user_install:

Requirements
------------

arim has the following requirements:

- Python 3
- `numpy <https://www.numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `numba <https://numba.pydata.org/>`_: efficient computation
- `matplotlib <https://matplotlib.org/>`_: plotting library used in :mod:`arim.plot` and in scripts
- `pyyaml <https://pyyaml.org/>`_: reading configuration files

The following dependencies are needed for the example scripts, although they may not be directly used in arim:

- `pandas <https://pandas.pydata.org/>`_: data analysis tools
- `pooch <https://www.fatiando.org/pooch/latest/>`_: for downloading data files

Finally, some users may require the following package:

- `h5py <https://www.h5py.org/>`_: reading MATLAB v7 datafile (used in :mod:`arim.io`)


.. _stable_intall:

Installation (stable)
---------------------

.. tab-set::

    .. tab-item:: Anaconda

        Make sure you have a working `Anaconda <https://www.anaconda.com/download/>`_ installation with Python v3.9 or greater.

        Anaconda will already have many requirements already installed, however you may have to install ...

        From the `arim releases page <https://github.com/ndtatbristol/arim/releases>`_, download the most recent wheel package
        (``*.whl`` file).

        Open the Anaconda Prompt, navigate to the directory where the wheel file was downloaded, and install this file

        .. code-block:: shell

            cd <download-directory>
            pip install <arim-wheel-file>

        For the most recent version, this is

        .. code-block:: shell

            cd <download-directory>
            pip install arim-0.9-py3-none-any.whl


    .. tab-item:: Conda / mamba environment
        :name: conda-env

        Make sure you have a working `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
        or `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ installation. Note that
        for the remainder of this section the `conda` command will be used - this may be replaced with `mamba` if this is
        your package manager.

        Open the Anaconda Prompt or Miniforge Prompt, and create a new environment with the required packages installed

        .. code-block:: shell

            conda create -n <env-name> numpy scipy numba matplotlib pyyaml pandas pooch h5py

        Activate this new environment

        .. code-block:: shell

            conda activate <env-name>

        From the `arim releases page <https://github.com/ndtatbristol/arim/releases>`_, download the most recent wheel
        package (``*.whl`` file).

        In the prompt, navigate to the directory where the wheel file was downloaded, and install this file

        .. code-block:: shell

            cd <download-directory>
            pip install <arim-wheel-file>

        For the most recent version, this is

        .. code-block:: shell

            cd <download-directory>
            pip install <arim-wheel-file>


arim will now be installed in your environment. Verify your installation

.. code-block:: python

    >>> import arim
    >>> print(arim.__version__)


Installation (latest)
---------------------

Make sure you have a working installation of Anaconda, Conda or Mamba. Make sure that your desired environment is
active, and that the requirements are installed. See :ref:`conda-env` for instructions.

.. tab-set::

    .. tab-item:: Github

        From the `main arim repository <https://github.com/ndtatbristol/arim>`_, click the green ``Code`` button, and
        ``Download ZIP`` to download the latest version.

        After it has downloaded, extract the contents of the zip file.

        Open your Anaconda Prompt or Miniforge Prompt, navigate to the extracted folder, and install the contents of the
        directory

        .. code-block:: shell

            cd <extract-directory>
            pip install .


    .. tab-item:: Git

        Open your Anaconda Prompt or Miniforge Prompt, and run

        .. code-block:: shell

            pip install git+https://github.com/ndtatbristol/arim


arim will now be installed in your environment. Verify your installation

.. code-block:: python

    >>> import arim
    >>> print(arim.__version__)
