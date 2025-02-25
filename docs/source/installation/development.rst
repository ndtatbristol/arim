.. _dev_install:

============
Contributing
============

Thank you for your interest in arim! There are many ways to contribute, including:

- Reporting issues for bug fixes,
- Writing documentation,
- Writing example scripts to help users get up to speed quickly,
- Writing unit tests to ensure the code works as intended,
- Adding new features.

Developer installation
======================

Installation for development requires extra packages for testing, running the code formatting and linting, and building
documentation. The recommended instructions are as follows.

First, create a fork of arim under your own Github account.

.. tab-set::

    .. tab-item:: Hatch

        Install `hatch <https://hatch.pypa.io/latest/install/>`_.

        Clone your forked arim repository locally.

        Create a new virtual environment and install the dependencies. You can do this from any prompt or terminal window

        .. code-block:: shell

            cd <arim-clone-directory>
            hatch env create

        To activate this virtual environment, use ``hatch shell``. Refer to the `hatch documentation <https://hatch.pypa.io/latest/intro/>`_
        for further details.

    .. tab-item:: conda / mamba environment

        Make sure you have a working `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
        or `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ installation. For the
        remainder of this section, ``mamba`` will be used - replace this with ``conda`` if this is your preferred package
        manager.

        Clone your forked arim repository locally.

        Create a new ``mamba`` environment, and install arim.

        .. code-block:: shell

            mamba create -n arim-env
            cd <arim-clone-directory>
            pip install -e .

        This will install the required dependencies, as well as arim in editable mode. Any changes you make to the code
        base in this folder will therefore be used when running arim.


.. _quality_guidelines:

Code quality guidelines
=======================

Adhering to a common code style helps make the code more readable. As code is written once but read multiple times, it
is important that it is written well to save time in the long run. Meaningful and expressive variable and function names,
with no or little abbreviation, are essential.

arim code follows the general guidelines defined in `Python PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_, with the
amendments defined in the following sections.


Code formatting and linting
---------------------------

arim uses the `black <https://black.readthedocs.io/en/stable/>`_ formatter and the `ruff <https://docs.astral.sh/ruff/>`_
linter. To format and lint from your environment, use

.. code-block:: shell

    hatch run lint:fmt

To run a check without changing any code, use

.. code-block:: shell

    hatch run lint:check

The linter configuration is defined in `pyproject.toml <https://github.com/ndtatbristol/arim/blob/master/pyproject.toml>`_.


Docstrings
----------

Docstrings for functions, classes and modules follow `numpy's docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_


Documentation
=============

arim's documentation is powered by `Sphinx <http://sphinx-doc.org/>`_, with the most recent version deployed using
`Github Pages <https://pages.github.com/>`_.

The documentation is generated from two sources:

1. The files found in ``docs/source``, formatted as ReStructuredText, and
2. The docstrings in the codebase, compiled via `autosummary <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_.

If including academic references, please use the author-date style from the Chicago Manual of Style, e.g.
- Holmes, Caroline, Bruce W. Drinkwater, and Paul D. Wilcox. 2005. ‘Post-Processing of the Full Matrix of Ultrasonic Transmit–receive Array Data for Non-Destructive Evaluation’. NDT & E International 38 (8): 701–11. doi:10.1016/j.ndteint.2005.04.002.


Building the documentation
--------------------------

arim uses Github Actions to automatically build the documentation when new pushes are made, or when a new pull request
is accepted. Before this happens, please test that it works by building a version locally:

.. tab-set::

    .. tab-item:: hatch

        .. code-block:: shell

            cd <arim-clone-directory>/docs
            hatch shell default
            make html

    .. tab-item:: mamba / conda

        .. code-block:: shell

            cd <arim-clone-directory>/docs
            mamba activate arim-env
            make html

The output will be found in ``docs/build/html``.


Version control
===============

A commit should contain one functional change. In other words, it should not contain multiple unrelated features. It is
also important to use `informative commit messages <https://wiki.openstack.org/wiki/GitCommitMessages>`_.

It is best practice to only push to branch ``master`` versions of arim which successfully pass all tests. When developing
new features, please create a new branch first to develop the feature locally. Add tests, docstrings, examples, and if
necessary update the user guides in the documentation. Finally, only when all tests are passing, should you finally
create a pull request to ``master``. (See `this article <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow>`_
for more details).


.. _unit_testing:

Testing
=======

Unit tests ensure that a given function returns the intended results at the time of commit, as well as much later down
the line (i.e. it is non-regressive). arim uses `pytest <https://docs.pytest.org/>`_ to do unit testing. Tests are defined
in the ``tests`` directory. Please consider adding new tests!

To run the tests, use

.. tab-set::

    .. tab-item:: hatch

        .. code-block:: shell

            hatch run test

    .. tab-item:: mamba / conda

        .. code-block:: shell

            mamba activate arim-env
            pytest

All tests must pass before a pull request will be accepted into the ``master`` branch.


Pull requests
=============

You can propose changes to arim using `pull requests <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests>`_.
By submitting a pull request, you accept that the proposed changes are licensed under the MIT license. The proposed
changes must also comply with arim's `code quality guidelines <quality_guidelines>`_.


Releases
========

Releases should be made when new features are added. To create a release,
1. Ensure all `unit tests <unit_testing>`_ pass.
2. Change arim's version number in ``src/arim/__init__.py``, following the `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_
convention. Commit with an instructive description.
3. Assign a `tag <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`_ to the release commit. For example, if the version
number is 1.1, the tag name should be "v1.1".
4. Build the documentation, and save the HTML files in a zip names "documentation.zip", outside of the tracked repository.
5. Create a wheel package

.. code-block:: shell

    hatch build

6. `Create a new release on Github <https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository>`_.
Select the newly created tag, and describe the changes in this version. Attach both the wheel (``.whl``) file and
``documentation.zip``.
