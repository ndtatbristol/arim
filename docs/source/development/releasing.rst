==============
How to release
==============

Release arim to share it with users.

Remark: it is not necessary to release arim for each commit.

Check all tests pass
====================

All tests must pass. Cf. :ref:`run_tests`.

Bump version number
===================

Version number must follow `PEP 440 <pep440>`_. The release number must have two components ("major.minor"). Use suffix
``.devN`` for development versions; use no suffix for stable versions. Examples (in chronological order)::

	- 0.3.dev0
	- 0.3.dev1
	- O.3.dev2
	- 0.3
	- 0.4.dev0
	- 0.4
	- 1.0.dev0
	- 1.0.dev1

To change the version number, edit the ``__init__.py`` in ``arim/__init__.py``::

	__version__ = '0.4'

.. _pep440: https://www.python.org/dev/peps/pep-0440/

Build documentation
===================

Cf. :ref:`build_doc`


Create distribution
===================

Conda package
-------------

Conda packages can be installed with ``conda``. They contain built files, therefore they are platform specific.

In arim root directory, where the file ``meta.yaml`` is::

  conda build .

The result is a ``.tar.gz2`` conda distribution file in your Anaconda installation (on Windows: ``C:\Anaconda3\conda-bld\win64``).

Wheel package
-------------

Wheel packages can be installed with ``pip``. They contain built files, therefore they are platform specific.

In arim root directory::

  python setup.py bdist_wheel

The result is a ``.whl`` wheel file in directory ``dist``.

Create source distribution
--------------------------

A source distribution contains only source files. Its content is very close to what is versionned in git. It is not
platform specific; however it has build to be executed.

In arim root directory::

	python setup.py sdist


The result is a ``.zip`` or ``.tar.gz`` file in directory ``dist``.

.. seealso::

	`Python Packaging User Guide <http://python-packaging-user-guide.readthedocs.io/en/latest/>`_

Commit the changes
==================

Commit
------

If everything went smoothly, commit if it has not been done previously and push.

::

	git add <list of changed files>
	git commit -m "description of the changes"
	git push


Tagging (stable releases only)
------------------------------

For stable releases (i.e. non development ones), create an annotated tag:

::

	git tag -a v0.4 -m "arim version 0.4"


Push it:

::

	git push origin v0.4

Then bump version again in ``arim/__init__.py``::

	__version__ = '0.5.dev0'

And commit:

	git commit -a -m 'bump version - back to development'

**Annex**: quick memo on tags. See also: `Git documentation on tagging <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`_.


::

	# Change to workspace to a specific tag:
	git checkout v0.1

	# Show metadata about a tag:
	git show v0.1

	# See all tags:
	git tag


http://python-packaging-user-guide.readthedocs.io/en/latest/