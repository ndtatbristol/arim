=================
How to contribute
=================


.. seealso::

  - :doc:`documentation`
  - :doc:`testing`

Code style
==========

The code must be formatted with Black_ formatter.

.. _Black: https://github.com/ambv/black

Version control system
======================

arim uses the version control system git.

.. seealso::

  `git documentation <https://git-scm.com/documentation>`_

Branches
--------

Please push in branch ``master`` only versions of arim which **successfully pass all tests**. When developing new
complex features, please a create a new branch first, develop the feature, add tests, and finally create a pull request.
to ``master`` when it is ready (`feature branch workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow>`_).

.. seealso:

  - `git documentation on branching <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_.
  - `Using pull requests on Github <https://help.github.com/articles/using-pull-requests/>`_

Commit
------

A commit must contain one functional change. In other words a commit must not contain changes in several unrelated
features.

Always use informative commit messages (`good practises for commit messages <https://wiki.openstack.org/wiki/GitCommitMessages>`_).

Example scripts
===============

Example scripts are intended to show features of arim. To analyse data with arim, please do not edit directly these scripts
but work on copies of them. Please avoid commit changes on these scripts.

Other conventions
=================

Prefer using the word "array" to refer to the data structure (numbers indexed by a set of indices). To refer to
the ultrasonic probe, use the word "probe" preferentially.