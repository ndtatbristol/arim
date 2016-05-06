=======
Testing
=======

.. _run_tests:

Running tests
=============

In arim root directory, type:

::

  py.test


Adding tests
============

Tests check functions work as expected at writing time, and keep working over time (non-regression). A very basic
test is still better than no test at all. arim uses the framework `pytest <http://pytest.org/>`_ for testing.

Tests for the module ``arim.foo.bar`` must be placed in script ``tests/foo/test_bar.py``.

Here is a basic example of test of function ``arim.foo.bar.baz``::

  import arim.foo.bar

  def test_baz():
      expected_output = 'something'
      assert arim.foo.bar.baz(1) == expected_output

      expected_output = 'something_else'
      assert arim.foo.bar.baz(2) == expected_output


.. seealso::

  `pytest: getting started <http://pytest.org/latest/getting-started.html>`_

Datasets for tests
==================

Binary files used for tests must be placed in directory ``tests/data`` or one of its subdirectory.

