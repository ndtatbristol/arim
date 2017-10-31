.. |source_doc_dir| replace:: ``docs/source``
.. |build_doc_dir| replace:: ``docs/build``
.. |rst| replace:: reStructuredText

=================
Documenting arim
=================

Sphinx
======

arim uses the documentation generator Sphinx_ for its own documentation. Sphinx is configured for using the
extension autodoc_ to generate documentation from docstrings in Python files, and napoleon_ to parse the docstrings
written in NumPy format.

.. _Sphinx: http://sphinx-doc.org/
.. _napoleon: http://sphinx-doc.org/ext/napoleon.html
.. _autodoc: http://sphinx-doc.org/ext/autodoc.html


Writing documentation
======================

Write documentation in |rst| files in `docs/source` and in `docstrings <https://en.wikipedia.org/wiki/Docstring#Python>`_
of Python files.

The configuration file for Sphinx is located in ``docs/source/conf.py``.


Docstring style
---------------

Docstrings must be written in the `NumPy docstring format`_.

.. _NumPy docstring format: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

If present, the sections in docstrings must be in the following order:

#) Short summary (one line summary)
#) Deprecation warning
#) Extended summary
#) Parameters
#) Attributes
#) Returns
#) Yields
#) Other Parameters
#) Raises
#) See Also
#) Notes
#) References
#) Examples

In a class, `__init__` must be documented in the docstring of the class. Arguments for `__init__`
must be described in **Parameters** section. See for example source code of :class:`arim.core.probe`.

Examples of well-written docstrings:

* https://github.com/numpy/numpy/blob/master/doc/example.py
* http://sphinx-doc.org/ext/example_numpy.html#example-numpy

References
----------

Academic references must conform style defined in Chicago Manual of Style 16th edition (author-date).

Example:

::

  Holmes, Caroline, Bruce W. Drinkwater, and Paul D. Wilcox. 2005. ‘Post-Processing of the Full Matrix of Ultrasonic Transmit–receive Array Data for Non-Destructive Evaluation’. NDT & E International 38 (8): 701–11. doi:10.1016/j.ndteint.2005.04.002.

Titles
------

Titles used in reStructuredText::

    =================
    Title of the page
    =================

    First level title
    =================

    Second level title
    ------------------

    Third level title
    ^^^^^^^^^^^^^^^^^

Other conventions
-----------------

Lines in |rst| files must be kept below 120-ish characters.


.. _build_doc:

Build the documentation
=======================

The following commands must be executed in directory ``docs``, where the files ``Makefile`` and ``make.bat`` are.

Build documentation
-------------------

:: 

  make clean
  make html


Optionally, the documentation can be exported as a single HTML page with ``make singlehtml``.

The built pages are located in |build_doc_dir|.

.. _howto_doc:

Memos
=====

.. seealso::

  - `reST memo <http://rest-sphinx-memo.readthedocs.org/en/latest/ReST.html>`_
  - `Sphinx memo <http://rest-sphinx-memo.readthedocs.org/en/latest/Sphinx.html>`_


How to document a function from its docstring
---------------------------------------------

Modules are auto-documented in the reference (:doc:`../reference/arim`). If it is relevant to show the documentation in
another page, use the following template:

.. code-block:: ReST

  .. autofunction:: arim.mymodule.myfunction
    :noindex:

  .. autoclass:: arim.mymodule.myclass
    :noindex:


See also: `autodoc documentation <http://sphinx-doc.org/ext/autodoc.html#directive-autofunction>`_

How to create a new page
------------------------

To create a new page in the documentation, follow these steps:

#. Create a new file |rst| with the extension *rst* in |source_doc_dir| or one of its subdirectories, and
   fill it the template above. Exemple: *userdoc/mymodule.rst*
#. Open the *index.rst* file contained in the same directory as the new page. In the *toctree* directive,
   add the name of the file with no extension.
#. Compile the documentation (:ref:`build_doc`)

**Template of a new reStructuredText page**

.. code-block:: rest

  .. highlight:: python

  ===========
  My new page
  ===========

  First part
  ==========

  Example of code::

    >>> 1+1
    2


How to add a figure generated automatically
-------------------------------------------

It is possible to embed in Sphinx a plot which is generated at compilation time.

#. Create a Python script in a relevant directory in |source_doc_dir|. Example: ``docs/source/foobar``.
#. In the |rst| file, at the location where the figure must be displayed, use the directive
   **plot** (see bellow).
#. Optionally, the source code used to generate the figure can also be inserted, with the directive
   **literalinclude** (see bellow).

**reStructuredText directives:**

.. code-block:: ReST

    .. plot:: foobar/<scriptname>.py

    .. literalinclude:: /foobar/<scriptname>.py
        :caption:

How to add a table
------------------

Writing a table in reStructuredText manually is far from being a pleasant operation. Such tables are
also hard to maintain. We recommend to write table in CSV files, with a spreadsheet program.

#. Create a spreadsheet in Excel.
#. Save it as CSV with semicolon ';' as delimiter, in the same directory as the reStructuredText
   file. Example: ``docs/source/devdoc/mytable.csv``
#. In the reStructuredText file, add the directive **csv-table** (see bellow).

**reStructuredText directive:**

.. code-block:: ReST

    .. csv-table:: Table - This is the title of the table
       :file: mytable.csv
       :header-rows: 1
       :delim: ;
       :name: mytable

To create a link to the table: ``:ref:`mytable```.

See also: `csv-table directive <http://docutils.sourceforge.net/docs/ref/rst/directives.html#csv-table>`_
