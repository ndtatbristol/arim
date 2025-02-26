.. _metadata:

========
Metadata
========

Several objects store non-essential information in a `metadata` dictionary.

.. _metadata_general:

General fields
==============

    - ``short_name``: an ASCII identifier
    - ``long_name``: unicode
    - ``description``
    - ``last_modified``: format YYYY-MM-DD (ISO 8601)
    - ``author``
    - ``version``: unsigned int, must be unique for the same 'short_name'. Starts at 0.


.. _metadata_frame:

Frame
=====

    - ``capture_method``: FMC or HMC. Must be a value in the enum ``CaptureMethod``
    - ``probes_locations`` (MFMC style positionning)
    - ``probes_orientations`` (MFMC style positionning)
    - ``from_brain``: filename of original exp_data file (str)

.. _metadata_probe:

Probe
=====

    - ``probe_type``: linear, array, single
    - ``serial``: str
    - ``from_brain``: filename of original exp_data file (str)

