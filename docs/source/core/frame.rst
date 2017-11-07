.. _frame:

=====
Frame
=====

.. py:currentmodule:: arim.core

A :class:`Frame` is a data container for a :class:`Probe`, an :class:`ExaminationObject` and the voltage-time data
:attr:`Frame.scanlines`. FMC, HMC, or any subset of FMC frames are supported.
It corresponds roughly to the content of the ``exp_data`` structure of the Matlab ndt-library of the
University of Bristol.

Limits:

  - A frame can contain only one probe.
  - In a frame, each ultrasonic scanline is associated to exactly one transmitter and one receiver.

Creating a Frame from scratch::

    import arim
    import numpy

    # Scanline per scanline, the list of transmitters and receivers.
    # For the first scanline, the first element transmits (tx[0] = 0) and
    # also receives (rx[0] = 0).
    # For the second scanline, the first element transmits (tx[1] = 0) and
    # the second element receives (rx[1] = 1).
    # Et caetera.
    # In this example there are 9 scanlines.
    tx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    rx = [0, 1, 2, 0, 1, 2, 0, 1, 2]

    probe = arim.probes['ima_50_MHz_128_1d']

    examination_object = arim.ExaminationObject(arim.Material(6300.))

    time = arim.Time(start=0., step=50e-9, num=1000)

    # The ultrasonic data stored as a matrix. One scanline per row.
    scanlines = np.zeros((len(tx), len(time)))

    frame = arim.Frame(scanlines, time, tx, rx, probe, examination_object)

Remark: the functions :func:`arim.ut.fmc` and :func:`arim.ut.hmc` creates lists of transmitters and receivers for HMC and FMC acquisitions.

To load a frame exported from the Bristol ndt-library (Matlab), see :ref:`io_brain`.
