.. _glossary:

Glossary
========

..
    Comment:
    To refer to a term, use :term:`something`. Optionaly: :term:`Something <something>` to render a capital letter.


.. glossary::
    :sorted:

    acquisition instrument
        Electronic device that allows the UT expert to control a probe and record the raw signals.
        When connected to computer, the instrument is the physical interface between the computer
        and the probe.

    examination object
        Component examined in order to detect defects.

    frame
        Set of :term:`timetraces <timetrace>` obtained at a given position of the probes. An
        inspection is usually made of several frames. See also: :ref:`frame`. Examples:

        - A :term:`FMC` frame is a frame.
        - A FMC frame where one timetrace is missing is a frame but is not a FMC frame.

    FMC
        Full Matrix Capture.

    GCS
        Global Coordinate System. In the GCS, the default imaging plane is given by the equation $y=0$.
        
    HMC
        Half Matrix Capture.

    image
        Output of imaging algorithm. An image attempts to represent the examination object in such a
        way that the UT expert is able to confirm or invalidate the presence of a defect, or to
        characterize the defect. See also :term:`image array` and :term:`image plot`.

    image array
        Internal representation of an image. An image array is given by scalar values (real or complex)
        on a regular grid (2D or 3D).

    image plot
        Visual representation of an :term:`image array` (*i.e.* a rectangular set of colored
        pixels). An image plot is always 2D.

    image point
        Image point

    imaging algorithm
        Set of operations which creates from one or more frames an image of the examination object.

    index axis
        Axis normal to probe during an inspectiog, i.e. axis towards where the probe is transmitting.

    MFMC
        Multi Frame Capture Format. File format.

    multi-view TFM
        Variant of :term:`TFM`

    ndarray
        Multidimensional array. This is the base datatype provided by the library ``numpy`` to
        perform optimized operations on numeric data in Python.
        See also: `numpy.ndarray <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_.

    PCS
        Probe Cordinate System. **Warning:** PCS does not refer to
        *Probe Center Separation*.

    scan axis
        Axis along which the probe is moved during an inspection.

    timetrace
        Electrical signal over time obtained from exactly one transmitter and one receiver (possibly
        the same).

    TFM
        Total Focusing Method.

    UT
        Ultrasonic testing. UT is a family technique of non-destructive testing based on the
        propagation of ultrasonic waves in the object or material tested.


