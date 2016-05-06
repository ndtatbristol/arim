.. highlight:: python

.. _signal:

=================
Signal processing
=================

:mod:`arim.signal` provides common signal processing operations usable with arim core objects.


To call a filter, first initiliase a :class:`arim.signal.Filter` object. The parameters depends on the kind of filter,
for example cutoff frequencies and order. Once initialised, a :class:`arim.signal.Filter` works as a regular function
whose argument is the data to filter::

    from arim.signal import Hilbert
    f_hil = Hilbert()
    filtered_data = f_hil(raw_data)


Filters can be composed::

    from arim.signal import Hilbert
    f_hil = Hilbert()
    f_abs = Abs()

    f_abs_hil = f_abs + f_hil

    filtered_data = f_abs_hil(raw_data)

    # This is equivalent to:
    filtered_data = f_abs(f_hil(raw_data))


To filter scanlines, the recommended approach is to call :meth:`arim.core.frame.Frame.apply_filter`::

    frame.apply_filter(f_abs_hil)

    # filtered scanlines:
    frame.scanlines

    # raw scanlines:
    frame.scanlines_raw


To create a new filter, create a class derived from :meth:`arim.signal.Filter`::

    class PlusSomethingFilter(arim.signal.Filter):
        """A filter that adds a given value to the signals."""

        def __init__(self, value_to_add=1.0):
            self.value_to_add = value_to_add

        def __call__(self, arr):
            return arr + self.value_to_add

    plus_one_filter = PlusSomethingFilter(1.0)
    filtered_data = plus_one_filter(raw_data)



.. seealso::

    Full reference: :mod:`arim.signal`

