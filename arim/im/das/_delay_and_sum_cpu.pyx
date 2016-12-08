cimport cython
from cython cimport floatcomplex, doublecomplex, size_t

cdef extern from "delay_and_sum_cpu.hpp":
    cdef void delay_and_sum_nearest[Tdata, Tamp, Tfloat](
        const Tdata* scanlines,
        const unsigned int* tx,
        const unsigned int* rx,
        const Tfloat* lookup_times_tx,
        const Tfloat* lookup_times_rx,
        const Tamp* amplitudes_tx,
        const Tamp* amplitudes_rx,
        Tfloat invdt,
        Tfloat t0,
        Tdata fillvalue,
        Tdata* result,
        size_t numpoints,
        size_t numsamples,
        size_t numelements,
        size_t numscanlines)

# ==================================================================
# To modify the lines below: use script delay_and_sum_cpu_helpers.py
# ==================================================================
cpdef void _delay_and_sum_nearest_float32_float32_float32(
    const float[:, ::1] scanlines,
    const unsigned int[::1] tx,
    const unsigned int[::1] rx,
    const float[:, ::1] lookup_times_tx,
    const float[:, ::1] lookup_times_rx,
    const float[:, ::1] amplitudes_tx,
    const float[:, ::1] amplitudes_rx,
    float invdt,
    float t0,
    float fillvalue,
    float[::1] result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines):
    delay_and_sum_nearest[float, float, float](
        &scanlines[0, 0], &tx[0], &rx[0],
        &lookup_times_tx[0, 0], &lookup_times_rx[0, 0], 
        &amplitudes_tx[0, 0], &amplitudes_rx[0, 0], 
        invdt, t0, fillvalue,
        &result[0],
        numpoints,
        numsamples,
        numelements,
        numscanlines)

cpdef void _delay_and_sum_nearest_float32_float32_complex64(
    const float complex[:, ::1] scanlines,
    const unsigned int[::1] tx,
    const unsigned int[::1] rx,
    const float[:, ::1] lookup_times_tx,
    const float[:, ::1] lookup_times_rx,
    const float[:, ::1] amplitudes_tx,
    const float[:, ::1] amplitudes_rx,
    float invdt,
    float t0,
    float complex fillvalue,
    float complex[::1] result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines):
    delay_and_sum_nearest[floatcomplex, float, float](
        &scanlines[0, 0], &tx[0], &rx[0],
        &lookup_times_tx[0, 0], &lookup_times_rx[0, 0], 
        &amplitudes_tx[0, 0], &amplitudes_rx[0, 0], 
        invdt, t0, fillvalue,
        &result[0],
        numpoints,
        numsamples,
        numelements,
        numscanlines)

cpdef void _delay_and_sum_nearest_float32_complex64_complex64(
    const float complex[:, ::1] scanlines,
    const unsigned int[::1] tx,
    const unsigned int[::1] rx,
    const float[:, ::1] lookup_times_tx,
    const float[:, ::1] lookup_times_rx,
    const float complex[:, ::1] amplitudes_tx,
    const float complex[:, ::1] amplitudes_rx,
    float invdt,
    float t0,
    float complex fillvalue,
    float complex[::1] result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines):
    delay_and_sum_nearest[floatcomplex, floatcomplex, float](
        &scanlines[0, 0], &tx[0], &rx[0],
        &lookup_times_tx[0, 0], &lookup_times_rx[0, 0], 
        &amplitudes_tx[0, 0], &amplitudes_rx[0, 0], 
        invdt, t0, fillvalue,
        &result[0],
        numpoints,
        numsamples,
        numelements,
        numscanlines)

cpdef void _delay_and_sum_nearest_float64_float64_float64(
    const double[:, ::1] scanlines,
    const unsigned int[::1] tx,
    const unsigned int[::1] rx,
    const double[:, ::1] lookup_times_tx,
    const double[:, ::1] lookup_times_rx,
    const double[:, ::1] amplitudes_tx,
    const double[:, ::1] amplitudes_rx,
    double invdt,
    double t0,
    double fillvalue,
    double[::1] result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines):
    delay_and_sum_nearest[double, double, double](
        &scanlines[0, 0], &tx[0], &rx[0],
        &lookup_times_tx[0, 0], &lookup_times_rx[0, 0], 
        &amplitudes_tx[0, 0], &amplitudes_rx[0, 0], 
        invdt, t0, fillvalue,
        &result[0],
        numpoints,
        numsamples,
        numelements,
        numscanlines)

cpdef void _delay_and_sum_nearest_float64_float64_complex128(
    const double complex[:, ::1] scanlines,
    const unsigned int[::1] tx,
    const unsigned int[::1] rx,
    const double[:, ::1] lookup_times_tx,
    const double[:, ::1] lookup_times_rx,
    const double[:, ::1] amplitudes_tx,
    const double[:, ::1] amplitudes_rx,
    double invdt,
    double t0,
    double complex fillvalue,
    double complex[::1] result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines):
    delay_and_sum_nearest[doublecomplex, double, double](
        &scanlines[0, 0], &tx[0], &rx[0],
        &lookup_times_tx[0, 0], &lookup_times_rx[0, 0], 
        &amplitudes_tx[0, 0], &amplitudes_rx[0, 0], 
        invdt, t0, fillvalue,
        &result[0],
        numpoints,
        numsamples,
        numelements,
        numscanlines)

cpdef void _delay_and_sum_nearest_float64_complex128_complex128(
    const double complex[:, ::1] scanlines,
    const unsigned int[::1] tx,
    const unsigned int[::1] rx,
    const double[:, ::1] lookup_times_tx,
    const double[:, ::1] lookup_times_rx,
    const double complex[:, ::1] amplitudes_tx,
    const double complex[:, ::1] amplitudes_rx,
    double invdt,
    double t0,
    double complex fillvalue,
    double complex[::1] result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines):
    delay_and_sum_nearest[doublecomplex, doublecomplex, double](
        &scanlines[0, 0], &tx[0], &rx[0],
        &lookup_times_tx[0, 0], &lookup_times_rx[0, 0], 
        &amplitudes_tx[0, 0], &amplitudes_rx[0, 0], 
        invdt, t0, fillvalue,
        &result[0],
        numpoints,
        numsamples,
        numelements,
        numscanlines)
