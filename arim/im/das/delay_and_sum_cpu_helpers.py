if __name__ == '__main__':
    from collections import OrderedDict
    import numpy as np

    print("-" * 80)
    print("Declarations for delay_and_sum_cpu.cpp")
    print("-" * 80)

    datatypes_list = [
        dict(Tfloat='float32', Tamp='float32', Tdata='float32'),
        dict(Tfloat='float32', Tamp='float32', Tdata='complex64'),
        dict(Tfloat='float32', Tamp='complex64', Tdata='complex64'),
        dict(Tfloat='float64', Tamp='float64', Tdata='float64'),
        dict(Tfloat='float64', Tamp='float64', Tdata='complex128'),
        dict(Tfloat='float64', Tamp='complex128', Tdata='complex128')
    ]

    def parse_type(dtype):
        if dtype == 'float32':
            dtype_c = 'float'
            dtype_cython = 'float'
        elif dtype == 'float64':
            dtype_c = 'double'
            dtype_cython = 'double'
        elif dtype == 'complex64':
            dtype_c = 'std::complex<float>'
            dtype_cython = 'float complex'
        elif dtype == 'complex128':
            dtype_c = 'std::complex<double>'
            dtype_cython = 'double complex'
        else:
            raise ValueError
        return dtype_c, dtype_cython

    def populate_datatype_dict(d):
        for keybase in ['Tdata', 'Tfloat', 'Tamp']:
            dtype_c, dtype_cython = parse_type(d[keybase])
            d[keybase + '_c'] = dtype_c
            d[keybase + '_cython'] = dtype_cython
            d[keybase + '_cython2'] = dtype_cython.replace(" ", "")
        d['code'] = '{}_{}_{}'.format(d['Tfloat'], d['Tamp'], d['Tdata'])
    for d in datatypes_list:
        populate_datatype_dict(d)


    declaration = """template void delay_and_sum_nearest<{Tdata_c}, {Tamp_c}, {Tfloat_c}>(
    const {Tdata_c}* __restrict scanlines,
    const unsigned int* __restrict tx, const unsigned int* __restrict rx,
    const {Tfloat_c}* __restrict lookup_times_tx,
    const {Tfloat_c}* __restrict lookup_times_rx,
    const {Tamp_c}* __restrict amplitudes_tx, const {Tamp_c}* __restrict amplitudes_rx,
    {Tfloat_c} invdt, {Tfloat_c} t0, {Tdata_c} fillvalue, {Tdata_c}* __restrict results,
    size_t numpoints, size_t numsamples, size_t numelements, size_t numscanlines);"""

    for datatypes in datatypes_list:
        print(declaration.format(**datatypes))

    print("-" * 80)
    print("Functions for _delay_and_sum_cpu.pyx")
    print("-" * 80)

    cython_funcs = """
cpdef void _delay_and_sum_nearest_{code}(
    const {Tdata_cython}[:, ::1] scanlines,
    const unsigned int[::1] tx,
    const unsigned int[::1] rx,
    const {Tfloat_cython}[:, ::1] lookup_times_tx,
    const {Tfloat_cython}[:, ::1] lookup_times_rx,
    const {Tamp_cython}[:, ::1] amplitudes_tx,
    const {Tamp_cython}[:, ::1] amplitudes_rx,
    {Tfloat_cython} invdt,
    {Tfloat_cython} t0,
    {Tdata_cython} fillvalue,
    {Tdata_cython}[::1] result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines):
    delay_and_sum_nearest[{Tdata_cython2}, {Tamp_cython2}, {Tfloat_cython2}](
        &scanlines[0, 0], &tx[0], &rx[0],
        &lookup_times_tx[0, 0], &lookup_times_rx[0, 0], 
        &amplitudes_tx[0, 0], &amplitudes_rx[0, 0], 
        invdt, t0, fillvalue,
        &result[0],
        numpoints,
        numsamples,
        numelements,
        numscanlines)"""
    for datatypes in datatypes_list:
        print(cython_funcs.format(**datatypes))
