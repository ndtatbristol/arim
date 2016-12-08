template <typename Tdata, typename Tamp, typename Tfloat>
void delay_and_sum_nearest(
    const Tdata* __restrict scanlines,
    const unsigned int* __restrict tx,
    const unsigned int* __restrict rx,
    const Tfloat* __restrict lookup_times_tx,
    const Tfloat* __restrict lookup_times_rx,
    const Tamp* __restrict amplitudes_tx,
    const Tamp* __restrict amplitudes_rx,
    Tfloat invdt,
    Tfloat t0,
    Tdata fillvalue,
    Tdata* __restrict result,
    size_t numpoints,
    size_t numsamples,
    size_t numelements,
    size_t numscanlines);
