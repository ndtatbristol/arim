//#include <math.h>
//#include <iostream>
#include <complex>
#include <cmath>
#include <omp.h>

#include "delay_and_sum_cpu.hpp"

#define CHUNKSIZE 256
/*
Parameters
----------
scanlines : ndarray [numscanlines x numsamples]
tx, rx : ndarray [numscanlines]
    Mapping between the scanlines and the transmitter/receiver.
    Values: integers in [0, numelements[
lookup_times_tx : ndarray [numpoints x numelements]
    Times of flight (floats) between the transmitters and the grid points.
lookup_times_rx : ndarray [numpoints x numelements]
    Times of flight (floats) between the grid points and the receivers.
amplitudes_tx : ndarray [numpoints x numelements]
amplitudes_rx : ndarray [numpoints x numelements]
invdt: inverse of time step (1/dt)
t0: initial time
fillvalue: value to use for a pixel if out of range
result : ndarray [numpoints]
    Result. Must be initialised to zero at the beginning.
numpoints: int number of points in TFM
numsamples: int number of time samples in scanlines
numelements: int number of elements in Probe Array
numscanlines: int number of scanlines - which is 0.5*(numelements*numelements+numelements) for HMC,

Results
-------
Void.
Result is in array ``results``.

*/
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
    size_t numscanlines) {
    /*
    Cf. http://docs.oracle.com/cd/E19059-01/stud.10/819-0501/7_tuning.html
    Shared read-only data are 'firstprivate',
    shared read-write data are 'shared'.
    */
    #pragma omp parallel for firstprivate(scanlines, tx, rx, lookup_times_tx, \
        lookup_times_rx, amplitudes_tx, amplitudes_rx, invdt, \
        t0, fillvalue, \
        numpoints, numsamples, numelements, numscanlines) shared (result) \
        schedule(dynamic, CHUNKSIZE)
    for (int point = 0; point < (int)numpoints; point++) {
        for (unsigned int scan = 0; scan < numscanlines; scan++) {
            int lookup_index;
            size_t t_ind, r_ind;
            Tfloat lookup_time;

            t_ind = numelements * point + tx[scan];
            r_ind = numelements * point + rx[scan];
            lookup_time = lookup_times_tx[t_ind] + lookup_times_rx[r_ind] - t0;
            // NB: the next line is equivalent to:
            // lookup_index = (int)round(lookup_time * invdt );
            // lookup_index = (int)(lookup_time * invdt + 0.5);
            lookup_index = (int)round(lookup_time * invdt);
            if (lookup_index < 0 || lookup_index >= (int)numsamples) {
                result[point] += fillvalue;
            }
            else {
              int data_idx = scan * numsamples + lookup_index;
              Tamp amp_corr = amplitudes_tx[t_ind] * amplitudes_rx[r_ind];
              result[point] += amp_corr * scanlines[data_idx];
            }
        } // end loop over scanlines
    } // end loop over points
}

// Explicit instanciations:
// To modify this: use script delay_and_sum_cpu_helpers.py
template void delay_and_sum_nearest<float, float, float>(
    const float* __restrict scanlines,
    const unsigned int* __restrict tx, const unsigned int* __restrict rx,
    const float* __restrict lookup_times_tx,
    const float* __restrict lookup_times_rx,
    const float* __restrict amplitudes_tx, const float* __restrict amplitudes_rx,
    float invdt, float t0, float fillvalue, float* __restrict results,
    size_t numpoints, size_t numsamples, size_t numelements, size_t numscanlines);
template void delay_and_sum_nearest<std::complex<float>, float, float>(
    const std::complex<float>* __restrict scanlines,
    const unsigned int* __restrict tx, const unsigned int* __restrict rx,
    const float* __restrict lookup_times_tx,
    const float* __restrict lookup_times_rx,
    const float* __restrict amplitudes_tx, const float* __restrict amplitudes_rx,
    float invdt, float t0, std::complex<float> fillvalue, std::complex<float>* __restrict results,
    size_t numpoints, size_t numsamples, size_t numelements, size_t numscanlines);
template void delay_and_sum_nearest<std::complex<float>, std::complex<float>, float>(
    const std::complex<float>* __restrict scanlines,
    const unsigned int* __restrict tx, const unsigned int* __restrict rx,
    const float* __restrict lookup_times_tx,
    const float* __restrict lookup_times_rx,
    const std::complex<float>* __restrict amplitudes_tx, const std::complex<float>* __restrict amplitudes_rx,
    float invdt, float t0, std::complex<float> fillvalue, std::complex<float>* __restrict results,
    size_t numpoints, size_t numsamples, size_t numelements, size_t numscanlines);
template void delay_and_sum_nearest<double, double, double>(
    const double* __restrict scanlines,
    const unsigned int* __restrict tx, const unsigned int* __restrict rx,
    const double* __restrict lookup_times_tx,
    const double* __restrict lookup_times_rx,
    const double* __restrict amplitudes_tx, const double* __restrict amplitudes_rx,
    double invdt, double t0, double fillvalue, double* __restrict results,
    size_t numpoints, size_t numsamples, size_t numelements, size_t numscanlines);
template void delay_and_sum_nearest<std::complex<double>, double, double>(
    const std::complex<double>* __restrict scanlines,
    const unsigned int* __restrict tx, const unsigned int* __restrict rx,
    const double* __restrict lookup_times_tx,
    const double* __restrict lookup_times_rx,
    const double* __restrict amplitudes_tx, const double* __restrict amplitudes_rx,
    double invdt, double t0, std::complex<double> fillvalue, std::complex<double>* __restrict results,
    size_t numpoints, size_t numsamples, size_t numelements, size_t numscanlines);
template void delay_and_sum_nearest<std::complex<double>, std::complex<double>, double>(
    const std::complex<double>* __restrict scanlines,
    const unsigned int* __restrict tx, const unsigned int* __restrict rx,
    const double* __restrict lookup_times_tx,
    const double* __restrict lookup_times_rx,
    const std::complex<double>* __restrict amplitudes_tx, const std::complex<double>* __restrict amplitudes_rx,
    double invdt, double t0, std::complex<double> fillvalue, std::complex<double>* __restrict results,
    size_t numpoints, size_t numsamples, size_t numelements, size_t numscanlines);

int main () {
    return 0;
}

