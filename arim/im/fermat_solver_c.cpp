#include <omp.h>

#include "fermat_solver_c.hpp"

// indices for 2D/3D contiguous arrays
#define IDX_2(I1, I2, D1, D2)  ((I1)*(D2) + (I2))
#define IDX_3(I1, I2, I3, D1, D2, D3)  ((I1)*(D2)*(D3) + (I2)*(D3) + (I3))

#define CHUNKSIZE 4

/*
Expand the rays by one interface knowing the beginning of the rays and the
points the rays must go through at the last interface.

A0, A1, ..., A(d+1) are (d+2) interfaces.

n: number of points of interface A0
m: number of points of interface Ad
p: number of points of interface A(d+1)

Arrays layout must be contiguous.

Output: out_ray

Parameters
----------
interior_indices: *interior* indices of rays going from A(0) to A(d).
    Shape: (n, m, d)
indices_new_interface: indices of the points of interface A(d) that the rays
starting from A(0) cross to go to A(d+1).
    Shape: (n, p)
expanded_indices: OUTPUT
    Shape (n, p, d+1)

*/
void expand_rays(
    const unsigned int * __restrict interior_indices,
    const unsigned int * __restrict indices_new_interface,
    unsigned int * __restrict expanded_indices,
    const size_t n, const size_t m, const size_t p, const size_t d) {

    #pragma omp parallel firstprivate(n, m, p, d, interior_indices, indices_new_interface) shared(expanded_indices)
    {
        #pragma omp for schedule(static, CHUNKSIZE)
        for (int i=0; i<n; ++i) {
            for (int j=0; j<p; ++j) {
                // get the point on interface A(d) to which the ray goes 
                const unsigned int idx = (size_t)indices_new_interface[IDX_2(i, j, n, p)];

                // recopy the head of ray
                for (int k=0; k<d; ++k) {
                    expanded_indices[IDX_3(i, j, k, n, p, d+1)] = interior_indices[IDX_3(i, idx, k, n, m, d)];
                }

                // and add the last point:
                expanded_indices[IDX_3(i, j, d, n, p, d+1)] = (unsigned int)idx;
            }
        }
    }
}
