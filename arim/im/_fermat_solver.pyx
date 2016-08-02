#cython: boundscheck=False, wraparound=False

from cython cimport size_t

cdef extern from "fermat_solver_c.hpp":
    void c_expand_rays "expand_rays" (
        unsigned int *interior_indices,
        unsigned int *indices_new_interface,
        unsigned int *expanded_indices,
        size_t n, size_t m, size_t p, size_t d)


cpdef object _expand_rays(unsigned int[:,:,::1] interior_indices,
    unsigned int[:,::1] indices_new_interface,
    unsigned int[:,:,::1] expanded_indices,
    size_t n, size_t m, size_t p, size_t d):

    #cdef size_t n = interior_indices.shape[0]
    #cdef size_t d = interior_indices.shape[2]
    #cdef size_t p = indices_new_interface.shape[2]

    #assert indices_new_interface.shape == (n, p)
    #assert expanded_indices.shape == (n, p, d+1)

    c_expand_rays(
        &interior_indices[0,0,0],
        &indices_new_interface[0,0],
        &expanded_indices[0,0,0],
        n, m, p, d)
    return None

#