import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline long expansion(const double [:] edges, const long [:] no_bins, const double [:] starting_points, long dim, double[:] bin_centre, const long base_dim, long counter, double [:,:] out) nogil:
    cdef long idx
    cdef long i
    if dim > 0:
        for idx in xrange(no_bins[dim]):
            bin_centre[dim] = starting_points[dim] + idx * edges[dim]
            counter = expansion(edges, no_bins, starting_points, dim-1, bin_centre, base_dim, counter, out)
    else:
        for idx in xrange(no_bins[dim]):
            bin_centre[dim]  = starting_points[dim] + idx * edges[dim]
            for i in xrange(base_dim):
                out[counter, i] = bin_centre[i]
            counter += 1
    return counter
             


@cython.boundscheck(False)
@cython.wraparound(False)
def predict(double [:] edges, long [:] no_bins, double [:] starting_points, long dim):
    cdef long counter = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_centre = np.zeros(dim, dtype=np.float64)
    cdef long i
    cdef long len_out = 1
    for i in xrange(dim):
        len_out *= no_bins[i]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.zeros((len_out, dim), dtype=np.float64)
    cdef long number_of_cells_for_no_use
    number_of_cells_for_no_use = expansion(edges, no_bins, starting_points, dim-1, bin_centre, dim, counter, out)
    return out

