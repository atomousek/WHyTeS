import numpy as np
cimport numpy as cnp
cimport cython
from cython_gsl cimport *
from libc.math cimport sin
from libc.math cimport cos
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline long expansion(const double [:] edges, const long [:] no_bins, const double [:] starting_points, const double [:] periodicities, const long no_periods, long dim, double[:] bin_centre, const long base_dim, long counter, const double [:,:] C, const double [:,:,:] PREC, const double PI2, double [:] bin_minus_C, const long no_clusters, double [:] out, double [:] F) nogil:
    cdef long idx
    cdef long i
    cdef long j
    cdef long c
    cdef double tmp
    cdef double distance
    cdef long degrees
    cdef double time
    cdef long id_n_p
    cdef double prob
    if dim > 0:
        for idx in xrange(no_bins[dim]):
            bin_centre[dim-1] = starting_points[dim] + idx * edges[dim]
            counter = expansion(edges, no_bins, starting_points, periodicities, no_periods, dim-1, bin_centre, base_dim, counter, C, PREC, PI2, bin_minus_C, no_clusters, out, F)
    else:
        degrees = base_dim + 2*no_periods - 1
        for idx in xrange(no_bins[dim]):
            time = starting_points[dim] + idx * edges[dim]
            for id_n_p in xrange(no_periods):
                bin_centre[base_dim + 2*id_n_p - 1] = cos(time*PI2/periodicities[id_n_p]) 
                bin_centre[base_dim + 2*id_n_p] = sin(time*PI2/periodicities[id_n_p]) 
            prob = 0.0
            for c in xrange(no_clusters):
                for i in xrange(degrees):
                    bin_minus_C[i] = bin_centre[i] - C[c,i]
                distance = 0.0
                for j in xrange(degrees):
                    tmp = 0.0
                    for i in xrange(degrees):
                        tmp = tmp + PREC[c,i,j] * bin_minus_C[i]
                    tmp = tmp * bin_minus_C[j]
                    distance += tmp
                prob += gsl_cdf_chisq_Q(distance, <double>degrees) * F[c]
            out[counter] = prob
            counter += 1
    return counter
             


@cython.boundscheck(False)
@cython.wraparound(False)
def predict(double [:] edges, long [:] no_bins, double [:] starting_points, double [:] periodicities, long dim, double [:,:] C, double [:,:,:] PREC, double PI2, long no_clusters, double [:] F):
    cdef long counter = 0
    #cdef long no_periods = 2*len(periodicities) - 1
    cdef long no_periods = len(periodicities)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_centre = np.zeros(dim + 2*no_periods - 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_minus_C = np.ones(dim + 2*no_periods - 1, dtype=np.float64) *(-1)
    cdef long i
    cdef long len_out = 1
    for i in xrange(dim):
        len_out *= no_bins[i]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.zeros(len_out, dtype=np.float64)
    cdef long number_of_cells_for_no_use
    number_of_cells_for_no_use = expansion(edges, no_bins, starting_points, periodicities, no_periods, dim-1, bin_centre, dim, counter, C, PREC, PI2, bin_minus_C, no_clusters, out, F)
    return out

