import numpy as np
cimport numpy as cnp
cimport cython
from cython_gsl cimport *
from libc.math cimport sin
from libc.math cimport cos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline double expansion(const double [:] edges, const long [:] no_bins, const double [:] starting_points, const double [:] periodicities, const long no_periods, long dim, double[:] bin_centre, const long base_dim, double sum_of_probs, const double [:] C, const double [:,:] PREC, const double PI2, double [:] bin_minus_C) nogil:
    cdef long idx
    cdef long i
    cdef long j
    cdef double tmp
    cdef double distance
    cdef long degrees
    cdef double time
    cdef long id_n_p
    cdef double prob
    if dim > 0:
        for idx in xrange(no_bins[dim]):
            bin_centre[dim-1] = starting_points[dim] + idx * edges[dim]
            sum_of_probs = expansion(edges, no_bins, starting_points, periodicities, no_periods, dim-1, bin_centre, base_dim, sum_of_probs, C, PREC, PI2, bin_minus_C)
    else:
        degrees = base_dim + 2*no_periods - 1
        for idx in xrange(no_bins[dim]):
            time = starting_points[dim] + idx * edges[dim]
            for id_n_p in xrange(no_periods):
                bin_centre[base_dim + 2*id_n_p - 1] = cos(time*PI2/periodicities[id_n_p]) 
                bin_centre[base_dim + 2*id_n_p] = sin(time*PI2/periodicities[id_n_p]) 
            for i in xrange(degrees):
                bin_minus_C[i] = bin_centre[i] - C[i]
            distance = 0.0
            for j in xrange(degrees):
                tmp = 0.0
                for i in xrange(degrees):
                    tmp += PREC[i,j] * bin_minus_C[i]
                tmp *= bin_minus_C[j]
                distance += tmp
            prob = gsl_cdf_chisq_Q(distance, <double>degrees)
            sum_of_probs += prob
    return sum_of_probs
             


@cython.boundscheck(False)
@cython.wraparound(False)
def expand(double [:] edges, long [:] no_bins, double [:] starting_points, double [:] periodicities, long dim, double [:] C, double [:,:] PREC, double PI2, double U):
    cdef double sum_of_probs = 0.0
    cdef long no_periods = len(periodicities)
    cdef double summed_weights
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_centre = np.zeros(dim + 2*no_periods - 1, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_minus_C = np.zeros(dim + 2*no_periods - 1, dtype=np.float64)
    summed_weights = expansion(edges, no_bins, starting_points, periodicities, no_periods, dim-1, bin_centre, dim, sum_of_probs, C, PREC, PI2, bin_minus_C)
    if summed_weights > 0.0:
        return U / summed_weights
    else:
        return 0.0

