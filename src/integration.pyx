import numpy as np
cimport numpy as cnp
cimport cython
#from libc.math cimport pow
from cython_gsl cimport *
#python setup.py build_ext -i

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline double expansion(const double [:] edges, const long [:] boundaries, long dim, double[:] bin_centre, const long max_dim, double sum_of_probs, const double [:] C, const double [:,:] PREC) nogil:
    cdef long idx
    cdef long i
    cdef long j
    cdef double tmp
    cdef double tmp2
    cdef double distance
    cdef double degrees = max_dim
    if dim > 0:
        for idx in xrange(boundaries[dim], boundaries[dim+1]):
            bin_centre[dim] = edges[idx]
            sum_of_probs = expansion(edges, boundaries, dim-1, bin_centre, max_dim, sum_of_probs, C, PREC)
    else:
        for idx in xrange(boundaries[dim], boundaries[dim+1]):
            bin_centre[dim] = edges[idx]
            for i in xrange(max_dim):
                bin_centre[i] -= C[i]
            distance = 0.0
            for j in xrange(max_dim):
                tmp = 0.0
                for i in xrange(max_dim):
                    tmp += PREC[i,j] * bin_centre[i]
                tmp *= bin_centre[j]
                distance += tmp
            #tmp2 = gsl_cdf_chisq_P(distance, degrees)
            #tmp2 = gsl_cdf_gaussian_P(distance, degrees)
            #sum_of_probs += 1
            sum_of_probs += gsl_cdf_chisq_Q(distance, degrees)
    return sum_of_probs
             


@cython.boundscheck(False)
@cython.wraparound(False)
def expand(double [:] edges, long [:] boundaries, long dim, double [:] C, double [:,:] PREC):
    cdef double sum_of_probs = 0.0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_centre = np.empty(dim, dtype=np.float64)
    return expansion(edges, boundaries, dim-1, bin_centre, dim, sum_of_probs, C, PREC)

