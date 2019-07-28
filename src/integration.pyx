import numpy as np
cimport numpy as cnp
cimport cython
#from libc.math cimport pow
from cython_gsl cimport *
from libc.math cimport sin
from libc.math cimport cos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline double expansion(const double [:] edges, const long [:] no_bins, const double [:] starting_points, const double [:] periodicities, const long shift, long dim, double[:] bin_centre, const long max_dim, double sum_of_probs, const double [:] C, const double [:,:] PREC, const double PI2) nogil:
    cdef long idx
    cdef long i
    cdef long j
    cdef double tmp
    cdef double tmp2
    cdef double distance
    cdef double degrees = max_dim
    cdef double time
    cdef long num_per = (shift + 1) / 2
    cdef long id_n_p
    if dim > 0:
        #for idx in xrange(no_bins[dim], no_bins[dim+1]):
        #    bin_centre[dim] = edges[idx]
        for idx in xrange(no_bins[dim]):
            bin_centre[dim+shift] = starting_points[dim] + idx * edges[dim]
            sum_of_probs = expansion(edges, no_bins, starting_points, periodicities, shift, dim-1, bin_centre, max_dim, sum_of_probs, C, PREC, PI2)
    else:
        #for idx in xrange(no_bins[dim], no_bins[dim+1]):
        #    bin_centre[dim] = edges[idx]
        for idx in xrange(no_bins[dim]):
            time = starting_points[dim] + idx * edges[dim]
            for id_n_p in xrange(num_per):
                #bin_centre[2*id_n_p] = gsl_sf_cos(time*PI2/periodicities[id_n_p]) 
                #bin_centre[2*id_n_p + 1] = gsl_sf_sin(time*PI2/periodicities[id_n_p]) 
                bin_centre[2*id_n_p] = cos(time*PI2/periodicities[id_n_p]) 
                bin_centre[2*id_n_p + 1] = sin(time*PI2/periodicities[id_n_p]) 
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
            sum_of_probs += 1 - gsl_cdf_chisq_Q(distance, degrees)
    return sum_of_probs
             


@cython.boundscheck(False)
@cython.wraparound(False)
def expand(double [:] edges, long [:] no_bins, double [:] starting_points, double [:] periodicities, long dim, double [:] C, double [:,:] PREC, double PI2):
    cdef double sum_of_probs = 0.0
    cdef long shift = 2*len(periodicities) - 1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_centre = np.empty(dim + shift, dtype=np.float64)
    return expansion(edges, no_bins, starting_points, periodicities, shift, dim-1, bin_centre, dim, sum_of_probs, C, PREC, PI2)

