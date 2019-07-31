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
cdef inline double expansion(const double [:] edges, const long [:] no_bins, const double [:] starting_points, const double [:] periodicities, const long shift, long dim, double[:] bin_centre, const long max_dim, double sum_of_probs, const double [:] C, const double [:,:] PREC, const double PI2, double [:] bin_minus_C) nogil:
    cdef long idx
    cdef long i
    cdef long j
    cdef double tmp
    cdef double tmp2
    cdef double distance
    cdef double degrees = max_dim + shift
    cdef double time
    cdef long num_per = (shift + 1) / 2
    cdef long id_n_p
    cdef double prob
    if dim > 0:
        #for idx in xrange(no_bins[dim], no_bins[dim+1]):
        #    bin_centre[dim] = edges[idx]
        for idx in xrange(no_bins[dim]):
            bin_centre[dim-1] = starting_points[dim] + idx * edges[dim]
            #print(str(bin_centre[0]) + ' ' + str(bin_centre[1]) + ' ' + str(bin_centre[2]) + ' ' + str(bin_centre[3]) + ' ' + str(bin_centre[4]) + ' ' + str(bin_centre[5]))
            sum_of_probs = expansion(edges, no_bins, starting_points, periodicities, shift, dim-1, bin_centre, max_dim, sum_of_probs, C, PREC, PI2, bin_minus_C)
            #print(str(bin_centre[0]) + ' ' + str(bin_centre[1]) + ' ' + str(bin_centre[2]) + ' ' + str(bin_centre[3]) + ' ' + str(bin_centre[4]) + ' ' + str(bin_centre[5]))
    else:
        #for idx in xrange(no_bins[dim], no_bins[dim+1]):
        #    bin_centre[dim] = edges[idx]
        #print(str(bin_centre[0]) + ' ' + str(bin_centre[1]) + ' ' + str(bin_centre[2]) + ' ' + str(bin_centre[3]) + ' ' + str(bin_centre[4]) + ' ' + str(bin_centre[5]))
        for idx in xrange(no_bins[dim]):
            time = starting_points[dim] + idx * edges[dim]
            for id_n_p in xrange(num_per):
                #bin_centre[2*id_n_p] = gsl_sf_cos(time*PI2/periodicities[id_n_p]) 
                #bin_centre[2*id_n_p + 1] = gsl_sf_sin(time*PI2/periodicities[id_n_p]) 
                bin_centre[max_dim + 2*id_n_p - 1] = cos(time*PI2/periodicities[id_n_p]) 
                bin_centre[max_dim + 2*id_n_p] = sin(time*PI2/periodicities[id_n_p]) 
            #print(str(bin_centre[0]) + ' ' + str(bin_centre[1]) + ' ' + str(bin_centre[2]) + ' ' + str(bin_centre[3]) + ' ' + str(bin_centre[4]) + ' ' + str(bin_centre[5]))
            for i in xrange(max_dim + shift):
                bin_minus_C[i] = bin_centre[i] - C[i]
            #print(str(bin_minus_C[0]) + ' ' + str(bin_minus_C[1]) + ' ' + str(bin_minus_C[2]) + ' ' + str(bin_minus_C[3]) + ' ' + str(bin_minus_C[4]) + ' ' + str(bin_minus_C[5]))
            distance = 0.0
            for j in xrange(max_dim + shift):
                tmp = 0.0
                for i in xrange(max_dim + shift):
                    tmp += PREC[i,j] * bin_minus_C[i]
                tmp *= bin_minus_C[j]
                distance += tmp
            #tmp2 = gsl_cdf_chisq_P(distance, degrees)
            #tmp2 = gsl_cdf_gaussian_P(distance, degrees)
            #sum_of_probs += 1
            prob = gsl_cdf_chisq_Q(distance, degrees)
            #sum_of_probs += gsl_cdf_chisq_Q(distance, degrees)
            sum_of_probs += prob
            #print(str(bin_centre[0]) + ' ' + str(bin_centre[1]) + ' ' + str(bin_centre[2]) + ' ' + str(bin_centre[3]) + ' ' + str(bin_centre[4]) + ' ' + str(bin_centre[5]) + ' ' + str(distance) + ' ' + str(prob) + ' ' + str(time))
            #print(str(bin_minus_C[0]) + ' ' + str(bin_minus_C[1]) + ' ' + str(bin_minus_C[2]) + ' ' + str(bin_minus_C[3]) + ' ' + str(bin_minus_C[4]) + ' ' + str(bin_minus_C[5]) + ' ' + str(distance) + ' ' + str(prob))
    return sum_of_probs
             


@cython.boundscheck(False)
@cython.wraparound(False)
def expand(double [:] edges, long [:] no_bins, double [:] starting_points, double [:] periodicities, long dim, double [:] C, double [:,:] PREC, double PI2):
    cdef double sum_of_probs = 0.0
    cdef long shift = 2*len(periodicities) - 1
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_centre = np.empty(dim + shift, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_centre = np.zeros(dim + shift, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bin_minus_C = np.zeros(dim + shift, dtype=np.float64)
    #print(str(bin_centre[0]) + ' ' + str(bin_centre[1]) + ' ' + str(bin_centre[2]) + ' ' + str(bin_centre[3]) + ' ' + str(bin_centre[4]) + ' ' + str(bin_centre[5]))
    #print('')
    #print(str(starting_points[0]) + ' ' + str(starting_points[1]) + ' ' + str(starting_points[2]))
    return expansion(edges, no_bins, starting_points, periodicities, shift, dim-1, bin_centre, dim, sum_of_probs, C, PREC, PI2, bin_minus_C)

