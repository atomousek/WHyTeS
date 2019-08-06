import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport floor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void rounding_and_indexes(const double[:,:] dataset, const double[:] edges, const long [:] no_bins, const double [:] starting_points, const long rows, const long columns, double [:] out, double [:] indexes) nogil:

    cdef long i
    cdef long j
    cdef long k
    cdef long l
    cdef double position
    cdef double tmp

    for i in xrange(rows):
        position = 0
        for j in xrange(columns):
            indexes[j] = floor((dataset[i, j] - starting_points[j]) / edges[j])
        for k in xrange(columns):
            tmp = indexes[k]
            for l in xrange(k):
                tmp *= no_bins[l]
            position += tmp
        out[<long>position] += dataset[i, columns]
                
        




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def target(double[:,:] dataset, double [:] edges, long [:] no_bins, double [:] starting_points):
    cdef long rows = dataset.shape[0]
    cdef long columns = dataset.shape[1] - 1
    cdef long i
    cdef long len_out = 1
    for i in xrange(len(no_bins)):
        len_out *= no_bins[i]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.zeros(len_out, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] indexes = np.empty(columns + 1, dtype=np.float64)
    rounding_and_indexes(dataset, edges, no_bins, starting_points, rows, columns, out, indexes)
    return out
    
