import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport floor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void rounding_and_indexes(const double[:,:] dataset, const double[:] edges, const long [:] no_bins, const double [:] starting_points, const long rows, const long columns, double [:] out, double [:] indexes, long len_out_minus) nogil: #, double [:] count) nogil:

    cdef long i
    cdef long j
    cdef long k
    cdef long l
    cdef double position
    cdef double tmp

    for i in xrange(rows):
        position = 0
        if dataset[i, columns] != -1:
            for j in xrange(columns):
                indexes[j] = floor((dataset[i, j] - starting_points[j]) / edges[j])
            for k in xrange(columns):
                if indexes[k] < 0.0 or indexes[k] > no_bins[k]-1 :
                    position = -1
                    break
                tmp = indexes[k]
                #print('k: ' + str(k))
                #print('index: ' + str(indexes[k]))
                for l in xrange(k):
                    tmp *= no_bins[l]
                position += tmp
                #print('tmp: ' + str(tmp))
                #if position > 6336:
                #    print('pretekl jsem')
            #print('position: ' + str(position))
            if position >= 0.0 and position <= len_out_minus:
                out[<long>position] += dataset[i, columns]
                #count[<long>position] += 1
                #print('vystupni hodnota: ' + str(out[<long>position]))
            #else:
                #print('mimo')
            #print('')
                
        




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def target(double[:,:] dataset, double [:] edges, long [:] no_bins, double [:] starting_points):
    cdef long rows = dataset.shape[0]
    #print('rows: ' + str(rows))
    cdef long columns = dataset.shape[1] - 1
    #print('columns: ' + str(columns))
    cdef long i
    cdef long len_out = 1
    for i in xrange(len(no_bins)):
        len_out *= no_bins[i]
    #print('len_out: ' + str(len_out))
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.zeros(len_out, dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] count = np.zeros(len_out, dtype=np.float64)
    #print('shape_of_out: ' + str(np.shape(out)))
    cdef cnp.ndarray[cnp.float64_t, ndim=1] indexes = np.empty(columns + 1, dtype=np.float64) # why (+ 1) ?
    #print('shape_of_indexes: ' + str(np.shape(indexes)))
    rounding_and_indexes(dataset, edges, no_bins, starting_points, rows, columns, out, indexes, len_out - 1) #, count)
    return out#, count
    
