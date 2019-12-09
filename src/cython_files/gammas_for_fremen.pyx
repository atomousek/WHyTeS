import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos
from libc.math cimport sin


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline void mean_gamma(double [:,:] out, double [:] gamma, const double [:] T, const double [:] W, const double PI2, const long lenT, const long lenW) nogil:
    cdef long i
    cdef long j
    cdef double angle
    for i in xrange(lenW):
        gamma[0] = 0
        gamma[1] = 0
        for j in xrange(lenT):
            angle = -W[i] * T[j] * PI2
            gamma[0] += cos(angle)
            gamma[1] += sin(angle)
        out[i, 0] = gamma[0] / lenT
        out[i, 1] = gamma[1] / lenT
                


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate(double [:] T, double [:] W, double PI2):
    cdef long lenT = len(T)
    cdef long lenW = len(W)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] gamma = np.zeros(2, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((lenW,2), dtype=np.float64)
    mean_gamma(out, gamma, T, W, PI2, lenT, lenW)
    return out

