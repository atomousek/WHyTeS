"""
basic FreMEn to find most influential periodicity, call
chosen_period(T, S, W):
it returns the most influential period in the timeseries, where timeseries are
    the residues between reality and model

for the creation of a list of reasonable frequencies call
build_frequencies(longest, shortest):
where
longest - float, legth of the longest wanted period in default units,
        - usualy four weeks
shortest - float, legth of the shortest wanted period in default units,
         - usualy one hour.
It is necessary to understand what periodicities you are looking for (or what
    periodicities you think are the most influential)
"""

import numpy as np
from collections import defaultdict


def chosen_period(T, S, W, weights=1.0, return_all=False, return_W=False):
    """
    input: T numpy array Nx1, time positions of measured values
           S numpy array Nx1, difference between predicted and measured values
           W numpy array Lx1, sequence of reasonable frequencies
           weights float or numpy array Nx1, weights of every measurement (or error)
           return_all boolean, if True, FreMEn returns all periodicities ordered by prominence
           return_W boolean, if True, FreMEn returns W without chosen frequency
    output: P float64, length of the most influential periodicity in default units
            W numpy array Lx1, sequence of reasonable frequencies without the chosen one
    uses: np.sum(), np.max(), np.absolute()
          complex_numbers_batch(), max_influence()
    objective: to choose the most influencing period in the timeseries, where
               timeseries are the residues between reality and model
    """
    # originally: S = (time_frame_sums - time_frame_freqs)[valid_timesteps]
    # pokus
    S = (S > np.mean(S))*1.0
    G = complex_numbers_batch(T, S, W, weights)
    P = max_influence(W, G, return_all)
    # power spectral density ???
    #print('SUM OF AMPLITUDES: ' + str(np.sum(np.absolute(G))))
    if return_W:
        if return_all:
            print('return_W=True not supported with return_all=True')
            return P, W
        remove = 1.0/P
        idx = np.argmin(np.abs(W - remove))
        if abs(W[idx] - remove) < 1e-14:
            W = W[range(0,idx)+range(idx+1, len(W))]
        else:
            print('cannot remove: ' + str(P))
        return P, W
    else:
        return P#, sum_of_amplitudes


def complex_numbers_batch(T, S, W, weights):
    """
    input: T numpy array Nx1, time positions of measured values
           S numpy array Nx1, sequence of measured values
           W numpy array Lx1, sequence of reasonable frequencies
    output: G numpy array Lx1, sequence of complex numbers corresponding
            to the frequencies from W
    uses: np.e, np.newaxis, np.pi, np.mean()
    objective: to find sparse(?) frequency spectrum of the sequence S
    """
    G = []
    for i in xrange(len(W)):
        Gs = weights * S * (np.e ** (W[i] * T * (-1j) * np.pi * 2))
        G.append(np.mean(Gs))
    G = np.array(G)
    return G


def max_influence(W, G, return_all=False):
    """
    input: W numpy array Lx1, sequence of reasonable frequencies
           G numpy array Lx1, sequence of complex numbers corresponding
                              to the frequencies from W
    output: P float64, length of the most influential frequency in default
                       units
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
    uses: np.absolute(), np.argmax(), np.float64(),np.array()
    objective: to find length of the most influential periodicity in default
               units and return changed list of frequencies
    """
    if return_all:
        idxs = np.argsort(np.absolute(G))
        return 1.0/W[idxs]
    else:
        #maximum_position = np.argmax(np.absolute(G[1:])) + 1
        maximum_position = np.argmax(np.absolute(G))
        influential_frequency = W[maximum_position]
        # not sure if it is necessary now
        if influential_frequency == 0 or np.isnan(np.max(np.absolute(G))):
            print('problems in fremen.max_influence')
            P = np.float64(0.0)
        else:
            P = 1 / influential_frequency
        return P


def build_frequencies(longest, shortest, remove_one=-1.0):  # should be part of initialization of learning
    """
    input: longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
    output: W numpy array Lx1, sequence of frequencies
    uses: np.arange()
    objective: to find frequencies w_0 to w_k
    """
    k = int(longest / shortest)  # + 1
    W = np.float64(np.arange(k) + 1) / float(longest) # removed zero periodicity
    if remove_one != -1:
        idx = np.argmin(W - 1.0/remove_one)
        if W[idx] - remove_one < 1e-14:
            W = W[range(0,idx)+range(idx+1, len(W))]
        else:
            print('cannot remove: ' + str(remove_one))
    return W


