# Created on Fri Jun  2 13:52:33 2017
# @author: tom

"""
There are three used functions, the first loads data from file, the second
transforms these data to the requested space-hypertime and the third function
returns directory of this file.
Other functions were built only for testing.

loading_data(path):
loads data from file on adress 'path' of structure (t, x, y, ...),
    the first dimension (variable, column) is understood as a time,
    others are measured variables in corresponding time.
    If there is only one column in the dataset, it is understood as positive
    occurences in measured times. Expected separator between values is SPACE
    (' ').

create_X(data, structure): create X from a loaded data as a data in hypertime,
                           where the structure of a space is derived from
                           the varibale structure
where
input: data numpy array nxd*, matrix of measures IRL, where d* is number
       of measured variables
       structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
and
output: X numpy array nxd, matrix of measures in hypertime
"""

import numpy as np

def create_X(data, structure):
    """
    input: data numpy array nxd*, matrix of measures IRL, where d* is number
                                  of measured variables
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: X numpy array nxd, matrix of measures in hypertime
    uses: np.empty(), np.c_[]
    objective: to create X as a data in hypertime, where the structure
               of a space is derived from the varibale structure
    """
    # for every period one circe, 
    # data: input data, 
    dim = structure[0]
    radii = structure[1]
    wavelengths = structure[2]
    X = np.empty((len(data), dim + len(radii) * 2))
    X[:, : dim] = data[:, 1: dim + 1]
    for period in range(len(radii)):
        r = radii[period]
        Lambda = wavelengths[period]
        X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                   r * np.sin(data[:, 0] * 2 * np.pi / Lambda)]
        dim = dim + 2
    return X
