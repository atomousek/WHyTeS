import numpy as np
import fremen
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def transform_data(data, periodicity):
    """
        input: whole dataset - multidimensional array
               periodicity - integer
        objective: warps dataset 
        output: transformed dataset
    """  
    X = np.empty((data.shape[0], 3))
    X[:, 2] = data[:, 1]
    X[:, 0 : 2] = np.c_[np.cos(data[:, 0] * 2 * np.pi / periodicity), np.sin(data[:, 0] * 2 * np.pi / periodicity)]
    return X


path = '../data/training_data_ones.txt'
dataset = np.loadtxt(path)

transformed_data = transform_data(dataset, 60*60*24*7)

plt.plot(transformed_data[:, 0], transformed_data[:, 1], 'ro')
plt.ylim(ymax=1.1, ymin=-1.1)
plt.xlim(xmax=1.1, xmin=-1.1)
plt.show()

clustering = DBSCAN(eps=0.05 , min_samples= 100).fit(transformed_data[:, :2])

print clustering.labels_

"""
arr = []
arr0 = []

for i in range(dataset.shape[0]):
    if dataset[i, 1] == 1:
        arr.append(dataset[i, :])
    else:
        arr0.append(dataset[i, :])

arr = np.array(arr)
arr0 = np.array(arr0)

np.savetxt('training_data_ones.txt', arr)
np.savetxt('training_data_zeros.txt', arr0)
"""


