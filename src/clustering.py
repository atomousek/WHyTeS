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
X = transformed_data[:, :2]

"""
plt.plot(transformed_data[:, 0], transformed_data[:, 1], 'ro')
plt.ylim(ymax=1.1, ymin=-1.1)
plt.xlim(xmax=1.1, xmin=-1.1)
plt.show()
"""
db = DBSCAN(eps=0.05 , min_samples= 100).fit(transformed_data[:, :2])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels =  db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print 'Estimated number of clusters: %d' % n_clusters_
print 'Estimated number of noise points: %d' % n_noise_

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

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


