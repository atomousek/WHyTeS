import fremen
import directions
import numpy as np


path  = '../data/training_data.txt'


longest = 14*24*60*60
shortest = 60*60

list_of_periodicities = fremen.build_frequencies(longest, shortest)

dataset = np.loadtxt(path)


for i in range(dataset.shape[0]):
    if dataset[i, 1] == 0:
        dataset[i, 1] = -1

np.savetxtx('new_training_data.txt', dataset)
