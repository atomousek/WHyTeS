import fremen
import directions
import numpy as np


path  = '../data/new_training_data.txt'


longest = 14*24*60*60
shortest = 60*60

list_of_periodicities = fremen.build_frequencies(longest, shortest)

dataset = np.loadtxt(path)


#for i in range(dataset.shape[0]):
#    if dataset[i, 1] == 0:
#        dataset[i, 1] = -1

#dataset.astype(int)
#np.savetxt('new_training_data.txt', dataset)

T = dataset[:, 0]
S = dataset[:, -1]
weights = np.ones(S.shape[0])
weights = weights / S.shape[0]

print weights[0]

P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities, weights)

print P
