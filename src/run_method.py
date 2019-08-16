"""
this is an example, how to run the method
"""
import frequencies
import numpy as np
from time import time
#import fremen_wrapper as fremen
import fremen

periods = 2

TS = np.loadtxt('../data/training_03_04_rotated.txt')[:, [0,-1]]

#W = fremen.build_frequencies(60*60*24, 60*60)
P = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=fremen.build_frequencies(60*60*24, 60*60), weights=1.0, return_all=True)
print(1/P)

start = time()
freqs = frequencies.Frequencies(train_path='../data/training_03_04_rotated.txt', edges_of_cell=np.array([3600.0, 1.0, 1.0, np.pi/4.0, 1.0]), structure=[2, [], True], clusters=5)
finish = time()
print('whole creation of model time: ' + str(finish-start))

"""
periodicities = fremen.get_periodicities(path='/home/tom/pro/my/whyte_branches/2020_icra_RAL/data/data_for_visualization/two_weeks_days_nights_weekends_only_ones_times_only.txt', number_of_periods=periods, max_periods=60*60*24*7)
generated_structure = [2, [1.0]*periods, periodicities]
print(generated_structure)

start = time()
freqs = frequencies.Frequencies(train_path='../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt', edges_of_cell=np.array([3600.0, 1.0, 1.0]), structure=generated_structure)
finish = time()
print('whole creation of model time: ' + str(finish-start))


freqs.F = freqs.F / (10.0 * 10.0 * 10.0)
freqs.edges_of_cell = np.array([360.0, 0.1, 0.1])
start = time()
print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt')))
finish = time()
print('whole rmse time: ' + str(finish-start))


############

start = time()
freqs = frequencies.Frequencies(train_path='../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt', edges_of_cell=np.array([360.0, 0.1, 0.1]), structure=generated_structure)
finish = time()
print('whole creation of model time: ' + str(finish-start))


start = time()
print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt')))
finish = time()
print('whole rmse time: ' + str(finish-start))
"""





