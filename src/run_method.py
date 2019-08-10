"""
this is an example, how to run the method
"""
import frequencies
import numpy as np
from time import time

start = time()
freqs = frequencies.Frequencies(train_path='../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt', edges_of_cell=np.array([3600.0, 0.1, 0.1]))
finish = time()
print('whole creation of model time: ' + str(finish-start))


start = time()
print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt')))
finish = time()
print('whole rmse time: ' + str(finish-start))




