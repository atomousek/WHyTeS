"""
this is an example, how to run the method
"""
import directions
import fremen
import run_testing_method as tm
import numpy as np
from time import time

# parameters for the method
number_of_clusters = 2
#number_of_spatial_dimensions = 2  # known from data
number_of_spatial_dimensions = 4  # france data
#list_of_periodicities = [21600.0, 43200.0, 86400.0, 86400.0*7.0]  # the most prominent periods, found by FreMEn on different data :)
#list_of_periodicities = [4114.285714285715, 10800.0, 5400.0, 4320.0, 7854.545454545455, 604800.0, 12342.857142857143, 7200.0, 86400.0]  # the most prominent periods, found by FreMEn
list_of_periodicities = [86400.0, 604800.0]  # the most prominent periods, found by FreMEn
#list_of_periodicities = []  # for model_of_8angles_0.5m_over_month
angle_and_speed = False  # using velocity vector precalculated in dataset.

structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, angle_and_speed]  # suitable input
# load and train the predictor
start = time()
dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
#dirs = dirs.fit('../data/training_dataset.txt')
dirs = dirs.fit('../data/training_dataset_new_format.txt')
finish = time()
print('time to create model: ' + str(finish-start))

dirs.model2xml()
#np.savetxt('C.txt', dirs.C)
#np.savetxt('W.txt', dirs.W)
#for idx, prec in enumerate(dirs.PREC):
#    np.savetxt('PREC' + str(idx) + '.txt', prec)
#
#X = dirs.transform_data('../data/training_dataset_new_format.txt')[0]
#y = dirs.predict(X)
#np.savetxt('predictions.txt', y)
#data = np.loadtxt('../data/training_dataset_new_format.txt')[:, [0, 2, 3, 6, 5]]
#np.savetxt('test_dataset_to_test.txt', data)



"""
print(dirs.rmse('../data/training_dataset.txt'))

"""

"""
#list_of_times = np.loadtxt('../data/test_times.txt').astype(int)
list_of_times = [0]  # for model_of_8angles_0.5m_over_month

counter = 0
for model_time in list_of_times:
    start = time()
    #out = dirs.model_to_directions_for_kevin_no_time_dimension()
    out = dirs.model_to_directions(model_time)
    #np.savetxt('../results/model_of_8angles_0.5m_over_month.txt', out)  # for model_of_8angles_0.5m_over_month
    #np.savetxt('../results/1_cluster_9_periods/' + str(model_time) + '_model.txt', out)
    #np.savetxt('smaz.txt', out)
    finish = time()
    counter += 1
    print('time to save model number ' + str(counter) + ' for specific time: ' + str(finish-start))
"""
