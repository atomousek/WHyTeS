"""
this is an example, how to run the method
"""
import directions
import fremen

import numpy as np
from time import time

# parameters for the method
number_of_clusters = 1
#number_of_spatial_dimensions = 2  # known from data
number_of_spatial_dimensions = 4  # france data
#list_of_periodicities = [21600.0, 43200.0, 86400.0, 86400.0*7.0]  # the most prominent periods, found by FreMEn
#list_of_periodicities = []
#list_of_periodicities = [86400.0]  # the most prominent periods, found by FreMEn
#movement_included = True  # True, if last two columns of dataset are phi and v, i.e., the angle and speed of human.
movement_included = False  # using velocity vector precalculated in dataset.

#structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input

TS = np.loadtxt('../data/training_dataset.txt')[:, [0,-1]]
start = time()
W=fremen.build_frequencies(60*60*24, 60*60)
P = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=1.0, return_all=True)
finish = time()
print('time to run fremen: ' + str(finish-start))
print(P)


for number_of_clusters in xrange(1,11):
    print('\n######################\nnumber_of_clusters: ' + str(number_of_clusters))
    list_of_periodicities = []

    for no_periodicities in xrange(1, 5):
        print('\nlist_of_periodicities: ' + str(list_of_periodicities))
        structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input
        #for i in xrange(4):
        #print(i)
        # load and train the predictor
        start = time()
        dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
        #dirs = dirs.fit('../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
        dirs = dirs.fit('../data/training_dataset.txt')
        finish = time()
        print('time to create model: ' + str(finish-start))

        start = time()
        print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/test_dataset.txt')))
        finish = time()
        print('time to calculate RMSE: ' + str(finish-start))
        
        X, target = dirs.transform_data('../data/training_dataset.txt')
        pred_for_fremen = dirs.predict(X)
        sample_weights = target - pred_for_fremen
        P, W = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=sample_weights, return_W=True)
        print P
        print 1/W
        list_of_periodicities.append(P)
        print len(W)


"""
# predict values from dataset
# first transform data and get target values
#X, target = dirs.transform_data('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
X, target = dirs.transform_data('../data/test_dataset.txt')
# than predict values
prediction = dirs.predict(X)
# now, you can compare target and prediction in any way, for example RMSE
print('manually calculated RMSE: ' + str(np.sqrt(np.mean((prediction - target) ** 2.0))))

# or calculate RMSE of prediction of values directly
#print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')))
print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/test_dataset.txt')))

out = dirs.model_to_directions_for_kevin_no_time_dimension()
print(np.sum(out[:,-1]))
np.savetxt('../data/model_of_8angles_0.5m_over_month.txt', out)

"""


