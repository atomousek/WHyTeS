"""
this is an example, how to run the method
"""
import directions
import fremen
import run_testing_method as tm
import numpy as np
from time import time

# parameters for the method
number_of_clusters = 1
number_of_spatial_dimensions = 4  # france data
movement_included = False  # using velocity vector precalculated in dataset.


TS = np.loadtxt('../data/training_dataset.txt')[:, [0,-1]]
start = time()
W=fremen.build_frequencies(60*60*24*7, 60*60)
P = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=1.0, return_all=True)
finish = time()
print('time to run fremen: ' + str(finish-start))
#print(P)
print('')
number_of_periodicities = 10


for number_of_clusters in xrange(1,2):
    print('####################')
    print('number of clusters: ' +str(number_of_clusters))
    W=fremen.build_frequencies(60*60*24*7, 60*60)
    #P, W = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=1.0, return_W=True)
    #list_of_periodicities = [P]
    list_of_periodicities = []

    for no_periodicities in xrange(number_of_periodicities+1):
        structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input
        # load and train the predictor
        start = time()
        dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
        dirs = dirs.fit('../data/training_dataset.txt')
        finish = time()
        print('time to create model: ' + str(finish-start))
        start = time()
        X, target = dirs.transform_data('../data/training_dataset.txt')
        pred_for_fremen = dirs.predict(X)
        sample_weights = target - pred_for_fremen
        finish = time()
        #start = time()
        #Ps = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=sample_weights, return_all=True)
        #finish = time()
        #print('time to run fremen: ' + str(finish-start) + ' with this order: ')
        #print(Ps)
        #print('time to create sample weights: ' + str(finish-start))
        start = time()
        P, W = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=sample_weights, return_W=True)
        finish = time()
        print('time to run fremen: ' + str(finish-start))
        print('\nnext P: ' + str(P))
        #print 1/W
        list_of_periodicities.append(P)
