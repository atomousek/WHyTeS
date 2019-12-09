"""
for testing different parameters, do not use :)
"""
#!/usr/bin/python

import sys
#import directions
import directions_my_data as directions
import fremen
import numpy as np
from time import time
import os

# parameters for the method
number_of_spatial_dimensions = 4  # france data
#number_of_spatial_dimensions = 2  # france data
movement_included = False  # using velocity vector precalculated in dataset.

# time [ms] (unixtime + milliseconds/1000), person id, position x [mm], position y [mm], position z (height) [mm], velocity [mm/s], angle of motion [rad], facing angle [rad]
#TS = np.loadtxt('../data/training_dataset.txt')[:, [0,-1]]
#dataset_path = '../data/training_dataset_new_format.txt'
dataset_path = '../data/training_dataset.txt'
T = np.loadtxt(dataset_path)[:, 0]
number_of_periodicities = 21  # max number of periodicities
#list_of_times = np.loadtxt('../data/test_times.txt').astype(int)



#for number_of_clusters in xrange(1, 7):
for number_of_clusters in xrange(1, 2):
    print('####################')
    print('number of clusters: ' +str(number_of_clusters))
    W=fremen.build_frequencies(60*60*24*7, 60*60)
    list_of_periodicities = []
    #list_of_periodicities = [86400.0, 604800.0, 4320.0, 21600.0, 120960.0]

    for no_periodicities in xrange(0,number_of_periodicities+1):
    #for no_periodicities in xrange(number_of_periodicities,number_of_periodicities+1):
        structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input
        # load and train the predictor
        start = time()
        dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
        dirs = dirs.fit(dataset_path)
        finish = time()
        print('time to create model: ' + str(finish-start))
        if finish-start > 100.0:
            print('this time looks too long, I will try to recalculate the model')
            start = time()
            dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
            dirs = dirs.fit(dataset_path)
            finish = time()
            print('time to create model: ' + str(finish-start))

        # calculation of error for fremen
        start = time()
        X = dirs.transform_data(dataset_path)
        finish = time()
        print('time to transform data: ' + str(finish-start))
        start = time()
        pred_for_fremen = dirs.predict(X)
        finish = time()
        print('time to predict: ' + str(finish-start))

        # rmse
        #print(np.sqrt(np.mean((pred_for_fremen - target) ** 2.0)))
        print(np.sqrt(np.mean((pred_for_fremen - 1.0) ** 2.0)))

        """
        # create the directory (copied from https://thispointer.com/how-to-create-a-directory-in-python/ )
        dirName = '../results/default_' + str(number_of_clusters) + '_clusters_' + str(no_periodicities) + '_periodicities/'
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")  

        #testing on the testing datasets
        if no_periodicities > -1:
            start_all = time()
            counter = 0
            for model_time in list_of_times:
                start = time()
                out = dirs.model_to_directions(model_time)
                np.savetxt(dirName + str(model_time) + '_model.txt', out)
                finish = time()
                counter += 1
                if counter%100.0 >= 99.0:
                    print('time to save model number ' + str(counter) + ' for specific time: ' + str(finish-start))
            finish_all = time()
            print('time to save models: ' + str(finish_all-start_all))
        """

        start = time()
        sample_weights = 1.0 - pred_for_fremen
        #sample_weights = pred_for_fremen
        #P, W = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=sample_weights, return_W=True)
        #P, W = fremen.chosen_period(T=T, S=sample_weights, W=W, weights=1.0, return_W=True)
        #P, W = fremen.chosen_period(T=T, S=sample_weights, W=W, return_W=True)
        P = fremen.chosen_period(T=T, S=sample_weights, W=W, return_W=False)

        #all_P = fremen.chosen_period(T=T, S=sample_weights, W=W, weights=1.0, return_all=True)
        finish = time()
        print('time to run fremen: ' + str(finish-start))
        #print(all_P)
        #break
        print('\nnext P: ' + str(P))
        #print 1/W
        list_of_periodicities.append(P)

