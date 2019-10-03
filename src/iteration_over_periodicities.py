"""
for testing different parameters, do not use :)
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
#start = time()
#W=fremen.build_frequencies(60*60*24*7, 60*60)
#P = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=1.0, return_all=True)
#finish = time()
#print('time to run fremen: ' + str(finish-start))
#print(P)
#print('')
number_of_periodicities = 3
list_of_times = np.loadtxt('../data/test_times.txt').astype(int)


#for number_of_clusters in xrange(1, 7):
for number_of_clusters in xrange(3, 4):
    print('####################')
    print('number of clusters: ' +str(number_of_clusters))
    W=fremen.build_frequencies(60*60*24*7, 60*60)
    #P, W = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=1.0, return_W=True)
    #list_of_periodicities = [P]
    list_of_periodicities = []

    for no_periodicities in xrange(0,number_of_periodicities+1):
        structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input
        # load and train the predictor
        start = time()
        dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
        dirs = dirs.fit('../data/training_dataset.txt')
        finish = time()
        print('time to create model: ' + str(finish-start))
        if finish-start > 100.0:
            print('this time looks too long, I will try to recalculate the model')
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
        print('time to create sample weights: ' + str(finish-start))
        #start = time()
        #Ps = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=sample_weights, return_all=True)
        #finish = time()
        #print('time to run fremen: ' + str(finish-start) + ' with this order: ')
        #print(Ps)

        #list_of_times = np.loadtxt('../data/test_times.txt').astype(int)
        #list_of_times = [0]  # for model_of_8angles_0.5m_over_month

        #if (number_of_clusters == 5 and no_periodicities < 9) or (no_periodicities in [6, 7, 8]):
        if no_periodicities > 1:
            start_all = time()
            counter = 0
            for model_time in list_of_times:
                start = time()
                #out = dirs.model_to_directions_for_kevin_no_time_dimension()
                #print('C: ' + str(dirs.C))
                #print('PREC: ' + str(dirs.PREC))
                #print('clusters: ' + str(dirs.clusters))
                #print('structure: ' + str(dirs.structure))
                out = dirs.model_to_directions(model_time)
                #print('C: ' + str(dirs.C))
                #print('PREC: ' + str(dirs.PREC))
                #print('clusters: ' + str(dirs.clusters))
                #print('structure: ' + str(dirs.structure))
                #np.savetxt('../results/model_of_8angles_0.5m_over_month.txt', out)  # for model_of_8angles_0.5m_over_month

                #np.savetxt('../results/' + str(number_of_clusters) + '_clusters_' + str(no_periodicities) + '_periodicities/' + str(model_time) + '_model.txt', out)
                # pokus
                np.savetxt('../results/euc_' + str(number_of_clusters) + '_clusters_' + str(no_periodicities) + '_periodicities/' + str(model_time) + '_model.txt', out)
                finish = time()
                counter += 1
                if counter%100.0 >= 99.0:
                    print('time to save model number ' + str(counter) + ' for specific time: ' + str(finish-start))
            finish_all = time()
            print('time to save models: ' + str(finish_all-start_all))

        start = time()
        P, W = fremen.chosen_period(T=TS[:,0], S=TS[:,1], W=W, weights=sample_weights, return_W=True)
        finish = time()
        print('time to run fremen: ' + str(finish-start))
        print('\nnext P: ' + str(P))
        #print 1/W
        list_of_periodicities.append(P)

