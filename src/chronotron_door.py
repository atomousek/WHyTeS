import fremen
import numpy as np
import directions


path  = '../data/training_data.txt'

longest = 14*24*60*60
shortest = 60*60

list_of_periodicities = fremen.build_frequencies(longest, shortest)

#print list_of_periodicities

#print P


dataset = np.loadtxt(path)
new_dataset = []

print dataset.shape

for i in range(dataset.shape[0]):
    if dataset[i, 1] != 0:
        new_dataset.append(dataset[i, :])

print dataset.shape
print len(new_dataset)


new_dataset = np.array(new_dataset)

new_path = 'new_dataset.txt'

np.savetxt(new_path, new_dataset)

print dataset.shape
print new_dataset.shape



models = []
list_of_predictions = []
list_of_errors = []

list_of_fremen_predictions = []
list_of_fremen_errors = []

T = dataset[:, 0]
S = dataset[:, -1]

weights = None

for i in xrange(10):
    print '-----------------------------------------------------------------'
    print 'cycle number ' + str(i)
    new_dataset = np.loadtxt(path)  # dataset which is not for fremen
    print dataset.shape


    P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities)
    print 'most dominant periodicity of this cycle: ' + str(P)
    #print P

    structure = [0, [P], 0]

    #predict for fremen
    #beginning
    print path
    fremen_pred = directions.Directions(1, structure, weights)
    fremen_pred.fit(path)
    transformed_dataset_fremen, target_fremen = fremen_pred.transform_data(path)
    prediction_fremen = fremen_pred.predict(transformed_dataset_fremen)
    list_of_fremen_predictions.append(prediction_fremen)

    error_fremen = fremen_pred.my_rmse(target_fremen, prediction_fremen)    
    list_of_fremen_errors.append(error_fremen)
    weights_fremen = fremen_pred.calc_weights(list_of_fremen_predictions, list_of_fremen_errors)
    print weights_fremen.shape
    S -= weights_fremen
    print weights_fremen.shape
    #end

    circle = directions.Directions(1, structure, weights)
    
    circle.fit(new_path)
    
    models.append(circle)

    transformed_dataset, target = circle.transform_data(new_path)
    
    #print 'transformed dataset size: '+ str(transformed_dataset.shape)
    prediction = circle.predict(transformed_dataset)
    list_of_predictions.append(prediction)
    
    error = circle.my_rmse(target, prediction)
    print 'rmse: ' + str(error)
    list_of_errors.append(error)

    #weights = np.average(list_of_predictions, weights=list_of_errors)
    weights = circle.calc_weights(list_of_predictions, list_of_errors)
    dataset[:, -1] -= weights
    #weights = prediction
   
    new_path = 'dir' + str(i) + '.txt' # new path
    np.savetxt(new_path, dataset)
    print prediction
