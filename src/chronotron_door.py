import fremen
import numpy as np
import directions
import matplotlib.pyplot as plt


path  = '../data/training_data_short.txt'

longest = 14*24*60*60
shortest = 60*60

list_of_periodicities = fremen.build_frequencies(longest, shortest)

#print list_of_periodicities

#print P


dataset = np.loadtxt(path)
new_dataset = []


for i in range(dataset.shape[0]):
    if dataset[i, 1] != 0:
        new_dataset.append(dataset[i, :])




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

weights = None
weights_fremen = None

for i in xrange(10):
    print '-----------------------------------------------------------------'
    print 'cycle number ' + str(i)
    new_dataset = np.loadtxt(new_path)  # dataset which is not for fremen
    dataset = np.loadtxt(path)
    T = dataset[:, 0]
    S = dataset[:, -1]

    print 'new_dataset shape: ' +str(new_dataset.shape)
    print 'dataset shape: ' + str(dataset.shape)


    P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities)
    print 'most dominant periodicity of this cycle: ' + str(P)
    #print P

    structure = [0, [P], 0]

    #predict for fremen
    #beginning
    fremen_pred = directions.Directions(1, structure, weights_fremen)
    fremen_pred.fit(path)
    transformed_dataset_fremen, target_fremen = fremen_pred.transform_data(path)
    prediction_fremen = fremen_pred.predict(transformed_dataset_fremen)
    list_of_fremen_predictions.append(prediction_fremen)

    error_fremen = fremen_pred.my_rmse(target_fremen, prediction_fremen)    
    list_of_fremen_errors.append(error_fremen)
    weights_fremen = fremen_pred.calc_weights(list_of_fremen_predictions, list_of_fremen_errors)
    #print list_of_fremen_predictions[0].shape     
    #print 'fremen weights shape:' + str(weights_fremen.shape)
    S -= weights_fremen
    dataset[:, -1] -= weights_fremen
    path = 'fremen_dataset.txt'
    np.savetxt(path, dataset)
 
    
    plt.plot(prediction_fremen)
    #plt.scatter(target)
    plt.ylim(ymax = 1.2, ymin = -0.1)
    plt.savefig('srovnani_hodnot_uhly_vse'+str(i)+'.png')
    plt.close()

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
    #print weights.shape
    new_dataset[:, -1] -= weights
    #weights = prediction
   
    new_path = 'dir' + str(i) + '.txt' # new path
    np.savetxt(new_path, new_dataset)
    print prediction
