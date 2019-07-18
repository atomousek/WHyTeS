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


models = []
weights = None

for i in xrange(10):
    print 'cycle number ' + str(i)
    #dataset = np.loadtxt(path)
    print dataset.shape

    T = dataset[:, 0]
    S = dataset[:, -1]

    P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities)
    print 'most dominant periodicity of this cycle: ' + str(P)
    #print P

    structure = [0, [P], 0]
    circle = directions.Directions(1, structure, weights)
    print 1
    circle.fit(path)
    print 2
    models.append(circle)

    transformed_dataset, target = circle.transform_data(path)
    
    print 'transformed dataset size: '+ str(transformed_dataset.shape)
    prediction = circle.predict(transformed_dataset)
    #dataset[:, -1] -= prediction
    weights = prediction
   
    output_name = 'dir' + str(i) + '.txt' # new path
    np.savetxt(output_name, weights)
    print prediction
