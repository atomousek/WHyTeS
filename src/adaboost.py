import fremen
from directions import my_rmse
import numpy as np
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt

def transform_data(data, periodicity):
    X = np.empty((data.shape[0], 3))
    X[:, 2] = data[:, 1]
    X[:, 0 : 2] = np.c_[np.cos(data[:, 0] * 2 * np.pi / periodicity), np.sin(data[:, 0] * 2 * np.pi / periodicity)]
    return X


def classify(transformed_dataset, sample_weights):
    clf = LogisticRegression(solver='liblinear', multi_class='ovr', class_weight='balanced').fit(X=transformed_data[:, 0 : 2], y=transformed_data[:, -1], sample_weight=sample_weights)
    prediction = clf.predict(transformed_data[:, 0 : 2])

    np.savetxt('pred.txt', prediction)
    return prediction


def calc_error(dataset, classification, sample_weights):
    error = 0
    #wrong_classification = 0
    for i in range(dataset.shape[0]):
        error += sample_weights[i]*(dataset[i, 1] != classification[i])
        #wrong_classification += (dataset[i, 1] != classification[i])
    return error / sum(sample_weights) #, wrong_classification
 

path  = '../data/new_training_data.txt'


longest = 14*24*60*60
shortest = 60*60

list_of_periodicities = fremen.build_frequencies(longest, shortest)

dataset = np.loadtxt(path)

classifier_weights = []

#for i in range(dataset.shape[0]):
#    if dataset[i, 1] == 0:
#        dataset[i, 1] = -1

#dataset.astype(int)
#np.savetxt('new_training_data.txt', dataset)

T = dataset[:, 0]
S = dataset[:, -1]

sample_weights = np.ones(S.shape[0])
sample_weights = sample_weights / S.shape[0]

print sample_weights[0]

#P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities, sample_weights)

#print 'found periodicity: ' + str(P)

#transformed_data = transform_data(dataset, P)
#print transformed_data.shape
#print transformed_data[:, 0:2]
#plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
#plt.show()

"""
clf = LogisticRegression(solver='liblinear', multi_class='ovr', class_weight='balanced').fit(transformed_data[:, 0 : 2], transformed_data[:, -1])
prediction = clf.predict(transformed_data[:, 0 : 2])

np.savetxt('pred.txt', prediction)

error = calc_error(dataset, prediction, sample_weights)
"""

#print prediction
#print clf.score(transformed_data[:, 0 : 2], transformed_data[:, -1])
#print clf.coef_

alpha = []

for i in range(10):   # for each weak classifier
    print '----------------------------------------------------'
    P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities, sample_weights)
    print 'found periodicity: ' + str(P)
    transformed_data = transform_data(dataset, P)
    
    classification = classify(transformed_data, sample_weights)
    rmse = my_rmse(dataset[:, -1], classification) 
    print 'rmse :' + str(rmse)   
    error = calc_error(dataset, classification, sample_weights)
    print 'current error: ' + str(error)
    alpha.append(0.5*np.log((1-error)/error))

    for j in range(dataset.shape[0]):   # for each element in dataset
        sample_weights[j] = sample_weights[j]*np.e**(-1*alpha[i]*classification[j]*dataset[j, 1])
    sample_weights = sample_weights / sum(sample_weights)
    print sum(sample_weights)

