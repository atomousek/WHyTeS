import fremen
from directions import my_rmse
import numpy as np
#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#classifiers = []

def transform_data(data, periodicity):
    X = np.empty((data.shape[0], 3))
    X[:, 2] = data[:, 1]
    X[:, 0 : 2] = np.c_[np.cos(data[:, 0] * 2 * np.pi / periodicity), np.sin(data[:, 0] * 2 * np.pi / periodicity)]
    return X


def classify(transformed_dataset, sample_weights):
    clf = LogisticRegression(solver='liblinear', multi_class='ovr', class_weight='balanced').fit(X=transformed_data[:, 0 : 2], y=transformed_data[:, -1], sample_weight=sample_weights)
    prediction = clf.predict(transformed_data[:, 0 : 2])
    #classifiers.append(clf)
    np.savetxt('pred.txt', prediction)
    return prediction

def strong_classify(transformed_dataset,  alphas):
    final_calssification = np.zeros(dataset.shape[0])
    for i in range(len(alphas)):
        #transformed_dataset = transform_data(dataset, periodicities[i])
        final_classification += aplhas[i].classify(transformed_dataset, None)
    return final_classification
        

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

errors  = []

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
P = 86400
for i in range(3):   # for each weak classifier
    print '----------------------------------------------------'
    #P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities, sample_weights)
    print 'found periodicity: ' + str(P)
    transformed_data = transform_data(dataset, P)
    
    classification = classify(transformed_data, sample_weights)
    #weighted_rmse, rmse = my_rmse(dataset[:, -1], classification, sample_weights) 
    #print 'weighted rmse: ' + str(weighted_rmse)   
    error = calc_error(dataset, classification, sample_weights)
    #errors.append(rmse)
    print 'classification error: ' + str(error)
    alpha.append(0.5*np.log((1-error)/error))

    for j in range(dataset.shape[0]):   # for each element in dataset
        sample_weights[j] = sample_weights[j]*np.e**(-1*alpha[i]*classification[j]*dataset[j, 1])
    sample_weights = sample_weights / sum(sample_weights)
    print sum(sample_weights)

#plt.plot(errors)
#plt.show()
transformed_data = transform_data(dataset, 86400)
#strong_classification = np.zeros(dataset.shape[0], dtype=float)
#for i in range(len(classifiers)):
#    strong_classification += alpha[i]*classifiers[i].predict(transformed_data[:, 0: 2])
    #print alpha[i]
#strong_classification = np.sign(strong_classification)
print 'strong classifier classification error: ' + str(calc_error(dataset, strong_classification, np.ones(dataset.shape[0])))
#w_rmse, rmse =  my_rmse(dataset[:, -1], classification, sample_weights)
#print 'rmse: ' + str(rmse)

strong_classification = strong_classify(transformed_data, alpha)

no_of_wrong_ones = 0
no_of_wrong_zeros = 0
no_of_target_ones = 0
for i in range(dataset.shape[0]):
    if (dataset[i, 1] == 1) and (strong_classification[i] == -1):
        no_of_wrong_ones += 1
    if (dataset[i, 1] == -1 ) and (strong_classification[i] == 1):
        no_of_wrong_ones += 1
    if (dataset[i, 1] == 1):
        no_of_target_ones += 1

print 'number of ones classified as zero: ' + str(no_of_wrong_ones)
print 'number of zeros classified as one: ' + str(no_of_wrong_zeros)
print 'number of target ones: ' + str(no_of_target_ones)
print dataset.shape
#print sum(strong_classification)



