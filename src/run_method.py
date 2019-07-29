from adaboost import transform_data, boost, strong_classify, calc_error, calc_rmse
from sklearn.linear_model import LogisticRegression
import numpy as np 

alphas, chosen_periodicities = boost(5)

path  = '../data/new_training_data.txt'
dataset = np.loadtxt(path)

transformed_data = transform_data(dataset, 86400)
#strong_classification = np.zeros(dataset.shape[0], dtype=float)

"""
for i in range(len(classifiers)):
    strong_classification += alphas[i]*classifiers[i].predict(transformed_data[:, 0: 2])
    print alphas[i]
strong_classification = np.sign(strong_classification)
print 'strong classifier classification error: ' + str(calc_error(dataset, strong_classification, np.ones(dataset.shape[0])))
"""

strong_classification = strong_classify(dataset, alphas, chosen_periodicities)
strong_classification = np.sign(strong_classification)

no_of_wrong_ones = 0
no_of_wrong_zeros = 0
no_of_target_ones = 0

for i in range(dataset.shape[0]):
    if (dataset[i, 1] > 0) and (strong_classification[i] < 0):
        no_of_wrong_ones += 1
    if (dataset[i, 1] < 0) and (strong_classification[i] > 0):
        no_of_wrong_zeros += 1
    if (dataset[i, 1] == 1):
        no_of_target_ones += 1

print '----------------------------------------------------'
print 'RESULTS'
print 'number of ones classified as zero: ' + str(no_of_wrong_ones)
print 'number of zeros classified as one: ' + str(no_of_wrong_zeros)
print 'number of target ones: ' + str(no_of_target_ones)
print 'strong classifier classification error: ' + str(calc_error(dataset, strong_classification, np.ones(dataset.shape[0])))
w_rmse_s, rmse_s = calc_rmse(dataset[:, -1], strong_classification, np.ones(dataset.shape[0])/dataset.shape[0])
print 'rmse: ' + str(rmse_s)
print dataset.shape
