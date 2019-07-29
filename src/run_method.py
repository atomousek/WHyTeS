from adaboost import transform_data, boost, strong_classify
from sklearn.linear_model import LogisticRegression
import numpy as np 

alphas = boost(3)

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

strong_classification = strong_classify(transformed_data, alphas)

no_of_wrong_ones = 0
no_of_wrong_zeros = 0
no_of_target_ones = 0

for i in range(dataset.shape[0]):
    if (dataset[i, 1] == 1) and (strong_classification[i] == -1):
        no_of_wrong_ones += 1
    if (dataset[i, 1] == -1) and (strong_classification[i] == 1):
        no_of_wrong_zeroes += 1
    if (dataset[i, 1] == 1):
        no_of_target_ones += 1

print 'number of ones classified as zero: ' + str(no_of_wrong_ones)
print 'number of zeros classified as one: ' + str(no_of_wrong_zeros)
print 'number of target ones: ' + str(no_of_target_ones)
print dataset.shape
