from adaboost import transform_data, boost, strong_classify, calc_error, calc_rmse, sigmoid
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt 

alphas, chosen_periodicities = boost(5)

path  = '../data/new_training_data.txt'
dataset = np.loadtxt(path)

transformed_data = transform_data(dataset, 86400)

strong_classification = strong_classify(dataset, alphas, chosen_periodicities)
np.savetxt('strong_classification.txt', strong_classification)
strong_classification = np.sign(strong_classification)
#strong_classification = sigmoid(strong_classification)

no_of_wrong_ones = 0
no_of_wrong_zeros = 0
no_of_target_ones = 0
zeros = 0
ones = 0

for i in range(dataset.shape[0]):
    if (dataset[i, 1] > 0) and (strong_classification[i] < 0):
        no_of_wrong_ones += 1
    if (dataset[i, 1] < 0) and (strong_classification[i] > 0):
        no_of_wrong_zeros += 1
    if (dataset[i, 1] == 1):
        no_of_target_ones += 1
    if strong_classification[i] < 0:
        zeros += 1
    if strong_classification[i] > 0:
        ones += 1

#plt.plot(dataset[:, -1])
#plt.plot(strong_classification)

plt.subplot(2, 1, 1)
plt.plot(dataset[:40000, -1])

plt.subplot(2, 1, 2)
plt.plot(strong_classification[:40000])

plt.show()

#plt.show()

plt.savefig('plot.png')
plt.close()

print zeros
print ones
print '----------------------------------------------------'
print 'RESULTS'
print 'number of target ones classified as zero: ' + str(no_of_wrong_ones)
print 'number of target zeros classified as one: ' + str(no_of_wrong_zeros)
print 'number of target ones: ' + str(no_of_target_ones)
print 'classification error: ' + str(calc_error(dataset, strong_classification, np.ones(dataset.shape[0])))
w_rmse_s, rmse_s = calc_rmse(dataset[:, -1], strong_classification, np.ones(dataset.shape[0])/dataset.shape[0])
print 'rmse: ' + str(rmse_s)
print dataset.shape
