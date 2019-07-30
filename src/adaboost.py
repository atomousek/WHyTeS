import fremen
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

classifiers = []

def sigmoid(x):
    """
        input: numpy array
        objective: applies sigmoid function elementwise
        output: numpy array
    """                                        
    return 1 / (1 + np.exp(-x))


def calc_rmse(target, prediction, sample_weights):   # weighted rmse
    
    """
        input: dataset target values - array
               predicted values - array
               weight of each sample/value - array
        objective: caclulates rmse and weighted rmse
        output: weighted rmse - float
                rmse - float
    """  


        w_rmse = 0
        rmse = 0
        for i in range(prediction.shape[0]):
            tmp_pred = 0
            tmp_target = 0
            if prediction[i] == -1:
                tmp_pred = 0
            else:
                tmp_pred = 1
            if target[i] == -1:
                tmp_target = 0
            else:
                tmp_target = 1
            w_rmse += sample_weights[i]*(tmp_pred - tmp_target)**2.0
            #print rmse
            rmse += (tmp_pred - tmp_target)**2.0
        rmse = rmse / target.shape[0]
        return np.sqrt(w_rmse), np.sqrt(rmse)


def transform_data(data, periodicity):
    """
        input: whole dataset - multidimensional array
               periodicity - integer
        objective: warps dataset 
        output: transformed dataset
    """  
    X = np.empty((data.shape[0], 3))
    X[:, 2] = data[:, 1]
    X[:, 0 : 2] = np.c_[np.cos(data[:, 0] * 2 * np.pi / periodicity), np.sin(data[:, 0] * 2 * np.pi / periodicity)]
    return X


def classify(transformed_data, sample_weights, proba):
     """
        input: transformed data - dataset transformed using transform_data() function - multidimensional array
               sample_weights - weight of each sample/value from dataset - array
               proba - either 1 or 0 - if 0 -> returns classification
                                       else if 1 -> returns probability of given time being in each of classes 
        objective: classify data
        output: if proba is 0 then returns array of classifications
                else if proba is 1 then returns array of probabilities of given times being in class '1'
    """

    if proba == 1:
        clf = LogisticRegression(solver='liblinear', multi_class='ovr', class_weight='balanced').fit(X=transformed_data[:, 0 : 2], y=transformed_data[:, -1], sample_weight=sample_weights)
        prediction = clf.predict_proba(transformed_data[:, 0 : 2])
        prediction = prediction[:, 1]*2 - 1
    elif proba == 0:
        clf = LogisticRegression(solver='liblinear', multi_class='ovr', class_weight='balanced').fit(X=transformed_data[:, 0 : 2], y=transformed_data[:, -1], sample_weight=sample_weights)
        prediction = clf.predict(transformed_data[:, 0 : 2])
    classifiers.append(clf)
    np.savetxt('pred.txt', prediction)
    return prediction


def strong_classify(dataset, alphas, periodicities):
     """
        input: dataset
               aplhas - weigh of each classifier - array
               peridodicities - periodicity for each classifier - array
        objective: combines weak classifiers into one strong classifier
        output: prediction array of probabilites (values between 0 and 1)
    """

    w = np.ones(dataset.shape[0])/dataset.shape[0]
    final_classification = np.zeros(dataset.shape[0])
    for i in range(len(alphas)):
        transformed_dataset = transform_data(dataset, periodicities[i])
        final_classification += alphas[i]*classify(transformed_dataset, None, 1)
    return final_classification
        

def calc_error(dataset, classification, sample_weights):
    """
        input: dataset 
               classification - array
               sample weights - weight of each sample/value - array
        objective: calculates weighted classification error
        output: classification error
    """

    error = 0
    #wrong_classification = 0
    for i in range(dataset.shape[0]):
        error += sample_weights[i]*(dataset[i, 1] != classification[i])
        #wrong_classification += (dataset[i, 1] != classification[i])
    return error / sum(sample_weights) #, wrong_classification
 

def boost(classifiers_number, periodicity = -1):
    """
        ADABOOST
        input: number of weak classifiers you want
               custom periodicity
        objective: calculates weight (alpha) for each weak classifier
        output: weight for each classier - alphas - array
                peridodicity for each classifier - chosen periodicity - array
    """

    longest = 14*24*60*60
    shortest = 60*60
    list_of_periodicities = fremen.build_frequencies(longest, shortest)
    chosen_periodicities = []

    path  = '../data/new_training_data.txt'
    dataset = np.loadtxt(path)
    
    T = dataset[:, 0]
    S = dataset[:, -1]

    sample_weights = np.ones(S.shape[0])
    sample_weights = sample_weights / S.shape[0]
    
    alphas = []
    
    for i in range(classifiers_number):   # for each weak classifier
        print '----------------------------------------------------'
        
        if periodicity == -1:
            P, sum_of_amplitudes = fremen.chosen_period(T, S, list_of_periodicities, sample_weights)
        else:
            P = periodicity
        chosen_periodicities.append(P)
        print 'found periodicity: ' + str(P)
        
        transformed_data = transform_data(dataset, P)
        classification = classify(transformed_data, sample_weights, 0)
        weighted_rmse, rmse = calc_rmse(dataset[:, -1], classification, sample_weights) 
        print 'weighted rmse: ' + str(weighted_rmse)   
        error = calc_error(dataset, classification, sample_weights)
        
        print 'weighted classification error: ' + str(error)
        alphas.append(0.5*np.log((1-error)/error))
        print 'alpha: ' + str(alphas[i])

        for j in range(dataset.shape[0]):   # for each element in dataset
            sample_weights[j] = sample_weights[j]*np.e**(-1*alphas[i]*classification[j]*dataset[j, 1])
        sample_weights = sample_weights / sum(sample_weights)
        
        print sum(sample_weights)
        
    return alphas, chosen_periodicities



"""
transformed_data = transform_data(dataset, 86400)
strong_classification = np.zeros(dataset.shape[0], dtype=float)


for i in range(len(classifiers)):
    strong_classification += alpha[i]*classifiers[i].predict(transformed_data[:, 0: 2])
    print alpha[i]
strong_classification = np.sign(strong_classification)
print 'strong classifier classification error: ' + str(calc_error(dataset, strong_classification, np.ones(dataset.shape[0])))


#strong_classification = strong_classify(transformed_data, alpha)

no_of_wrong_ones = 0
no_of_wrong_zeros = 0
no_of_target_ones = 0

for i in range(dataset.shape[0]):
    if (dataset[i, 1] == 1) and (strong_classification[i] == -1):
        no_of_wrong_ones += 1
    if (dataset[i, 1] == -1 ) and (strong_classification[i] == 1):
        no_of_wrong_zeroes += 1
    if (dataset[i, 1] == 1):
        no_of_target_ones += 1

print 'number of ones classified as zero: ' + str(no_of_wrong_ones)
print 'number of zeros classified as one: ' + str(no_of_wrong_zeros)
print 'number of target ones: ' + str(no_of_target_ones)
print dataset.shape
"""
#print sum(strong_classification)



