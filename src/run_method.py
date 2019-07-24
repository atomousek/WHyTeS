"""
this is an example, how to run the method
"""
import frequencies
import numpy as np

# parameters for the method
#number_of_clusters = 3
#number_of_spatial_dimensions = 2  # known from data
#list_of_periodicities = [21600.0, 43200.0, 86400.0]  # the most prominent periods, found by FreMEn

# load and train the predictor
freqs = frequencies.Frequencies(edges_of_cell = np.array([60.0, 1.0, 1.0]))
#freqs = freqs.fit('../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')

# predict values from dataset
# first transform data and get target values
X, target = freqs.transform_data('../data/wednesday_thursday_nights.txt')
# than predict values
prediction = freqs.predict(X)
# now, you can compare target and prediction in any way, for example RMSE
print('manually calculated RMSE: ' + str(np.sqrt(np.mean((prediction - target) ** 2.0))))

# or calculate RMSE of prediction of values directly
print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/wednesday_thursday_nights.txt')))

# and now, something copletely defferent
probs = freqs.poisson('../data/wednesday_thursday_nights.txt')
print("poisson prosel!")

print("pocet malych outlieru: " + str(len(probs[probs<0.05])))
print("pocet velkych outlieru: " + str(len(probs[probs>0.95])))
print("pocet vsech hodnot: " + str(len(probs)))
