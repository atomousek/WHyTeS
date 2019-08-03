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
#freqs = frequencies.Frequencies(train_path='../data/trenovaci_dva_tydny.txt', edges_of_cell=np.array([3600.0, 1.0, 1.0]))
freqs = frequencies.Frequencies(train_path='../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt', edges_of_cell=np.array([3600.0, 0.1, 0.1]))
#freqs = freqs.fit('../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')


print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt')))
#print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/trenovaci_dva_tydny.txt')))
#print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/testovaci_dva_dny.txt')))

"""
# predict values from dataset
# first transform data and get target values
#X, target = freqs.transform_data('../data/wednesday_thursday_nights.txt')
X, target = freqs.transform_data('../data/testovaci_dva_dny.txt')
# than predict values
prediction = freqs.predict(X)
# now, you can compare target and prediction in any way, for example RMSE
print('manually calculated RMSE: ' + str(np.sqrt(np.mean((prediction - target) ** 2.0))))

# or calculate RMSE of prediction of values directly
print('RMSE between target and prediction is: ' + str(freqs.rmse('../data/testovaci_dva_dny.txt')))

# and now, something copletely defferent
#probs = freqs.poisson('../data/testovaci_dva_dny.txt')
probs = freqs.poisson('../data/data_for_visualization/wednesday_thursday_days_nights_only_ones.txt')
#probs = freqs.poisson('../data/trenovaci_dva_tydny.txt')
print("poisson prosel!")

print("pocet malych outlieru: " + str(len(probs[probs<0.05])))
print("pocet velkych outlieru: " + str(len(probs[probs>0.95])))
print("pocet vsech hodnot: " + str(len(probs)))
"""
