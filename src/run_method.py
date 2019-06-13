"""
this is an example, how to run the method
"""
import directions
import numpy as np

# parameters for the method
number_of_clusters = 3
number_of_spatial_dimensions = 2  # known from data
list_of_periodicities = [21600.0, 43200.0, 86400.0]  # the most prominent periods, found by FreMEn
movement_included = True  # True, if last two columns of dataset are phi and v, i.e., the angle and speed of human.
structure_of_extended_space = [number_of_spatial_dimensions, list_of_periodicities, movement_included]  # suitable input

# load and train the predictor
dirs = directions.Directions(clusters=number_of_clusters, structure=structure_of_extended_space)
dirs = dirs.fit('../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')

# predict values from dataset
# first transform data and get target values
X, target = dirs.transform_data('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
# than predict values
prediction = dirs.predict(X)
# now, you can compare target and prediction in any way, for example RMSE
print('manually calculated RMSE: ' + str(np.sqrt(np.mean((prediction - target) ** 2.0))))

# or calculate RMSE of prediction of values directly
print('RMSE between target and prediction is: ' + str(dirs.rmse('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')))
