"""
this is an example, how to run the method
"""
import directions

# parameters for the method
number_of_clusters = 3
number_of_spatial_dimensions = 2  # known from data
list_of_periodicities = [21600.0, 43200.0, 86400.0]  # the most prominent periods, found by FreMEn
structure_of_extended_space = [number_of_spatial_dimensions, [1.0]*len(list_of_periodicities), list_of_periodicities]  # suitable input

# load and train the predictor
dirs = directions.Directions(number_of_clusters, structure_of_extended_space, '../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')

# predict values from dataset
# first transform data and get target values
X, target = dirs.transform_data('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
# than predict values
prediction = dirs.predict(X)
# now, you can compare target and prediction in any way

# or calculate RMSE of prediction of values directly
print('RMSE between target and prediction is:')
print(dirs.rmse('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt'))
