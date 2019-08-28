import numpy as np

data = np.loadtxt('test_dataset_all.txt')
list_of_times = np.loadtxt('time_windows/testing_times.txt')

for time in list_of_times:
    part = data[(data[:,0] > time - 10.0) & (data[:,0] < time + 50.0), :]
    np.savetxt('time_windows/' + str(int(time)) + '_test_data.txt', part)
