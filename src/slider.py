import numpy as np
import multiprocessing as mp
import ctypes

path = '../results/outliers.txt'
data = np.loadtxt(path)
unshared = np.array(data.copy())
shared_result = mp.Array(ctypes.c_double, unshared.reshape(-1))
print len(shared_result.get_obj())
#result = data.copy()


# def callback(a):
#     global result
#     result = a
#     return result


def travel_around_the_cell (i, time, length, time_max, time_interval, x_interval, y_interval):
    global data
    global z
    j = i
    total_k = 0
    total_lambda = 0
    #print str(i) + ' / ' + str(length)
    while j < len(data[:, 0]) and i < len(data[:, 0]) and data[j, 0] <= time + time_interval:

        if time + time_interval >= time_max:
            break
            return 0
        #print 'j = ' + str(j)
        if abs( data[i, 1] - data[j, 1] ) < x_interval and abs( data[i, 2] - data[j, 2] ) < y_interval:
            total_k += data[j, 4]
            total_lambda += data[j, 5]
        j += 1

    shared_result[(i + z)*6 + 0] = time + time_interval/2
    shared_result[(i + z)*6 + 1] = data[i, 1]
    shared_result[(i + z)*6 + 2] = data[i, 2]
    shared_result[(i + z)*6 + 4] = total_k
    shared_result[(i + z)*6 + 5] = total_lambda
    return i


def travel_in_time(i, time, length, time_max, time_interval=7200.):
    global data
    result = np.zeros((0, 6))
    global z
    j = i
    total_k = 0
    total_lambda = 0

    print str(i) + ' / ' + str(length)


    while j < len(data[:, 0]) and i < len(data[:, 0]) and data[j, 0] <= time + time_interval:

        if time + time_interval >= time_max:
            return result
        #print 'j = ' + str(j)
        if abs( data[i, 1] - data[j, 1] ) < x_interval and abs( data[i, 2] - data[j, 2] ) < y_interval:
            total_k += data[j, 4]
            total_lambda += data[j, 5]
        j += 1

    result[i + z, 0] = time + 450.
    result[i + z, 1] = data[i, 1]
    result[i + z, 2] = data[i, 2]
    result[i + z, 4] = total_k
    result[i + z, 5] = total_lambda
    return result


#def slide(time_interval, x_interval, y_interval):

time_interval = 3600.
x_interval = 1.
y_interval = 1.


length = len(data[:, 0])
time_max = np.max(data[:, 0])
z = 0
while data[0, 0] + time_interval >= data[z, 0]:
    z += 1



pool = mp.Pool(mp.cpu_count())
#for i in enumerate(data[:, 0]):
results = [pool.apply_async(travel_around_the_cell, args=(i, data[i,0], length, time_max, time_interval, x_interval, y_interval)) for i in xrange(length)]

pool.close()
pool.join()
print length
#np.frombuffer(shared_result.get_obj()).reshape(data.shape)
np.savetxt('../results/slided_cells.txt', np.frombuffer(shared_result.get_obj()).reshape(data.shape))


#slide(time_interval, x_interval, y_interval)
