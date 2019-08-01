import numpy as np
import multiprocessing as mp

path = '../results/outliers.txt'
data = np.loadtxt(path)
result = data.copy()


def callback(a):
    global result
    result = a
    return result


def travel_around_the_cell (i, time, length, time_max, time_interval=7201., x_interval=2.1, y_interval=2.1):
    global data
    global result
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
    return i


def slide(time_interval, x_interval, y_interval):
    length = len(data[:, 0])
    time_max = np.max(data[:, 0])
    z = 0
    while data[0, 0] + time_interval >= data[z, 0]:
        z += 1



    pool = mp.Pool(mp.cpu_count())
    for i, row in enumerate(data[:, 0]):
        result_list = [pool.apply_async(travel_around_the_cell, args=(i, row, length, time_max, x_interval, y_interval), callback=callback)]

    pool.close()
    pool.join()

    np.savetxt('../results/slided_cells.txt', result)

time_interval = 7201.
x_interval = 2.1
y_interval = 2.1
slide(time_interval, x_interval, y_interval)
