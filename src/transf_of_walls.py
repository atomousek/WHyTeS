import numpy as np

dataset = np.loadtxt('../data/boarders_of_space_UTBM.txt')
filtered = dataset[(dataset[:,2]<0.1) & (dataset[:,2]>-1.0), :2]

deg29=0.506145483
sinus = np.sin(deg29)
cosinus = np.cos(deg29)

#R = np.array([[0.891, 0.454],[-0.454,0.891]])
R = np.array([[cosinus, sinus],[-sinus,cosinus]])

rotated = np.dot(R, filtered.T).T

# boarders of the room
wall10 = np.c_[np.ones(1590)*(-9.25), np.arange(1590)*0.01 + 0.1]
wall20 = np.c_[np.ones(1590)*(3.0), np.arange(1590)*0.01 + 0.1]
wall30 = np.c_[np.arange(1225)*0.01 - 9.25, np.ones(1225)*(16.0)]
wall40 = np.c_[np.arange(1225)*0.01 - 9.25, np.ones(1225)*(0.1)]
# elevator
wall11 = np.c_[np.ones(320)*(-6.8), np.arange(320)*0.01 + 12.8]
wall21 = np.c_[np.ones(320)*(-4.7), np.arange(320)*0.01 + 12.8]
wall31 = np.c_[np.arange(210)*0.01 - 6.8, np.ones(210)*(16.0)]
wall41 = np.c_[np.arange(210)*0.01 - 6.8, np.ones(210)*(12.8)]
# column
wall12 = np.c_[np.ones(100)*(-7.0), np.arange(100)*0.01 + 8.0]
wall22 = np.c_[np.ones(100)*(-6.0), np.arange(100)*0.01 + 8.0]
wall32 = np.c_[np.arange(100)*0.01 - 7.0, np.ones(100)*(9.0)]
wall42 = np.c_[np.arange(100)*0.01 - 7.0, np.ones(100)*(8.0)]
# counter
wall13 = np.c_[np.ones(90)*(-1.5), np.arange(90)*0.01 + 0.1]
wall23 = np.c_[np.ones(90)*( 1.0), np.arange(90)*0.01 + 0.1]
wall33 = np.c_[np.arange(250)*0.01 - 1.5, np.ones(250)*(1.0)]
wall43 = np.c_[np.arange(250)*0.01 - 1.5, np.ones(250)*(0.1)]
#extended = np.concatenate((rotated, wall10, wall20, wall30, wall40, wall11, wall21, wall31, wall41, wall12, wall22, wall32, wall42, wall13, wall23, wall33, wall43))
extended = np.concatenate((wall10, wall20, wall30, wall40, wall11, wall21, wall31, wall41, wall12, wall22, wall32, wall42, wall13, wall23, wall33, wall43))


np.savetxt('../data/artificial_boarders_of_space_in_UTBM.txt', extended)


H = np.histogramdd(extended, bins=500)[0]
import matplotlib.pyplot as plt

H = (H>0)*1

plt.imshow(H)
plt.show()
