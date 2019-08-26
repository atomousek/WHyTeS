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
wall10 = np.c_[np.ones(1270)*(-9.25), np.arange(1270)*0.01 + 0.1]
wall20 = np.c_[np.ones(1270)*(3.0), np.arange(1270)*0.01 + 0.1]
wall30 = np.c_[np.arange(1225)*0.01 - 9.25, np.ones(1225)*(12.8)]
wall40 = np.c_[np.arange(1225)*0.01 - 9.25, np.ones(1225)*(0.1)]
## elevator
#wall11 = np.c_[np.ones(50)*(-6.8), np.arange(50)*0.01 + 12.8]
#wall21 = np.c_[np.ones(50)*(-4.7), np.arange(50)*0.01 + 12.8]
#wall31 = np.c_[np.arange(210)*0.01 - 6.8, np.ones(210)*(13.3)]
#wall41 = np.c_[np.arange(210)*0.01 - 6.8, np.ones(210)*(12.8)]
# column 1
wall12 = np.c_[np.ones(50)*(-6.9), np.arange(50)*0.01 + 8.2]
wall22 = np.c_[np.ones(50)*(-6.4), np.arange(50)*0.01 + 8.2]
wall32 = np.c_[np.arange(50)*0.01 - 6.9, np.ones(50)*(8.7)]
wall42 = np.c_[np.arange(50)*0.01 - 6.9, np.ones(50)*(8.2)]
# column 2
wall14 = np.c_[np.ones(50)*(-2.0), np.arange(50)*0.01 + 8.2]
wall24 = np.c_[np.ones(50)*(-1.5), np.arange(50)*0.01 + 8.2]
wall34 = np.c_[np.arange(50)*0.01 - 2.0, np.ones(50)*(8.7)]
wall44 = np.c_[np.arange(50)*0.01 - 2.0, np.ones(50)*(8.2)]
## small column
#wall15 = np.c_[np.ones(50)*(-1.6), np.arange(50)*0.01 + 12.8]
#wall25 = np.c_[np.ones(50)*(-1.3), np.arange(50)*0.01 + 12.8]
#wall35 = np.c_[np.arange(30)*0.01 - 1.6, np.ones(30)*(13.3)]
#wall45 = np.c_[np.arange(30)*0.01 - 1.6, np.ones(30)*(12.8)]
# counter
wall13 = np.c_[np.ones(90)*(-1.5), np.arange(90)*0.01 + 0.1]
wall23 = np.c_[np.ones(90)*( 1.0), np.arange(90)*0.01 + 0.1]
wall33 = np.c_[np.arange(250)*0.01 - 1.5, np.ones(250)*(1.0)]
wall43 = np.c_[np.arange(250)*0.01 - 1.5, np.ones(250)*(0.1)]
#ALL = np.concatenate((rotated, wall10, wall20, wall30, wall40, wall11, wall21, wall31, wall41, wall12, wall22, wall32, wall42, wall13, wall23, wall33, wall43, wall14, wall24, wall34, wall44, wall15, wall25, wall35, wall45))
#extended = np.concatenate((wall10, wall20, wall30, wall40, wall11, wall21, wall31, wall41, wall12, wall22, wall32, wall42, wall13, wall23, wall33, wall43, wall14, wall24, wall34, wall44, wall15, wall25, wall35, wall45))
ALL = np.concatenate((rotated, wall10, wall20, wall30, wall40, wall12, wall22, wall32, wall42, wall13, wall23, wall33, wall43, wall14, wall24, wall34, wall44))
extended = np.concatenate((wall10, wall20, wall30, wall40, wall12, wall22, wall32, wall42, wall13, wall23, wall33, wall43, wall14, wall24, wall34, wall44))


np.savetxt('../data/artificial_boarders_of_space_in_UTBM.txt', extended)
np.savetxt('../data/boarders_of_space_UTBM_rotated.txt', rotated)
np.savetxt('../data/boarders_of_space_UTBM_rotated_with_artificial.txt', ALL)


He = np.histogramdd(extended, bins=500)[0]
Hr = np.histogramdd(rotated, bins=500)[0]
Ha = np.histogramdd(ALL, bins=500)[0]
import matplotlib.pyplot as plt

He = (He>0)*1
plt.figure(figsize=(10, 10))
plt.imshow(He)
plt.savefig('../data/artificial_boarders_of_space_in_UTBM.png')
plt.close()
Hr = (Hr>0)*1
plt.figure(figsize=(10, 10))
plt.imshow(Hr)
plt.savefig('../data/boarders_of_space_UTBM_rotated.png')
plt.close()
Ha = (Ha>0)*1
plt.figure(figsize=(10, 10))
plt.imshow(Ha)
plt.savefig('../data/boarders_of_space_UTBM_rotated_with_artificial.png')
plt.close()
