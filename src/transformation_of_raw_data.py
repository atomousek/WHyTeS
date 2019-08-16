import numpy as np

"""
default data include column of ids and first row of head. It has to be deleted.
cut -d, -f2 --complement data.csv > 03-04_no_head.csv   (in raw_data directory)
sometimes, there is also BOM - when opening 03-04_no_head.csv with mousepad, we can delete first line and uncheck Document/Write_Unicode_BOM
"""

dataset = np.loadtxt('../data/raw_data/03-04_no_head.csv', delimiter=',')
time = dataset[:,0] / 1000000000.0
x = dataset[:,1]
y = dataset[:,2]
x_vel = dataset[:,4]
y_vel = dataset[:,5]
ones = np.ones_like(time)

deg29=0.506145483
sinus = np.sin(deg29)
cosinus = np.cos(deg29)
#R = np.array([[0.891, 0.454],[-0.454,0.891]])
R = np.array([[cosinus, sinus],[-sinus,cosinus]])

vec = np.c_[x, y]
vel = np.c_[x_vel, y_vel]

#print(vec)
vec_t = np.dot(R, vec.T).T
#print(vec_t)
vel_t = np.dot(R, vel.T).T

v = np.sqrt(np.sum(vel_t ** 2, axis=1))
#v2 = np.sqrt(np.sum(vel ** 2, axis=1))
#print(v)
#print(v2)
phi = np.sign(vel_t[:,1])*np.arccos(vel_t[:,0]/v)



data = np.c_[time, vec_t, phi, v, ones]

filtered = data[(data[:,1]>-9.25) & (data[:,1]<3.0) & (data[:,2]>0.0) & (data[:,2]<16.0), :]

np.savetxt('../data/training_03_04_rotated.txt', filtered)
