import numpy as np

dataset = np.loadtxt('../data/raw_data/03-04_no_head.csv', delimiter=',')
time = dataset[:,0] / 1000000000.0
x = dataset[:,1]
y = dataset[:,2]
x_vel = dataset[:,8]
y_vel = dataset[:,9]
ones = np.ones_like(time)

R = np.array([[0.891, 0.454],[-0.454,0.891]])

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

np.savetxt('../data/training_03_04_rotated.txt', data)
