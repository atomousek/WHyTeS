import numpy as np
import frequencies
import full_grid as fg
import scipy.stats as st
import multiprocessing as mp


edges_of_cell=np.array([600.0, 0.5, 0.5])
freqs = frequencies.Frequencies(train_path='../data/data_for_visualization/two_weeks_days_nights_weekends_only_ones.txt',
                                edges_of_cell=edges_of_cell)

path = '../data/data_for_visualization/wednesday_thursday_days_nights_only_ones.txt'

data = np.loadtxt('../results/slided_cells.txt')

K = data[:, 4]
Lambda = data[:, 5]

#probs = st.poisson.cdf(K, Lambda)
probs = st.poisson.cdf(K, Lambda)
probs[(probs>0.94) & (K==0)] = 0.5
################## only for one-time visualisation #########
gridded = fg.get_full_grid(np.loadtxt(path), edges_of_cell)[0]
labels = np.zeros_like(probs)
labels[probs<0.05] = -1
labels[probs>0.95] = 2
out = np.c_[gridded, labels]
np.savetxt('../results/outliers.txt', out)


# probs = freqs.poisson('../data/data_for_visualization/wednesday_thursday_days_nights_only_ones.txt')
# #probs = freqs.poisson('../data/trenovaci_dva_tydny.txt')
# print("poisson prosel!")
#
# print("pocet malych outlieru: " + str(len(probs[probs<0.05])))
# print("pocet velkych outlieru: " + str(len(probs[probs>0.95])))
# print("pocet vsech hodnot: " + str(len(probs)))
