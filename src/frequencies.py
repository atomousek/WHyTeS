from sklearn.mixture import GaussianMixture
#import transformation as tr
#import transformation_with_dirs as tr
#import grid
import full_grid as fg
import integration
import prediction_over_grid
import target_over_grid
import generate_full_grid

import multiprocessing as mp


import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

from time import clock, time

import copy_reg
import types
import multiprocessing


def _pickle_method(m):
    '''
    copied from https://github.com/stratospark/food-101-keras/issues/10
    '''

    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)



class Frequencies:


    def __init__(self, clusters=3,
                 edges_of_cell = np.array([1200.0, 0.5, 0.5, np.pi/4.0, 0.5]),
                 structure=[2, [86400.0, 604800.0], True],
                 train_path = '../data/two_weeks_days_nights_weekends.txt'):
        self.clusters = clusters
        self.edges_of_cell = edges_of_cell
        self.structure = structure
        self.C, self.F, self.PREC = self._create_model(train_path)


    def _create_model(self, path):
        C, U, PREC = self._get_params(path)
        start = time()
        F = self._calibration_test(path, C, U, PREC)
        finish = time()
        print('time for calibration: ' + str(finish - start))
        return C, F, PREC


    def _get_params(self, path):
        X = self._create_X(self._get_data(path))
        start = time()
        clf = GaussianMixture(n_components=self.clusters, max_iter=500).fit(X)
        finish = time()
        print('time for clustering: ' + str(finish-start))
        labels = clf.predict(X)
        PREC = self._recalculate_precisions(X, labels)
        U = np.sum(clf.predict_proba(X), axis=0)
        C = clf.means_
        return C, U, PREC


    def _get_data(self, path):
        """
        """
        dataset = np.loadtxt(path)
        training_data = dataset[dataset[:, -1] > 0, 0: -1]
        training_data_values = dataset[dataset[:, -1] > 0, -1]
        # !!! training data values must be whole numbers !!!
        if np.max(training_data_values) > 1.0:
            print('training data must be expanded')
            new_training = []
            for i in range(len(training_data_values)):
                how_many = training_data_values[i]
                if how_many > 1.0:
                    copied_data = training_data[i]
                    for j in range(int(np.ceil(how_many - 1))):
                        new_training.append(copied_data)
            training_data = np.r_[training_data, np.array(new_training)]
        return training_data


    def _recalculate_precisions(self, X, labels):
        dim = X.shape[1]
        COV = np.empty((self.clusters, dim, dim))
        for i in xrange(self.clusters):
            COV[i] = np.cov(X[labels == i].T)
        return np.linalg.inv(COV)


    def _get_density_test(self,C, U, PREC, edges, no_bins, starting_points, periodicities, dim, PI2):
        weights_sum = integration.expand(edges, no_bins, starting_points, periodicities, dim, C, PREC, PI2)
        # this should be part of integration
        if weights_sum == 0:
            density = 0
        else:
            density = U / weights_sum
        #with np.errstate(divide='raise'):
        #    try:
        #        density = U / weights_sum
        #    except FloatingPointError:
        #        print('vahy se souctem 0 nebo nevim')
        #        print('sum of weights')
        #        print weights_sum
        #        print('sum of U')
        #        print(U)
        #        density = 0
        return density


    def _calibration_test(self, path, C, U, PREC):
        """
        fast and working - will be used as benchmark
        """
        # params
        X = np.loadtxt(path)[:, :-1]
        bins_and_ranges = fg.get_bins_and_ranges(X, self.edges_of_cell)
        no_bins = np.array(bins_and_ranges[0])
        starting_points = np.array(bins_and_ranges[1])[:,0] + (self.edges_of_cell * 0.5)
        edges = self.edges_of_cell
        periodicities = np.array(self.structure[1])
        dim = np.shape(X)[1]
        PI2 = np.pi*2

        start = time()
        copy_reg.pickle(types.MethodType, _pickle_method)
        pool = mp.Pool(min(self.clusters, mp.cpu_count()))

        results = [pool.apply_async(self._get_density_test, (C[i], U[i], PREC[i], edges, no_bins, starting_points, periodicities, dim, PI2)) for i in xrange(self.clusters)]

        pool.close()
        pool.join()

        F = [r.get() for r in results]
        F = np.array(F)
        finish = time()
        print('integration using multi-cores: ' + str(finish-start))

        #start = time()
        #F = []
        #for i in xrange(self.clusters):
        #    F.append(self._get_density_test(C[i], U[i], PREC[i], edges, no_bins, starting_points, periodicities, dim, PI2))
        #F = np.array(F)
        #finish = time()
        #print('integration using for loop: ' + str(finish-start))

        return F


    def _get_distrubition(self, idx, X):
        return self.F[idx] * self._prob_of_belong(X, self.C[idx], self.PREC[idx])


    def predict_probabilities(self, X):
        """
        there are two problems of thos function:
        1) this shoud be helper method
        2) real _predict_probabilities accessible by user has to be prepared to:
            a) input of 1 to n vectors
            b) input of only "raw" vectors (there should be projection of X into hypertime)
        """
        copy_reg.pickle(types.MethodType, _pickle_method)
        pool = mp.Pool(min(self.clusters, mp.cpu_count()))

        results = [pool.apply_async(self._get_distrubition, (i, X)) for i in xrange(self.clusters)]

        pool.close()
        pool.join()

        DISTR = [r.get() for r in results]
        DISTR = np.array(DISTR)
        return DISTR.sum(axis=0)


    def _prob_of_belong(self, X, C, PREC):
        """
        massively inspired by:
        https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor
        """
        X_C = X - C
        c_dist_x = np.sum(np.dot(X_C, PREC) * X_C, axis=1)
        return 1 - st.chi2.cdf(c_dist_x, len(C))


    def transform_data_over_grid(self, path):
        gridded, target = fg.get_full_grid(np.loadtxt(path), self.edges_of_cell)
        X = self._create_X(gridded)
        return X, target


    def rmse(self, path, plot_it=False):
        start = time()
        y, target = self.predict_probabilities_test(path)
        finish = time()
        print('time for predidiction and reality(K and Lambda): ' + str(finish-start))
        print(np.sum(y))
        print(np.sum(target))
        if plot_it:
            ll = len(target)
            plt.scatter(range(ll), target, color='b', marker="s", s=1, edgecolor="None")
            plt.scatter(range(ll), y, color='g', marker="s", s=1, edgecolor="None")
            #plt.ylim(ymax = 1.2, ymin = -0.1)
            #plt.xlim(xmax = ll, xmin = 0)
            plt.savefig('srovnani_hodnot_uhly_vse.png')
            plt.close()
        return np.sqrt(np.mean((y - target) ** 2.0))


    def poisson(self, path):
        X, K = self._transform_data_over_grid(path)
        Lambda = self._predict_probabilities(X)
        probs = st.poisson.cdf(K, Lambda)
        probs[(probs>0.94) & (K==0)] = 0.5
        ################## only for one-time visualisation #########
        gridded = fg.get_full_grid(np.loadtxt(path), self.edges_of_cell)[0]
        labels = np.zeros_like(probs)
        labels[probs<0.05] = -1
        labels[probs>0.95] = 1
        out = np.c_[gridded, labels]
        np.savetxt('../results/outliers.txt', out)
        return probs
        


    def predict_probabilities_test(self, path):
        # params
        dataset = np.loadtxt(path)
        bins_and_ranges = fg.get_bins_and_ranges(dataset[:, :-1], self.edges_of_cell)
        no_bins = np.array(bins_and_ranges[0])
        starting_points = np.array(bins_and_ranges[1])[:,0]
        starting_points_plus = starting_points + (self.edges_of_cell * 0.5)
        periodicities = np.array(self.structure[1])
        dim = np.shape(dataset)[1] - 1
        PI2 = np.pi*2

        #copy_reg.pickle(types.MethodType, _pickle_method)

        #pool = mp.Pool(1)
        #results_y = [pool.apply_async(prediction_over_grid.predict, (self.edges_of_cell, no_bins, starting_points_plus, periodicities, dim, self.C, self.PREC, PI2, self.clusters, self.F)) for i in xrange(1)]
        #results_target = [pool.apply_async(target_over_grid.target, (dataset, self.edges_of_cell, no_bins, starting_points)) for i in xrange(1)]
        #pool.close()
        #pool.join()

        #y = [r.get() for r in results_y][0]
        #target = [r.get() for r in results_target][0]

        y = prediction_over_grid.predict(self.edges_of_cell, no_bins, starting_points_plus, periodicities, dim, self.C, self.PREC, PI2, self.clusters, self.F)

        target = target_over_grid.target(dataset, self.edges_of_cell, no_bins, starting_points)

        """
        starting_points = np.array(bins_and_ranges[1])[:,0]
        pool = mp.Pool(1)
        results = [pool.apply_async(target_over_grid.target, (dataset, self.edges_of_cell, no_bins, starting_points)) for i in xrange(1)]
        pool.close()
        pool.join()

        target = [r.get() for r in results][0]
        """
        return y, target


    def get_full_grid(self, path, save=True):
        # params
        dataset = np.loadtxt(path)
        bins_and_ranges = fg.get_bins_and_ranges(dataset[:, :-1], self.edges_of_cell)
        no_bins = np.array(bins_and_ranges[0])
        starting_points = np.array(bins_and_ranges[1])[:,0]
        starting_points_plus = starting_points + (self.edges_of_cell * 0.5)
        dim = np.shape(dataset)[1] - 1
        grid = generate_full_grid.predict(self.edges_of_cell, no_bins, starting_points_plus, dim)
        if save:
            np.savetxt('../results/full_grid.txt', grid)
            return None
        else:
            return grid



    def _create_X(self, data):
        """
        objective:
            return data of one class projected into the warped hypertime space
        input: 
            data numpy array nxd*, matrix of measures IRL, where d* is number of measured variables
                                   the first column need to be timestamp
                                   last two columns can be phi and v, the angle and speed of human
        output: 
            X numpy array nxd, data projected into the warped hypertime space
        """
        dim = self.structure[0]
        wavelengths = self.structure[1]
        X = np.empty((len(data), dim + (len(wavelengths) * 2) + self.structure[2]*2))
        X[:, : dim] = data[:, 1: dim + 1]
        for Lambda in wavelengths:
            X[:, dim: dim + 2] = np.c_[np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                       np.sin(data[:, 0] * 2 * np.pi / Lambda)]
            dim = dim + 2

        if self.structure[2]:
            X[:, dim: dim + 2] = np.c_[data[:, -1] * np.cos(data[:, -2]),
                                       data[:, -1] * np.sin(data[:, -2])]
        return X
