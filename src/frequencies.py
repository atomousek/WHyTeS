from sklearn.mixture import GaussianMixture
import transformation as tr
import grid
#import fremen
#import transformation_with_dirs as tr
#import grid_with_directions as grid


import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import matplotlib.patches as pat
import pandas as pd


class Frequencies:


    def __init__(self, clusters=3,
                 edges_of_cell = np.array([1200.0, 0.5, 0.5]),
                 structure=[2, [1.0, 1.0], [86400.0, 604800.0]],
                 train_path = '../data/two_weeks_days_nights_weekends.txt'):
        self.clusters = clusters
        self.edges_of_cell = edges_of_cell
        self.structure = structure
        self.C, self.F, self.PREC = self._create_model(train_path)


    def _create_model(self, path):
        C, U, COV, PREC = self._get_params(path)
        F = self._calibration(path, C, U, PREC)
        return C, F, PREC


    def _get_params(self, path):
        X = tr.create_X(self._get_data(path), self.structure)
        clf = GaussianMixture(n_components=self.clusters, max_iter=500).fit(X)
        labels = clf.predict(X)
        PREC, COV = self._recalculate_precisions(X, labels)
        U = clf.predict_proba(X)
        C = clf.means_
        return C, U.T, COV, PREC


    #def _get_data(self, path):
    #    dataset = np.loadtxt(path)
    #    return dataset[dataset[:, -1] == 1, : -1]
    def _get_data(self, path):
        """
        """
        dataset = np.loadtxt(path)
        #all_data = dataset[:, 0: -1]
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
                    #for j in xrange(int(how_many - 1)):
                    for j in range(int(np.ceil(how_many - 1))):
                        new_training.append(copied_data)
            #print(np.shape(np.array(new_training)))
            training_data = np.r_[training_data, np.array(new_training)]
        return training_data


    def _recalculate_precisions(self, X, labels):
        COV = []
        for i in xrange(self.clusters):
            COV.append(np.cov(X[labels == i].T))
        COV = np.array(COV)
        PREC = np.linalg.inv(COV)
        return PREC, COV


    def _calibration(self, path, C, U, PREC):
        DOMAIN = tr.create_X(grid.get_domain(np.loadtxt(path), self.edges_of_cell, self.edges_of_cell * 3)[0], self.structure)
        F = []
        for idx in xrange(self.clusters):
            weights = self._prob_of_belong(DOMAIN, C[idx], PREC[idx])
            with np.errstate(divide='raise'):
                try:
                    density = np.sum(U[idx]) / np.sum(weights)
                except FloatingPointError:
                    print('vahy se souctem 0 nebo nevim')
                    print('np.sum(weights))')
                    print(np.sum(weights))
                    print('np.sum(U[cluster]))')
                    print(np.sum(U[idx]))
                    density = 0
            F.append(density)
        F = np.array(F)
        return F  #, heights


    def predict(self, X):
        """
        """
        DISTR = []
        for idx in xrange(self.clusters):
            DISTR.append(self.F[idx] * self._prob_of_belong(X, self.C[idx], self.PREC[idx]))
        return np.array(DISTR).max(axis=0)


    def _prob_of_belong(self, X, C, PREC):
        """
        massively inspired by:
        https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor
        """
        X_C = X - C
        c_dist_x = []
        for x_c in X_C:
            c_dist_x.append(np.dot(np.dot(x_c.T, PREC), x_c))
        c_dist_x = np.array(c_dist_x)
        return 1 - st.chi2.cdf(c_dist_x, len(C))


    def transform_data(self, path, for_fremen=False):
        #gridded, target = grid.get_domain(np.loadtxt(path), self.edges_of_cell, self.edges_of_cell)
        #X = tr.create_X(gridded, self.structure)
        ##############################################
        # not working with fremen, need to rewrite !!!!
        X = tr.create_X(np.loadtxt(path)[:, : -1], self.structure)
        target = np.loadtxt(path)[:, -1]
        if for_fremen:
            return X, target, gridded[:, 0]
        else:
            return X, target


    def rmse(self, path, plot_it=False):
        X, target = self.transform_data(path)
        #print(np.sum(target))
        y = self.predict(X)
        #print(np.sum(y))
        if plot_it:
            ll = len(target)
            plt.scatter(range(ll), target, color='b', marker="s", s=1, edgecolor="None")
            plt.scatter(range(ll), y, color='g', marker="s", s=1, edgecolor="None")
            #plt.ylim(ymax = 1.2, ymin = -0.1)
            #plt.xlim(xmax = ll, xmin = 0)
            plt.savefig('srovnani_hodnot_uhly_vse.png')
            plt.close()
        return np.sqrt(np.mean((y - target) ** 2.0))


