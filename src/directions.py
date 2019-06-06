from sklearn.mixture import GaussianMixture
import transformation_with_dirs as tr
#import grid_with_directions as grid

#import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
#import matplotlib.patches as pat
#import pandas as pd



class Directions:
    """
    dirs = Directions(clusters, structure, train_path)
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
        train_path ... string, path to training dataset
    attributes:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
        C_1 ... np.array, centres of clusters from the detections's model
        Pi_1 ... np.array, weights of clusters from the detections's model
        PREC_1 ... np.array, precision matices of clusters from the detections's model
        C_0 ... np.array, centres of clusters from the not-detections's model
        Pi_0 ... np.array, weights of clusters from the not-detections's model
        PREC_0 ... np.array, precision matices of clusters from the not-detections's model
    methods:
        X, target = dirs.transform_data(path)
            path ... string, path to the test dataset
            X ... np.array, test dataset transformed into the hypertime space
            target ... target values
        y = dirs.predict(X)
            X ... np.array, test dataset transformed into the hypertime space
            y ... probability of the occurrence of detections
        err = dirs.rmse(path, plot_it=False)
            path ... string, path to the test dataset
            plot_it ... boolean, create and save graph of y and target
            err ... float, root of mean of squared errors
#        dirs.vykresli(time, window)
#            time ... number, value of timestep in wich we want to show the model
#            window ... integer, time step between two graphs
#
#        
#            
    """


    def __init__(self, clusters=3,
                 #edges_of_cell = np.array([1200.0, 0.5, 0.5]),  # ! pridano pro testovani
                 structure=[4, [1.0, 1.0], [86400.0, 604800.0]],
                 train_path = '../data/two_weeks_days_nights_weekends_with_dirs.txt'):
        self.clusters = clusters
        #self.edges_of_cell = edges_of_cell  # ! pridano pro testovani
        self.structure = structure
        self.C_1, self.Pi_1, self.PREC_1 = self._create_model(1, train_path)
        self.C_0, self.Pi_0, self.PREC_0 = self._create_model(0, train_path)
        #self.C_1, self.Pi_1, self.PREC_1 = self._create_model_freq(1, train_path)
        #self.C_0, self.Pi_0, self.PREC_0 = self._create_model_freq(0, train_path)


    def transform_data(self, path, for_fremen=False):
        dataset=np.loadtxt(path)
        X = tr.create_X(dataset[:, : -1], self.structure)
        target = dataset[:, -1]
        if for_fremen:
            return X, target, dataset[:, 0]
        else:
            return X, target
        #return X, target


    def _create_model(self, condition, path):
        X = self._projection(path, condition)
        clf = GaussianMixture(n_components=self.clusters, max_iter=500).fit(X)
        C = clf.means_
        labels = clf.predict(X)
        PREC = self._recalculate_precisions(X, labels)
        Pi = clf.weights_
        return C, Pi, PREC

    def _projection(self, path, condition):
        dataset=np.loadtxt(path)
        X = tr.create_X(dataset[dataset[:, -1] == condition, : -1], self.structure)
        return X


    def _recalculate_precisions(self, X, labels):
        COV = []
        for i in xrange(self.clusters):
            COV.append(np.cov(X[labels == i].T))
        COV = np.array(COV)
        PREC = np.linalg.inv(COV)
        return PREC


    def predict(self, X):
        DISTR_1 = []
        DISTR_0 = []
        for idx in xrange(self.clusters):
            DISTR_1.append(self.Pi_1[idx] * self._prob_of_belong(X, self.C_1[idx], self.PREC_1[idx]))
            DISTR_0.append(self.Pi_0[idx] * self._prob_of_belong(X, self.C_0[idx], self.PREC_0[idx]))
        DISTR_1 = np.array(DISTR_1)
        DISTR_0 = np.array(DISTR_0)
        model_1_s = np.sum(DISTR_1, axis=0)
        model_0_s = np.sum(DISTR_0, axis=0)
        model_01_s = model_1_s + model_0_s
        model_01_s[model_01_s == 0] = 1.0
        model_1_s[model_01_s == 0] = 0.5
        y = model_1_s / model_01_s
        return y


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


    def rmse(self, path, plot_it=False):
        X, target = self.transform_data(path)
        y = self.predict(X)
        if plot_it:
            ll = len(target)
            plt.scatter(range(ll), target, color='b', marker="s", s=1, edgecolor="None")
            plt.scatter(range(ll), y, color='g', marker="s", s=1, edgecolor="None")
            plt.ylim(ymax = 1.2, ymin = -0.1)
            plt.xlim(xmax = ll, xmin = 0)
            plt.savefig('srovnani_hodnot_uhly_vse.png')
            plt.close()
        return np.sqrt(np.mean((y - target) ** 2.0))

"""
    def vykresli(self, time, window):
        XYUVC = self._create_grid_model(time)
        plt.quiver(XYUVC[:, 0], XYUVC[:, 1], XYUVC[:, 2], XYUVC[:, 3], XYUVC[:, 4], angles='xy', scale_units='xy', scale=1.0)
        plt.colorbar()
        plt.savefig('sipky/sipky' + str(time/window) + '_test.png')
        plt.close()


    def _create_coords(self, time):
        X = np.arange(-8.0, 24.0, 0.5)
        Y = np.arange(-6.0, 17.0, 0.5)
        angles = np.arange(-np.pi, np.pi, np.pi/36.0)
        coords = self._cartesian_product(np.array([time]), X, Y, angles)
        return pd.DataFrame(coords)


    def _create_grid_model(self, time):
        table = self._create_coords(time)
        table[4] = np.cos(table[3])
        table[3] = np.sin(table[3])
        model = self.predict(tr.create_X(table.values, self.structure))
        table[5] = model
        a = table.groupby([1, 2]).apply(self._UVC_max).rename(columns={0:3, 1:4, 2:5})
        a.reset_index(inplace=True)
        return a.values
"""


#    def _cartesian_product(self, *arrays):
#        """
#        coppied from:
#        'https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of'+\
#        '-x-and-y-array-points-into-single-array-of-2d-points'
#        input: *arrays enumeration of central_points
#        output: numpy array (central positions of cels of grid)
#        uses: np.empty(),np.ix_(), np.reshape()
#        objective: to perform cartesian product of values in columns
#        """
#        la = len(arrays)
#        arr = np.empty([len(a) for a in arrays] + [la],
#                       dtype=arrays[0].dtype)
#        for i, a in enumerate(np.ix_(*arrays)):
#            arr[..., i] = a
#        return arr.reshape(-1, la)


"""
    def _entropy(self, series):
        try:
            normed = series/np.sum(series)
        except:
            normed = np.ones_like(series) / len(series)
        return -np.sum(normed*np.log2(normed))


    def _nonzero_median(self, series):
        bol_zer = (series!=0.0)
        if np.sum(bol_zer * 1.0) == 0.0:
            return 0.0
        else:
            return np.median(series[bol_zer])


    def _UVC_max(self, group):
        xyw = group[[3, 4, 5]].values
        xyw = xyw[xyw[:, 2] == np.max(xyw[:, 2]), :]
        idx = np.random.choice(range(np.shape(xyw)[0]))
        xyw = pd.Series(xyw[idx, :])
        xyw[0] = xyw[0] * xyw[2]
        xyw[1] = xyw[1] * xyw[2]
        xyw[2] = self._entropy(group[5])
        return xyw
"""


    #def _create_model_freq(self, condition, path):
    #    C, U, PREC = self._get_params(path, condition)
    #    F = self._calibration(path, C, U, PREC)
    #    return C, F, PREC


    #def _get_params(self, path, condition):
    #    X = self._projection(path, condition)
    #    #X = tr.create_X(self._get_data(path), self.structure)
    #    clf = GaussianMixture(n_components=self.clusters, max_iter=500).fit(X)
    #    labels = clf.predict(X)
    #    PREC = self._recalculate_precisions(X, labels)
    #    U = clf.predict_proba(X)
    #    C = clf.means_
    #    return C, U.T, PREC


    #def _calibration(self, path, C, U, PREC):
    #    DOMAIN = tr.create_X(grid.get_domain(np.loadtxt(path), self.edges_of_cell, self.edges_of_cell * 3)[0], self.structure)
    #    F = []
    #    for idx in xrange(self.clusters):
    #        weights = self._prob_of_belong(DOMAIN, C[idx], PREC[idx])
    #        with np.errstate(divide='raise'):
    #            try:
    #                density = np.sum(U[idx]) / np.sum(weights)
    #            except FloatingPointError:
    #                print('vahy se souctem 0 nebo nevim')
    #                print('np.sum(weights))')
    #                print(np.sum(weights))
    #                print('np.sum(U[cluster]))')
    #                print(np.sum(U[idx]))
    #                density = 0
    #        F.append(density)
    #    F = np.array(F)
    #    return F  #, heights

