from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import scipy.stats as st
import numpy as np

from time import time

#import transformation_with_dirs as tr
#import full_grid as fg
#import integration
#import multiprocessing as mp

from cython_files import generate_full_grid, speed_over_angles, one_time_prediction_over_grid


from time import clock



class Directions:
    """
    parameters:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
    attributes:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
        C ... np.array, centres of clusters from the detections's model
        Pi ... np.array, weights of clusters from the detections's model
        PREC ... np.array, precision matrices of clusters from the detections's model
        C_0 ... np.array, centres of clusters from the not-detections's model
        Pi_0 ... np.array, weights of clusters from the not-detections's model
        PREC_0 ... np.array, precision matrices of clusters from the not-detections's model
    methods:
        fit(training_path)
            objective:
                to train model
            input:
                training_path ... string, path to training dataset
            output:
                self
        transform_data(path)
            objective:
                return transformed data into warped hypertime space and also return target values (0 or 1)
            input:
                path ... string, path to the test dataset
            outputs:
                X ... np.array, test dataset transformed into the hypertime space
                target ... target values
        predict(X)
            objective:
                return predicted values
            input:
                X ... np.array, test dataset transformed into the hypertime space
            output:
                prediction ... probability of the occurrence of detections
        rmse(path)
            objective:
                return rmse between prediction and target of a test dataset
            inputs:
                path ... string, path to the test dataset
            output:
                err ... float, root of mean of squared errors
    """


    def __init__(self, clusters=2, structure=[4, [86400.0, 604800.0], False]):
        self.clusters = clusters
        self.structure = structure


    def fit(self, training_path = '../data/two_weeks_days_nights_weekends_with_dirs.txt'):
        """
        objective:
            to train model
        input:
            training_path ... string, path to training dataset
        output:
            self
        """
        self.C, self.Pi, self.PREC = self._estimate_distribution(training_path)
        return self


    def transform_data(self, path, for_fremen=False):
        """
        objective:
            return transformed data into warped hypertime space and also return target values (0 or 1)
        input:
            path ... string, path to the test dataset
        outputs:
            X ... np.array, test dataset transformed into the hypertime space
            target ... target values
        """
        dataset=np.loadtxt(path)
        # old type of dataset
        #X = self._create_X(dataset[:, : -1])
        #target = dataset[:, -1]
        X = self._create_X(dataset)
        target = np.ones(len(dataset))
        if for_fremen:
            return X, target, dataset[:, 0]
        else:
            return X, target


    def _estimate_distribution(self, path):
        """
        objective:
            return parameters of the mixture of gaussian distributions of the data from one class projected into the warped hypertime space
        inputs:
            path ... string, path to the test dataset
        outputs:
            C ... np.array, centres of clusters, estimation of expected values of each distribution
            Pi ... np.array, weights of clusters
            PREC ... np.array, precision matrices of clusters, inverse matrix to the estimation of the covariance of the distribution
        """
        X = self._projection(path)
        clf = GaussianMixture(n_components=self.clusters, max_iter=500, init_params='random').fit(X)
        Pi = clf.weights_
        C = clf.means_
        PREC = clf.precisions_
        return C, Pi, PREC

    def _projection(self, path):
        """
        objective:
            return data projected into the warped hypertime
        inputs:
            path ... string, path to the test dataset
        outputs:
            X ... numpy array, data projection
        """
        dataset=np.loadtxt(path)#[:3000, :]
        # old type of dataset
        #X = self._create_X(dataset[dataset[:, -1] == 1, : -1])
        X = self._create_X(dataset)
        return X


    def predict(self, X):
        """
        objective:
            return predicted values
        input:
            X ... np.array, test dataset transformed into the hypertime space
        output:
            prediction ... probability of the occurrence of detections
        """
        DISTR = []
        for idx in xrange(self.clusters):
            DISTR.append(self.Pi[idx] * self._prob_of_belong(X, self.C[idx], self.PREC[idx]))
        DISTR = np.array(DISTR)
        model_s = np.sum(DISTR, axis=0)
        return model_s


    def _prob_of_belong(self, X, C, PREC):
        """
        massively inspired by:
        https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor

        objective:
            return 1 - "p-value" of the hypothesis, that values were "generated" by the (normal) distribution
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
        imputs:
            X ... np.array, test dataset transformed into the hypertime space
            C ... np.array, centres of clusters, estimation of expected values of each distribution
            PREC ... numpy array, precision matrices of corresponding clusters
        output
            numpy array, estimation of probabilities for each tested vector
        """
        X_C = X - C
        c_dist_x = np.sum(np.dot(X_C, PREC) * X_C, axis=1)
        return 1 - st.chi2.cdf(c_dist_x, len(C))


    def rmse(self, path):
        """
        objective:
            return rmse between prediction and target of a test dataset
        inputs:
            path ... string, path to the test dataset
        output:
            err ... float, root of mean of squared errors
        """
        X, target = self.transform_data(path)
        y = self.predict(X)
        return np.sqrt(np.mean((y - target) ** 2.0))


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
        ## normalization
        #minimum = np.min(data[:, 1: dim + 1], axis=0)
        #maximum = np.max(data[:, 1: dim + 1], axis=0)
        #X[:, : dim] = data[:, 1: dim + 1] * 2.0 / (maximum-minimum)
        for Lambda in wavelengths:
            X[:, dim: dim + 2] = np.c_[np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                       np.sin(data[:, 0] * 2 * np.pi / Lambda)]
            dim = dim + 2

        if self.structure[2]:
            X[:, dim: dim + 2] = np.c_[data[:, -1] * np.cos(data[:, -2]),
                                       data[:, -1] * np.sin(data[:, -2])]
        return X


    def model_to_directions(self, time=0):#, angle_edges, central_points, finishes, steps):
        """
        this model is only for specialized task, where the model was created without the time domain, 
        but it is definitely also the testing version for the real function, which will be used for the evaluation method
        """
        # default parameters, spatial range litle bit higher, speeds limited to 4m/s (probably 3m/s is very close to max)
        edges = np.array([0.25, 0.25, 0.2, 0.2])
        #edges = np.array([1.0, 1.0, 1.0, 1.0])
        starting_bins_centres = np.array([-10.5, -0.75, -4.0, -4.0])
        finishing_bins_centres = np.array([3.0, 17.25, 4.0, 4.0])
        # derived parameters
        no_bins = (((finishing_bins_centres - starting_bins_centres)/ edges) + 1.0).astype(int)
        dim = 4

        # grid in x, y, v_x, v_y; for model creation
        grid = generate_full_grid.generate(edges, no_bins, starting_bins_centres, dim)#, grid)

        # prediction over that grid

        # default parameters, spatial range litle bit higher, speeds limited to 4m/s (probably 3m/s is very close to max)
        edges = np.array([1.0, 0.25, 0.25, 0.2, 0.2])
        #edges = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        starting_bins_centres = np.array([time, -10.5, -0.75, -4.0, -4.0])
        finishing_bins_centres = np.array([time, 3.0, 17.25, 4.0, 4.0])
        # derived parameters
        no_bins = (((finishing_bins_centres - starting_bins_centres)/ edges) + 1.0).astype(int)
        dim = 5
        periodicities = np.array(self.structure[1])
        PI2 = np.pi*2
        #start = clock()
        #y = prediction_over_grid.predict(edges, no_bins, starting_bins_centres, periodicities, dim, self.C, self.PREC, PI2, self.clusters, self.Pi)
        #finish = clock()
        #print('prediction_over_grid: ' + str(finish-start))
        ## testing version in python - memory inefficient
        #start = clock()
        #random = np.array([[time, 0.0, 0.0, 0.0, 0.0]])
        #transformed = self._create_X(random)[:, 4:].reshape(-1)
        #extended = np.broadcast_to(transformed, (len(grid), len(transformed)))
        #python_y = self.predict(np.c_[grid, extended])
        #finish = clock()
        #print('python prediction: ' + str(finish-start))
        ## testing the same thiing with cython
        start = clock()
        y = one_time_prediction_over_grid.predict(edges, no_bins, starting_bins_centres, periodicities, dim, self.C, self.PREC, PI2, self.clusters, self.Pi)
        finish = clock()
        #print('one_time_prediction_over_grid: ' + str(finish-start))
        #print('je to rozdilne? : ' + str(np.sum(y != new_y)))
        
        # projection of grid into x, y, velocity and speed
        # velocity vectors over the grid
        velocity = grid[:, -2:]
        # speed calculated from the velocity vector
        speed = np.sqrt(np.sum(velocity ** 2, axis=1))
        # signum, where 0 -> 1, positive for v_y non-negative
        my_signum = (velocity[:,1]>0.0)*2.0 - 1.0
        # angles to the velocity (1, 0)
        angle = my_signum*np.arccos(velocity[:,0]/speed)
        # now, I have to move angles between -pi and -7pi/8 to "positive" values
        # as I need the cell around pi (7pi/8 to 9pi/8)
        angle[angle<(-7.0*np.pi/8.0)] = 2*np.pi + angle[angle<(-7.0*np.pi/8.0)]
        # predicted values limited only for the speed lower than 4 m/s
        y[speed>4.0] = -1
        # x, y (first two columns from grid), angle, prediction (y)
        #angle_model = np.c_[grid[:, : 2], angle, y]
        angle_model = np.c_[grid[:, : 2], angle, y, speed]
        # cell edges for x, y, angle
        angle_edges = np.array([0.5, 0.5, np.pi/4.0])

        # input values for angles integration - gathered from the question by the evaluation system
        # now added by hand
        no_bins = np.array([24, 33, 8])
        # 1552352967.4318142 is little bit lower then the lowest time - so everything is rounded to that value
        # as the time would be included in the prediction, but it is constant for this calculation, it should not be included
        central_points = np.array([-9.5, 0.25, -3.0*np.pi/4.0])
        lower_edge_points = central_points - (angle_edges * 0.5)      
        dim = len(angle_edges)

        # create grid in the new projection
        grid_over_angles = generate_full_grid.generate(angle_edges, no_bins, central_points, dim)#, grid_over_angles)
        # summing up all the predicted values in the cells of new projection
        sums_over_angles, speed_weighted_mean, positions_sum = speed_over_angles.target(angle_model, angle_edges, no_bins, lower_edge_points)
        # not necessary to return count, count is only for testing the symetry
        return np.c_[grid_over_angles, sums_over_angles, speed_weighted_mean, positions_sum]
        #sums_over_angles, speed_weighted_mean = target_over_grid.target(angle_model, angle_edges, no_bins, lower_edge_points)
        ## not necessary to return count, count is only for testing the symetry
        #return np.c_[grid_over_angles, sums_over_angles, speed_weighted_mean]
