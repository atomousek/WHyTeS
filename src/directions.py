"""
WHyTeS, under construction, run-able version
"""
from sklearn.mixture import GaussianMixture
import scipy.stats as st
import numpy as np
import collections
import dicttoxml
from xml.dom.minidom import parseString


class Directions:
    """
    parameters:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
    attributes:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
        C ... np.array, centres of clusters from the detections's model
        W ... np.array, weights of clusters from the detections's model
        PREC ... np.array, precision matrices of clusters from the detections's model
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
        self.C, self.W, self.PREC = self._estimate_distribution(training_path)
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
        #dataset=np.loadtxt(path)
        dataset = self.load_data(path)
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
            W ... np.array, weights of clusters
            PREC ... np.array, precision matrices of clusters, inverse matrix to the estimation of the covariance of the distribution
        """
        X = self._projection(path)
        clf = GaussianMixture(n_components=self.clusters, max_iter=500, init_params='random').fit(X)
        W = clf.weights_
        C = clf.means_
        PREC = clf.precisions_
        return C, W, PREC

    def _projection(self, path):
        """
        objective:
            return data projected into the warped hypertime
        inputs:
            path ... string, path to the test dataset
        outputs:
            X ... numpy array, data projection
        """
        #dataset=np.loadtxt(path)
        dataset = self.load_data(path)
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
            DISTR.append(self.W[idx] * self._prob_of_belong(X, self.C[idx], self.PREC[idx]))
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
        return st.chi2._sf(c_dist_x, len(C))


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


    def model2xml(self):
        """
        objective: saving model to xml file to be readable from other sources
        """
        variables = [self.C, self.W, self.PREC, np.array(self.structure[1])]
        names = ['C', 'W', 'PREC', 'periodicities']
        some_dict = collections.OrderedDict({'no_periods': len(self.structure[1]), 'no_clusters': self.clusters})
        for idx, variable in enumerate(variables):
            shape, values = self._numpy2dict(variable)
            some_dict[names[idx]+'_shape'] = shape
            some_dict[names[idx]+'_values'] = values
        xml = dicttoxml.dicttoxml(some_dict)
        dom = parseString(xml)
        with open("whyte_map.xml", "w") as text_file:
            text_file.write(dom.toprettyxml())
        

    def _numpy2dict(self, np_arr):
        """
        objective: 
            unfold numpy array to ordered dict, where keys are indexes and values are values in numpy array
        input: 
            numpy array, any
        output: 
            shape ...  ordered dict, original shape of numpy array
            values ... ordered dict, all values of reshaped(-1) numpy array
        """
        shape = collections.OrderedDict({})
        for idx, val in enumerate(np.shape(np_arr)):
            shape['s_'+str(idx)] = val
        np_arr = np.reshape(np_arr, -1)
        values = collections.OrderedDict({})
        for idx, val in enumerate(np_arr):
            values['v_'+str(idx)] = val
        return shape, values


    def load_data(self, path):
        """
        objective: 
            load data from the format: 
                time [ms] (unixtime + milliseconds/1000), person id, position x [mm], position y [mm], position z (height) [mm], velocity [mm/s], angle of motion [rad], facing angle [rad]
            and return them in required format
        input:
            path ... string, address of the file
        output:
            dataset ... numpy array, (t, x, y, v_x, v_y)
        """
        data = np.loadtxt(path)[:, [0, 2, 3, 5, 6]]
        v_x = np.cos(data[:, -1]) * data[:, -2]
        v_y = np.sin(data[:, -1]) * data[:, -2]
        return np.c_[data[:, :-2], v_x, v_y]
        
