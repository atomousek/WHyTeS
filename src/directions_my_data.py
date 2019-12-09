"""
WHyTeS, under construction, run-able version
"""
from sklearn.mixture import GaussianMixture
import scipy.stats as st
import numpy as np
import collections
import dicttoxml
from xml.dom.minidom import parseString
from pandas import read_csv


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
                delimiter ... string, optional
                header ... None or 'infer', see pandas.read.csv, optional
                usecols ... iterable, optional
            output:
                self
        model2xml(address):
            objective:
                saves model to xml file to be readable from other sources
            input:
                address ... string, address to save the xml file
            output:
                None
        transform_data(path)
            objective:
                returns transformed data into warped hypertime space
                    optionally returns all timestamps
            input:
                path ... string, path to the test dataset
                for_fremen ... boolean, if True, method returns also array of timestamps
                delimiter ... string, optional
                header ... None or 'infer', see pandas.read.csv, optional
                usecols ... iterable, optional
            outputs:
                X ... np.array, test dataset transformed into the hypertime space
                times ... np.array, timestamps, only when for_fremen=True
        predict(X)
            objective:
                returns predicted values
            input:
                X ... np.array, test dataset projected into the hypertime space
            output:
                prediction ... probability of the occurrence of detections
    """


    def __init__(self, clusters=3, structure=[4, [86400.0]]):
        self.clusters = clusters
        self.structure = structure
        self.C = None
        self.W = None
        self.PREC = None


    def fit(self, training_path, delimiter=' ', header=None, usecols=(0, 2, 3, 5, 6)):
        """
        objective:
            to train model
        input:
            training_path ... string, path to training dataset
            delimiter ... string, optional
            header ... None or 'infer', see pandas.read.csv, optional
            usecols ... iterable, optional
        output:
            self
        """
        self.C, self.W, self.PREC = self._estimate_distribution(training_path, delimiter, header, usecols)
        return self


    def model2xml(self, address='whyte_map.xml'):
        """
        objective:
            saving model to xml file to be readable from other sources
        input:
            address ... string, address to save the xml file
        output:
            None
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
        with open(address, "w") as text_file:
            text_file.write(dom.toprettyxml())


    def transform_data(self, path, delimiter=' ', header=None, usecols=(0, 1, 2, 3, 4), for_fremen=False):
        """
        objective:
            returns transformed data into warped hypertime space and also return target values (only ones)
        input:
            path ... string, path to the test dataset
            for_fremen ... boolean, if True, method returns also array of timestamps
            delimiter ... string, optional
            header ... None or 'infer', see pandas.read.csv, optional
            usecols ... iterable, optional
        outputs:
            X ... np.array, test dataset transformed into the hypertime space
            times ... np.array, timestamps, optional, only when for_fremen=True
        """
        dataset = self._load_data(path, delimiter, header, usecols)
        X = self._create_X(dataset)
        if for_fremen:
            return X, dataset[:, 0]
        else:
            return X


    def predict(self, X):
        """
        objective:
            returns predicted values
        input:
            X ... np.array, test dataset projected into the hypertime space
        output:
            prediction ... probability of the occurrence of detections
        """
        DISTR = []
        for idx in xrange(self.clusters):
            DISTR.append(self.W[idx] * self._prob_of_belong(X, self.C[idx], self.PREC[idx]))
        DISTR = np.array(DISTR)
        prediction = np.sum(DISTR, axis=0)
        return prediction


    def _estimate_distribution(self, path, delimiter, header, usecols):
        """
        objective:
            returns parameters of the mixture of gaussian distributions of the data from one class (cluster) projected into the warped hypertime space
        inputs:
            path ... string, path to the test dataset
            delimiter ... string
            header ... None or 'infer', see pandas.read.csv
            usecols ... iterable
        outputs:
            C ... np.array, centres of clusters, estimation of expected values of each distribution
            W ... np.array, weights of clusters
            PREC ... np.array, precision matrices of clusters, inverse matrix to the estimation of the covariance of the distribution
        """
        X = self.transform_data(path, delimiter, header, usecols, for_fremen=False)
        clf = GaussianMixture(n_components=self.clusters, max_iter=500, init_params='random').fit(X)
        W = clf.weights_
        C = clf.means_
        PREC = clf.precisions_
        return C, W, PREC


    def _prob_of_belong(self, X, C, PREC):
        """
        inspired by:
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
            return data projected into the warped hypertime space
        input: 
            data numpy array nxd*, matrix of measures IRL, where d* is number of measured variables
                                   the first column need to be timestamp
        output: 
            X numpy array nxd, data projected into the warped hypertime space
        """
        dim = self.structure[0]
        wavelengths = self.structure[1]
        X = np.empty((len(data), dim + (len(wavelengths) * 2)))
        X[:, : dim] = data[:, 1: dim + 1]
        for Lambda in wavelengths:
            X[:, dim: dim + 2] = np.c_[np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                       np.sin(data[:, 0] * 2 * np.pi / Lambda)]
            dim = dim + 2
        return X
        

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


    def _load_data(self, path, delimiter, header, usecols):
        """
        objective: 
            load data from the format: 
                time [ms] (unixtime + milliseconds/1000), person id, position x [mm], position y [mm], position z (height) [mm], velocity [mm/s], angle of motion [rad], facing angle [rad]
            and return them in required format
        input:
            path ... string, address of the file
            delimiter ... string
            header ... None or 'infer', see pandas.read.csv
            usecols ... iterable
        output:
            dataset ... numpy array, (t, x, y, v_x, v_y)
        """
        #data = read_csv(filepath_or_buffer=path, sep=delimiter, header=header, usecols=usecols, engine='c', memory_map=True).values  # if you would like extra precision, use also float_precision='round_trip'
        data = read_csv(filepath_or_buffer=path, sep=delimiter, header=header, engine='c', memory_map=True).values  # if you would like extra precision, use also float_precision='round_trip'
        #v_x = np.cos(data[:, -1]) * data[:, -2]
        #v_y = np.sin(data[:, -1]) * data[:, -2]
        #return np.c_[data[:, :-2], v_x, v_y]
        return data
        
