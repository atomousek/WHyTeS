from sklearn.mixture import GaussianMixture
import scipy.stats as st
import numpy as np

#import transformation_with_dirs as tr


class Directions:
    """
    parameters:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
    attributes:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
        C_1 ... np.array, centres of clusters from the detections's model
        Pi_1 ... np.array, weights of clusters from the detections's model
        PREC_1 ... np.array, precision matrices of clusters from the detections's model
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


    def __init__(self, clusters=3, structure=[2, [1.0, 1.0], [86400.0, 604800.0]], weights=None):
        self.clusters = clusters
        self.structure = structure
        self.weights = weights


    def fit(self, training_path = '../data/two_weeks_days_nights_weekends_with_dirs.txt'):
        """
        objective:
            to train model
        input:
            training_path ... string, path to training dataset
        output:
            self
        """
        self.C_1, self.Pi_1, self.PREC_1 = self._estimate_distribution(1, training_path)
        #self.C_0, self.Pi_0, self.PREC_0 = self._estimate_distribution(0, training_path)
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
        X = self._create_X(dataset[:, : -1])
     
