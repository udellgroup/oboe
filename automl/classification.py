"""
Classification algorithms.
"""

import numpy as np
from model import Model

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBT
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.svm import LinearSVC as lSVM
from sklearn.svm import SVC as kSVM
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLP


class Classifier(Model):
    """An object representing a classification algorithm.

    """

    def __init__(self, algorithm, hyperparameters):
        super().__init__('classification', algorithm, hyperparameters)

    def error(self, y_predicted, y_observed):
        """Compute balanced error rate - defined as the average of the errors in each class.
        BER = 1/n * sum (0.5*(true positives/predicted positives + true negatives/predicted negatives))

        Args:
            y_predicted (np.ndarray): Predicted labels.
            y_observed (np.ndarray): Observed labels.

        Returns:
            float: Balanced error rate.
        """
        errors = []
        epsilon = 1e-15

        for i in np.unique(y_observed):
            pp = y_predicted == i
            pn = np.invert(pp)
            tp = pp & (y_observed == i)
            tn = pn & (y_observed != i)
            errors.append(0.5 * tp.sum()/np.max(pp.sum(), epsilon) + tn.sum()/np.max(pn.sum(), epsilon))

        return np.mean(errors)
