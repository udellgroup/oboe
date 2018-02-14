"""
Automatically tuned scikit-learn model.
"""

import numpy as np
import multiprocessing as mp
from model import Model, Ensemble
import linalg
import util

# Classification algorithms
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

# Regression algorithms
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


CLS = ['KNN', 'DT', 'RF', 'GBT', 'AB', 'lSVM', 'kSVM', 'Logit', 'Perceptron', 'GNB']
REG = ['Lasso', 'Ridge', 'ElasticNet']

CLASSIFICATION = dict(zip(CLS, list(map(lambda name: eval(name), CLS))))
REGRESSION = dict(zip(REG, list(map(lambda name: eval(name), REG))))

HYPERPARAMETERS_C = {'KNN':        {'n_neighbors':       np.arange(1, 15, 2, dtype=int)},
                     'DT':         {'min_samples_split': np.geomspace(0.01, 0.00001, 4)},
                     'RF':         {'min_samples_split': np.geomspace(0.01, 0.00001, 4)},
                     'GBT':        {'learning_rate':     np.geomspace(0.01, 0.001, 3)},
                     'AB':         {'n_estimators':      np.arange(50, 150, 50),
                                    'learning_rate':     np.arange(1, 3, 1)},
                     'lSVM':       {'C':                 [0.25, 0.5, 0.75, 1.0, 2.0]},
                     'kSVM':       {'C':                 [0.25, 0.5, 0.75, 1.0, 2.0]},
                     'Logit':      {'C':                 [0.25, 0.5, 0.75, 1.0, 2.0]},
                     'Perceptron': {},
                     'GNB':        {}
                     }
HYPERPARAMETERS_R = {}

DEFAULTS = {'algorithms':      {'classification': CLASSIFICATION,    'regression': REGRESSION},
            'hyperparameters': {'classification': HYPERPARAMETERS_C, 'regression': HYPERPARAMETERS_R}}


class AutoLearner:
    """An object representing an automatically tuned machine learning model.

    Attributes:
        type (str): Problem type. One of 'classification', 'regression'
        algorithms (list): A list of algorithm types to be considered, in strings. (e.g. ['KNN', 'lSVM', 'kSVM'])
        hyperparameters (dict): A nested dict of hyperparameters to be considered; see above for example.
    """

    def __init__(self, p_type, algorithms=None, hyperparameters=None):

        # check if arguments to constructor are valid; set to defaults if not specified
        default = util.check_arguments(p_type, algorithms, hyperparameters, DEFAULTS)
        # TODO: ask user if they want to generate error matrix.

        self.type = p_type
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters

    def fit(self, x_train, y_train):
        pass

    def refit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass
