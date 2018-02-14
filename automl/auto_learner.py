"""
Automatically tuned scikit-learn model.
"""

import numpy as np
import multiprocessing as mp
from model import Model, Ensemble
from pathos.multiprocessing import ProcessingPool as Pool
import linalg
import pandas as pd
import subprocess
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
# TODO: include more regression algorithms


CLS = ['KNN', 'DT', 'RF', 'GBT', 'AB', 'lSVM', 'kSVM', 'Logit', 'Perceptron', 'GNB']
REG = ['Lasso', 'Ridge', 'ElasticNet']

CLASSIFICATION = dict(zip(CLS, list(map(lambda name: eval(name), CLS))))
REGRESSION = dict(zip(REG, list(map(lambda name: eval(name), REG))))

HYPERPARAMETERS_C = {'KNN':        {'n_neighbors':       np.arange(1, 15, 2, dtype=int)},
                     'DT':         {'min_samples_split': np.geomspace(0.01, 0.00001, 4)},
                     'RF':         {'min_samples_split': np.geomspace(0.01, 0.00001, 4)},
                     'GBT':        {'learning_rate':     np.geomspace(0.01, 0.001, 3)},
                     'AB':         {'n_estimators':      np.array([50, 100]),
                                    'learning_rate':     np.array([1.0, 2.0])},
                     'lSVM':       {'C':                 np.array([0.25, 0.5, 0.75, 1.0, 2.0])},
                     'kSVM':       {'C':                 np.array([0.25, 0.5, 0.75, 1.0, 2.0])},
                     'Logit':      {'C':                 np.array([0.25, 0.5, 0.75, 1.0, 2.0])},
                     'Perceptron': {},
                     'GNB':        {}
                     }
HYPERPARAMETERS_R = {}

DEFAULTS = {'algorithms':      {'classification': CLASSIFICATION,    'regression': REGRESSION},
            'hyperparameters': {'classification': HYPERPARAMETERS_C, 'regression': HYPERPARAMETERS_R}}


class AutoLearner:
    """An object representing an automatically tuned machine learning model.

    Attributes:
        type (str): Problem type. One of {'classification', 'regression'}.
        algorithms (list): A list of algorithm types to be considered, in strings. (e.g. ['KNN', 'lSVM', 'kSVM']).
        hyperparameters (dict): A nested dict of hyperparameters to be considered; see above for example.
        n_cores (int): Maximum number of cores over which to parallelize (None means no limit).
        verbose (bool): Whether or not to generate print statements when a model finishes fitting.
        stacking_alg (str): Algorithm type to use for stacked learner.
        **stacking_hyperparams (dict): Hyperparameter settings of stacked learner.
    """
    def __init__(self, p_type, algorithms=None, hyperparameters=None, n_cores=None, verbose=False,
                 stacking_alg='Logit', **stacking_hyperparams):

        # check if arguments to constructor are valid; set to defaults if not specified
        old, new = util.check_arguments(p_type, algorithms, hyperparameters, DEFAULTS)
        self.p_type = p_type
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters
        self.n_cores = n_cores
        self.verbose = verbose

        if len(new) > 0:
            # if selected hyperparameters contain model configurations not included in default
            proceed = input("Your selected hyperparameters contain some not included in the default error matrix. \n"
                            "Do you want to generate your own error matrix? [yes/no]")
            if proceed == 'yes':
                subprocess.call(['./generate_matrix.sh'])
                # TODO: load newly generated error matrix file
            else:
                return
        else:
            # use default error matrix (or subset of)
            default_error_matrix = pd.read_csv('./defaults/error_matrix.csv', index_col=0)
            column_headings = np.array([eval(heading) for heading in list(default_error_matrix)])
            self.error_matrix = default_error_matrix.values[:, np.in1d(old, column_headings)]
            self.column_headings = sorted(old, key=lambda d: list(d.keys())[0])

        self.model = Ensemble(self.p_type, stacking_alg, **stacking_hyperparams)
        self.new_row = None

    def fit(self, x_train, y_train):
        """Fit an AutoLearner object on a new dataset. This will sample the performance of several algorithms on the
        new dataset, predict performance on the rest, then perform Bayesian optimization and construct an optimal
        ensemble model.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        self.new_row = np.zeros((1, x_train.shape[1]))
        known_indices = linalg.pivot_columns(self.error_matrix)

        print('Sampling {} entries of new row...'.format(len(known_indices)))
        pool1 = mp.Pool(self.n_cores)
        sample_models = [Model(self.p_type, list(self.column_headings[i].keys())[0],
                         list(self.column_headings[i].values())[0]) for i in known_indices]
        sample_model_errors = [pool1.apply_async(Model.kfold_fit_validate, args=[m, x_train, y_train, 5])
                               for m in sample_models]
        pool1.close()
        pool1.join()
        for i, error in sample_model_errors:
            self.new_row[:, known_indices[i]] = error.get()[0]
            # TODO: add predictions to second layer matrix?
        self.new_row = linalg.impute(self.error_matrix, self.new_row, known_indices)

        # Add new row to error matrix at the end (might be incorrect?)
        # self.error_matrix = np.vstack((self.error_matrix, self.new_row))

        # TODO: Fit ensemble candidates (?)

        if self.verbose:
            print('Conducting Bayesian optimization...')
        n_models = 3
        pool2 = Pool(self.n_cores)
        bayesian_opt_models = [Model(self.p_type, list(self.column_headings[i].keys())[0],
                               list(self.column_headings[i].values())[0]) for i in np.argsort(self.new_row)[:n_models]]
        optimized_models = pool2.map(Model.bayesian_optimize, bayesian_opt_models)
        pool2.close()
        pool2.join()
        for m in optimized_models:
            self.model.add_base_learner(m)

        if self.verbose:
            print('Fitting optimized ensemble...')
        self.model.fit(x_train, y_train)
        self.model.fitted = True

    def refit(self, x_train, y_train):
        """Refit an existing AutoLearner object on a new dataset. This will simply retrain the base-learners and
        stacked learner of an existing model, and so algorithm and hyperparameter selection may not be optimal.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        assert self.model.fitted, "Cannot refit unless model has been fit."
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """Generate predictions on test data.

        Args:
            x_test (np.ndarray): Features of the test dataset.
        """
        return self.model.predict(x_test)
