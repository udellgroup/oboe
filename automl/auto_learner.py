"""
Automatically tuned scikit-learn model.
"""

import numpy as np
import multiprocessing as mp
import pandas as pd
import pkg_resources
import linalg
import util
import convex_opt_c
import convex_opt_s
from model import Model, Ensemble
from pathos.multiprocessing import ProcessingPool as Pool


class AutoLearner:
    """An object representing an automatically tuned machine learning model.

    Attributes:
        p_type (str):                  Problem type. One of {'classification', 'regression'}.
        algorithms (list):             A list of algorithm types to be considered, in strings.
                                       (e.g. ['KNN', 'lSVM', 'kSVM']).
        hyperparameters (dict):        A nested dict of hyperparameters to be considered; see above for example.
        n_cores (int):                 Maximum number of cores over which to parallelize (None means no limit).
        verbose (bool):                Whether or not to generate print statements when a model finishes fitting.
        stacking_alg (str):            Algorithm type to use for stacked learner.
        **stacking_hyperparams (dict): Hyperparameter settings of stacked learner.
    """

    def __init__(self, p_type, error_matrix='default', runtime_matrix='default', algorithms=None, hyperparameters=None,
                 n_cores=None, verbose=False, selection_method='qr', runtime_limit=None, scalarization='D',
                 bayes_opt=False, stacking_alg='Logit', giant_ensemble=False, **stacking_hyperparams):

        assert selection_method in ['qr', 'min_variance'], "The method to select entries to sample must be " \
            "either qr (QR decomposition) or min_variance (minimize variance with time constraints)."

        # check if arguments to constructor are valid; set to defaults if not specified
        # default, new = util.check_arguments(p_type, algorithms, hyperparameters)
        self.p_type = p_type.lower()
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters
        self.n_cores = n_cores
        self.verbose = verbose
        self.selection_method = selection_method
        self.runtime_limit = runtime_limit

        self.n_cores = n_cores
        self.scalarization = scalarization
        self.giant_ensemble = giant_ensemble

        # TODO: determine whether to generate new error matrix or use default
        # use default error matrix (or subset of)
        if error_matrix == 'default':
            error_matrix_path = pkg_resources.resource_filename(__name__, 'defaults/error_matrix.csv')
            default_error_matrix = pd.read_csv(error_matrix_path, index_col=0)
        elif type(error_matrix) == pd.core.frame.DataFrame:
            default_error_matrix = error_matrix

        if runtime_matrix == 'default':
            runtime_matrix_path = pkg_resources.resource_filename(__name__, 'defaults/runtime_matrix.csv')
            default_runtime_matrix = pd.read_csv(runtime_matrix_path, index_col=0)
        elif type(runtime_matrix) == pd.core.frame.DataFrame:
            default_runtime_matrix = runtime_matrix

        assert set(default_error_matrix.index) == set(default_runtime_matrix.index), \
            "Indices of error and runtime matrices must match."
        column_headings = np.array([eval(heading) for heading in list(default_error_matrix)])
        selected_indices = np.full(len(column_headings), True)
        # selected_indices = np.array([heading in column_headings for heading in default])
        self.error_matrix = default_error_matrix.values[:, selected_indices]
        self.error_index = default_error_matrix.index.tolist()
        self.runtime_index = default_runtime_matrix.index.tolist()
        self.runtime_matrix = default_runtime_matrix.values[:, selected_indices]

        # self.column_headings = sorted(default, key=lambda d: d['algorithm'])
        self.column_headings = column_headings
        self.bayes_opt = bayes_opt

        self.ensemble = Ensemble(self.p_type, stacking_alg, stacking_hyperparams)
        self.optimized_settings = []
        self.new_row = None

    def fit(self, x_train, y_train):
        """Fit an AutoLearner object on a new dataset. This will sample the performance of several algorithms on the
        new dataset, predict performance on the rest, then perform Bayesian optimization and construct an optimal
        ensemble model.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        self.new_row = np.zeros((1, self.error_matrix.shape[1]))
        if self.selection_method == 'qr':
            known_indices = linalg.pivot_columns(self.error_matrix)
        elif self.selection_method == 'min_variance':
            runtime_predict = convex_opt_s.predict_runtime(x_train.shape)
            X, Y, Vt = linalg.pca(self.error_matrix, threshold=0.03)
            v_opt_x = convex_opt_s.solve(runtime_predict, self.runtime_limit, Y, scalarization=self.scalarization,
                                         n_cores=self.n_cores)
            known_indices = np.where(v_opt_x > 0.8)[0]
        
        if self.verbose:
            print('Sampling {} entries of new row...'.format(len(known_indices)))
        pool1 = mp.Pool(self.n_cores)
        sample_models = [Model(self.p_type, self.column_headings[i]['algorithm'],
                               self.column_headings[i]['hyperparameters'], verbose=self.verbose)
                         for i in known_indices]
        sample_model_errors = [pool1.apply_async(Model.kfold_fit_validate, args=[m, x_train, y_train, 5])
                               for m in sample_models]
        pool1.close()
        pool1.join()

        for i, error in enumerate(sample_model_errors):
            self.new_row[:, known_indices[i]] = error.get()[0].mean()
        self.new_row = linalg.impute(self.error_matrix, self.new_row, known_indices)

        # add every sampled model to ensemble
        if self.giant_ensemble:
            for model in sample_models:
                self.ensemble.add_base_learner(model)

        # Add new row to error matrix at the end (might be incorrect?)
        # self.error_matrix = np.vstack((self.error_matrix, self.new_row))

        if self.verbose:
            print('\nFitting optimized ensemble...')
        self.ensemble.fit(x_train, y_train)
        self.ensemble.fitted = True

        if self.verbose:
            print('\nAutoLearner fitting complete.')

    def refit(self, x_train, y_train):
        """Refit an existing AutoLearner object on a new dataset. This will simply retrain the base-learners and
        stacked learner of an existing model, and so algorithm and hyperparameter selection may not be optimal.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        assert self.ensemble.fitted, "Cannot refit unless model has been fit."
        self.ensemble.fit(x_train, y_train)

    def predict(self, x_test):
        """Generate predictions on test data.

        Args:
            x_test (np.ndarray): Features of the test dataset.
        """
        return self.ensemble.predict(x_test)

