"""
Automatically tuned scikit-learn model.
"""

import convex_opt
import linalg
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pkg_resources
import util
from model import Model, Ensemble

DEFAULTS = pkg_resources.resource_filename(__name__, 'defaults')


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
                 stacking_alg='Logit', giant_ensemble=False, debug_mode=False, **stacking_hyperparams):

        # TODO: check if arguments to constructor are valid; set to defaults if not specified
        assert selection_method in ['qr', 'min_variance'], "The method to select entries to sample must be " \
            "either qr (QR decomposition) or min_variance (minimize variance with time constraints)."

        # attributes of ML problem
        self.p_type = p_type.lower()
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters
        self.verbose = verbose

        # computational considerations
        self.n_cores = n_cores
        self.runtime_limit = runtime_limit

        # sample column selection
        self.selection_method = selection_method
        self.scalarization = scalarization
        self.known_indices = set()

        # error matrix attributes
        # TODO: determine whether to generate new error matrix or use default/subset of default
        if type(error_matrix) == str and error_matrix == 'default':
            error_matrix = pd.read_csv(os.path.join(DEFAULTS, 'error_matrix.csv'), index_col=0)
        if type(runtime_matrix) == str and runtime_matrix == 'default':
            runtime_matrix = pd.read_csv(os.path.join(DEFAULTS, 'runtime_matrix.csv'), index_col=0)
        assert util.check_dataframes(error_matrix, runtime_matrix)
        self.column_headings = np.array([eval(heading) for heading in list(error_matrix)])
        self.error_matrix = error_matrix.values
        self.runtime_matrix = runtime_matrix.values
        self.new_row = np.zeros((1, self.error_matrix.shape[1]))

        # ensemble attributes
        self.ensemble = Ensemble(self.p_type, stacking_alg, stacking_hyperparams)
        self.giant_ensemble = giant_ensemble
        
        #debug mode: whether to keep some more intermediate results
        self.debug_mode = debug_mode

    def fit(self, x_train, y_train):
        """Fit an AutoLearner object on a new dataset. This will sample the performance of several algorithms on the
        new dataset, predict performance on the rest, then construct an optimal ensemble model.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        t_predicted = convex_opt.predict_runtime(x_train.shape)

        if self.selection_method == 'qr':
            known_indices = linalg.pivot_columns(self.error_matrix)
        elif self.selection_method == 'min_variance':
            _, Y, _ = linalg.pca(self.error_matrix, threshold=0.03)
            # TODO: Is 50/50 allocation of time to sampling & fitting appropriate?
            v_opt = convex_opt.solve(t_predicted, self.runtime_limit/2, Y, self.scalarization, self.n_cores)
            known_indices = np.where(v_opt > 0.8)[0]
        else:
            known_indices = np.arange(0, self.new_row.shape[1])

        # only need to compute column entry if it has not been computed already
        to_sample = list(set(known_indices) - self.known_indices)
        if self.verbose:
            print('Sampling {} entries of new row...'.format(len(to_sample)))
        pool1 = mp.Pool(self.n_cores)
        # TODO: Determine appropriate number of folds for k-fold fit/validate (currently 5)
        sample_models = [Model(self.p_type, self.column_headings[i]['algorithm'],
                               self.column_headings[i]['hyperparameters'], verbose=self.verbose) for i in to_sample]
        sample_model_errors = [pool1.apply_async(Model.kfold_fit_validate, args=[m, x_train, y_train, 5])
                               for m in sample_models]
        pool1.close()
        pool1.join()

        for i, error in enumerate(sample_model_errors):
            self.new_row[:, to_sample[i]] = error.get()[0].mean()
        self.new_row = linalg.impute(self.error_matrix, self.new_row, known_indices)
        # update known indices
        self.known_indices = set(known_indices)

        # add every sampled model to ensemble
        if self.giant_ensemble:
            for model in sample_models:
                self.ensemble.add_base_learner(model)

        # TODO: Add new row to error matrix at the end (might be incorrect?)
        # self.error_matrix = np.vstack((self.error_matrix, self.new_row))

        # solve knapsack problem to select models to add to ensemble
        # TODO: Determine rounding scheme to discretize knapsack problem
        weights = t_predicted.astype(int)
        values = (1e3/self.new_row)[0].astype(int)
        # TODO: Determine remaining time left to allocate to fitting ensemble
        best_indices = util.knapsack(weights, values, int(self.runtime_limit/2))
        if self.debug_mode:
            self.num_best_indices = len(best_indices)
            pivot_columns_one_percent = linalg.pivot_columns(self.error_matrix, threshold=0.01)
            self.num_pivots_one_percent = len(pivot_columns_one_percent)
            self.num_overlap_with_pivots = len(set(best_indices).intersection(set(pivot_columns_one_percent)))
            
        for i in best_indices:
            m = Model(self.p_type, self.column_headings[i]['algorithm'], self.column_headings[i]['hyperparameters'],
                      verbose=self.verbose)
            self.ensemble.add_base_learner(m)

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
        Returns:
            np.ndarray: Predicted labels.
        """
        return self.ensemble.predict(x_test)

