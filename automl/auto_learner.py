"""
Automatically tuned scikit-learn model.
"""

import convex_opt
import json
import linalg
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pkg_resources
import time
import util
from model import Model, Ensemble
from sklearn.model_selection import train_test_split

DEFAULTS = pkg_resources.resource_filename(__name__, 'defaults')
ERROR_MATRIX = pd.read_csv(os.path.join(DEFAULTS, 'error_matrix.csv'), index_col=0)
RUNTIME_MATRIX = pd.read_csv(os.path.join(DEFAULTS, 'runtime_matrix.csv'), index_col=0)


class AutoLearner:
    """An object representing an automatically tuned machine learning model.

    Attributes:
        p_type (str):                  Problem type. One of {'classification', 'regression'}.
        algorithms (list):             A list of algorithm types to be considered, in strings. (e.g. ['KNN', 'lSVM']).
        hyperparameters (dict):        A nested dict of hyperparameters to be considered; see above for example.
        verbose (bool):                Whether or not to generate print statements when a model finishes fitting.

        n_cores (int):                 Maximum number of cores over which to parallelize (None means no limit).
        runtime_limit(int):            Maximum training time for AutoLearner (powers of 2 preferred).

        selection_method (str):        Method of selecting entries of new row to sample.
        scalarization (str):           Scalarization of objective for min-variance selection. Either 'D' or 'A'.

        error_matrix (DataFrame):      Error matrix to use for imputation; includes index and headers.
        runtime_matrix (DataFrame):    Runtime matrix to use for runtime prediction; includes index and headers.
        column_headings (list):        Column headings of error/runtime matrices; list of dicts.
        X, Y (np.ndarray):             PCA decomposition of error matrix.

        new_row (np.ndarray):          Predicted row of error matrix.
        sampled_indices (set):         Indices of new row that have been sampled.
        sampled_models (list):         List of models that have been sampled (i.e. k-fold fitted).
        fitted_indices (set):          Indices of new row that have been fitted (i.e. included in ensemble)
        fitted_models (list):          List of models that have been fitted.

        stacking_alg (str):            Algorithm type to use for stacked learner.
    """

    def __init__(self,
                 p_type, algorithms=None, hyperparameters=None, verbose=False,
                 n_cores=mp.cpu_count(), runtime_limit=512,
                 selection_method='min_variance', scalarization='D',
                 error_matrix=None, runtime_matrix=None,
                 stacking_alg='greedy', **stacking_hyperparams):

        # TODO: check if arguments to constructor are valid; set to defaults if not specified
        assert selection_method in {'qr', 'min_variance'}, "The method to select entries to sample must be " \
            "either qr (QR decomposition) or min_variance (minimize variance with time constraints)."

        with open(os.path.join(DEFAULTS, p_type + '.json')) as file:
            defaults = json.load(file)

        # attributes of ML problem
        self.p_type = p_type.lower()
        self.algorithms = algorithms or defaults['algorithms']
        self.hyperparameters = hyperparameters or defaults['hyperparameters']
        self.verbose = verbose

        # computational considerations
        self.n_cores = n_cores
        self.runtime_limit = runtime_limit

        # sample column selection
        self.selection_method = selection_method
        self.scalarization = scalarization

        # error matrix attributes
        # TODO: determine whether to generate new error matrix or use default/subset of default
        self.error_matrix = ERROR_MATRIX if error_matrix is None else error_matrix
        self.runtime_matrix = RUNTIME_MATRIX if runtime_matrix is None else runtime_matrix
        assert util.check_dataframes(self.error_matrix, self.runtime_matrix)
        self.column_headings = np.array([eval(heading) for heading in list(self.error_matrix)])
        self.X, self.Y, _ = linalg.pca(self.error_matrix.values, rank=min(self.error_matrix.shape)-1)

        # sampled & fitted models
        self.new_row = np.zeros((1, self.error_matrix.shape[1]))
        self.sampled_indices = set()
        self.sampled_models = [None] * self.error_matrix.shape[1]
        self.fitted_indices = set()
        self.fitted_models = [None] * self.error_matrix.shape[1]

        # ensemble attributes
        self.stacking_alg = stacking_alg
        self.stacking_hyperparams = stacking_hyperparams
        self.ensemble = Ensemble(self.p_type, self.stacking_alg, self.stacking_hyperparams)

    def fit(self, x_train, y_train, rank=None, runtime_limit=None):
        """Fit an AutoLearner object on a new dataset. This will sample the performance of several algorithms on the
        new dataset, predict performance on the rest, then construct an optimal ensemble model.

        Args:
            x_train (np.ndarray):  Features of the training dataset.
            y_train (np.ndarray):  Labels of the training dataset.
            rank (int):            Rank of error matrix factorization.
            runtime_limit (float): Maximum time to allocate to AutoLearner fitting.
        """
        # set to defaults if not provided
        rank = rank or linalg.approx_rank(self.error_matrix, threshold=0.01)
        runtime_limit = runtime_limit or self.runtime_limit

        if self.verbose:
            print('Fitting AutoLearner with max. runtime {}s'.format(runtime_limit))
        t_predicted = convex_opt.predict_runtime(x_train.shape, runtime_matrix=self.runtime_matrix)

        t0 = time.time()
        while time.time() - t0 < runtime_limit/2:
            # set of algorithms that are predicted to run in given budget
            options = np.where(t_predicted <= runtime_limit/2 - (time.time() - t0))[0]
            # remove algorithms that have been sampled already
            options = list(set(options) - self.sampled_indices)
            if len(options) == 0:
                if len(self.ensemble.candidate_learners) == 0:
                    to_sample = np.argmin(t_predicted)    
                else:
                    break
            else:
                to_sample = np.random.choice(options)
            self.sampled_indices.add(to_sample)
            self.sampled_models[to_sample] = Model(self.p_type, self.column_headings[to_sample]['algorithm'],
                                                   self.column_headings[to_sample]['hyperparameters'], self.verbose,
                                                   to_sample)
            self.sampled_models[to_sample].kfold_fit_validate(x_train, y_train, 5)
            self.ensemble.candidate_learners.append(self.sampled_models[to_sample])

        if self.verbose:
            print('\nFitting ensemble of max. size {}...'.format(len(self.ensemble.candidate_learners)))
        remaining = runtime_limit - (time.time() - t0)
        self.ensemble.fit(x_train, y_train, remaining, self.fitted_models)
        for model in self.ensemble.base_learners:
            assert model.index is not None
            self.fitted_indices.add(model.index)
            self.fitted_models[model.index] = model
        self.ensemble.fitted = True

        if self.verbose:
            print('\nAutoLearner fitting complete.')

    def fit_doubling(self, x_train, y_train, verbose=False):
        """Fit an AutoLearner object, iteratively doubling allowed runtime."""
        t_predicted = convex_opt.predict_runtime(x_train.shape)

        # split data into training and validation sets
        try:
            x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train, random_state=0)
        except ValueError:
            x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.15, random_state=0)

        ranks = [linalg.approx_rank(self.error_matrix, threshold=0.05)]
        t_init = 2**np.floor(np.log2(np.sort(t_predicted)[:int(1.1*ranks[0])].sum()))
        t_init = max(1, t_init)
        times = [t_init]
        losses = [1.0]

        e_hat, actual_times, sampled, ensembles = [], [], [], []
        k, t = ranks[0], times[0]

        start = time.time()
        counter, best = 0, 0
        while time.time() - start < self.runtime_limit - t:
            if verbose:
                print('Fitting with k={}, t={}'.format(k, t))
            t0 = time.time()
            self.ensemble = Ensemble(self.p_type, self.stacking_alg, self.stacking_hyperparams)
            self.fit(x_tr, y_tr, rank=k, runtime_limit=t)
            loss = util.error(y_va, self.ensemble.predict(x_va), self.p_type)

            # TEMPORARY: Record intermediate results
            e_hat.append(np.copy(self.new_row))
            actual_times.append(time.time() - start)
            sampled.append(self.sampled_indices)
            ensembles.append(self.ensemble)
            losses.append(loss)

            if loss == min(losses):
                ranks.append(k+1)
                best = counter
            else:
                ranks.append(k)

            times.append(2*t)
            k = ranks[-1]
            t = times[-1]
            counter += 1

        # after all iterations, restore best model
        self.new_row = e_hat[best]
        self.ensemble = ensembles[best]
        return {'ranks': ranks[:-1], 'runtime_limits': times[:-1], 'validation_loss': losses,
                'predicted_new_row': e_hat, 'actual_runtimes': actual_times, 'sampled_indices': sampled,
                'models': ensembles}

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
