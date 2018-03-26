"""
Automatically tuned scikit-learn model.
"""

import copy
import convex_opt
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


class AutoLearner:
    """An object representing an automatically tuned machine learning model.

    Attributes:
        p_type (str):                  Problem type. One of {'classification', 'regression'}.
        algorithms (list):             A list of algorithm types to be considered, in strings.
                                       (e.g. ['KNN', 'lSVM', 'kSVM']).
        hyperparameters (dict):        A nested dict of hyperparameters to be considered; see above for example.
        verbose (bool):                Whether or not to generate print statements when a model finishes fitting.
        n_cores (int):                 Maximum number of cores over which to parallelize (None means no limit).
        runtime_limit(int):            Maximum training time for AutoLearner (powers of 2 preferred).
        selection_method (str):        Method of selecting entries of new row to sample.
        scalarization (str):           Scalarization of objective for min-variance selection. Either 'D' or 'A'.
        stacking_alg (str):            Algorithm type to use for stacked learner.
        **stacking_hyperparams (dict): Hyperparameter settings of stacked learner.
    """
    def __init__(self,
                 p_type, algorithms=None, hyperparameters=None, verbose=False,
                 n_cores=mp.cpu_count(), runtime_limit=512,
                 selection_method='min_variance', scalarization='D',
                 error_matrix='default', runtime_matrix='default',
                 stacking_alg='Logit', giant_ensemble=False, **stacking_hyperparams):

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
        self.v_opt = None
        self.scalarization = scalarization
        self.known_indices = set()
        self.best_indices = set()

        # error matrix attributes
        # TODO: determine whether to generate new error matrix or use default/subset of default
        if type(error_matrix) == str and error_matrix == 'default':
            error_matrix = pd.read_csv(os.path.join(DEFAULTS, 'error_matrix.csv'), index_col=0)
        if type(runtime_matrix) == str and runtime_matrix == 'default':
            runtime_matrix = pd.read_csv(os.path.join(DEFAULTS, 'runtime_matrix.csv'), index_col=0)
        assert util.check_dataframes(error_matrix, runtime_matrix)
        self.column_headings = np.array([eval(heading) for heading in list(error_matrix)])
        self.error_matrix = error_matrix.values
        self.X, self.Y, _ = linalg.pca(self.error_matrix, rank=min(self.error_matrix.shape)-1)
        self.runtime_matrix = runtime_matrix.values
        self.runtime_matrix_df = runtime_matrix #(temporary) the dataframe containing runtime matrix
        self.new_row = np.zeros((1, self.error_matrix.shape[1]))

        # ensemble attributes
        self.ensemble = Ensemble(self.p_type, stacking_alg, stacking_hyperparams)
        self.giant_ensemble = giant_ensemble

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

        # (temporary) custom runtime matrix
        t_predicted = convex_opt.predict_runtime(x_train.shape, runtime_matrix=self.runtime_matrix_df)

        if self.selection_method == 'qr':
            known_indices = linalg.pivot_columns(self.error_matrix)
        elif self.selection_method == 'min_variance':
            Y = self.Y[:rank, :]
            # TODO: Is 50/50 allocation of time to sampling & fitting appropriate?
            self.v_opt = convex_opt.solve(t_predicted, self.n_cores * runtime_limit/2, Y, self.scalarization)
            known_indices = np.where(self.v_opt > 0.99)[0]
        else:
            known_indices = np.arange(0, self.new_row.shape[1])

        # only need to compute column entry if it has not been computed already
        to_sample = list(set(known_indices) - self.known_indices)
        if self.verbose:
            print('Sampling {} entries of new row...'.format(len(to_sample)))
        start = time.time()
        pool1 = mp.Pool(self.n_cores)
        # TODO: Determine appropriate number of folds for k-fold fit/validate (currently 5)
        sample_models = [Model(self.p_type, self.column_headings[i]['algorithm'],
                               self.column_headings[i]['hyperparameters'], verbose=self.verbose) for i in to_sample]
        sample_model_errors = [pool1.apply_async(Model.kfold_fit_validate, args=[m, x_train, y_train, 5])
                               for m in sample_models]
        pool1.close()
        pool1.join()
        remaining = (runtime_limit - (time.time()-start)) * self.n_cores

        for i, error in enumerate(sample_model_errors):
            self.new_row[:, to_sample[i]] = error.get()[0].mean()
        imputed = linalg.impute(self.error_matrix, self.new_row, known_indices, rank=rank)

        # update known indices; impute ONLY unknown entries
        self.known_indices = set(known_indices)
        unknown = sorted(list(set(range(self.new_row.shape[1])) - self.known_indices))
        self.new_row[:, unknown] = imputed[:, unknown]

        # add every sampled model to ensemble
        if self.giant_ensemble:
            for model in sample_models:
                self.ensemble.add_base_learner(model)

        # solve knapsack problem to select models to add to ensemble
        # TODO: Determine rounding scheme to discretize knapsack problem
        weights = t_predicted.astype(int)
        values = 1e3*(1-self.new_row)[0].astype(int)
        best_indices = util.knapsack(weights, values, int(remaining))

        # add models selected by knapsack problem to ensemble
        if self.verbose:
            print('Newly added models:', best_indices - self.best_indices)
        for i in best_indices - self.best_indices:
            m = Model(self.p_type, self.column_headings[i]['algorithm'], self.column_headings[i]['hyperparameters'],
                      verbose=self.verbose)
            self.ensemble.add_base_learner(m)
        self.best_indices = best_indices

        if self.verbose:
            print('\nFitting optimized ensemble of size {}...'.format(len(self.ensemble.base_learners)))
        self.ensemble.fit(x_train, y_train)
        self.ensemble.fitted = True

        if self.verbose:
            print('\nAutoLearner fitting complete.')

    def fit_doubling(self, x_train, y_train, verbose=False):
        """Fit an AutoLearner object, iteratively doubling allowed runtime."""
        t_predicted = convex_opt.predict_runtime(x_train.shape)

        # split data into training and validation sets
        x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.15)

        ranks = [linalg.approx_rank(self.error_matrix, threshold=0.05)]
        times = [2**np.floor(np.log2(np.sort(t_predicted)[:4*ranks[0]].sum()))]
        losses = [1.0]

        v_opt, e_hat, best_idx = [], [], []
        k, t = ranks[0], times[0]
        best_new_row, best_ensemble = None, None

        start = time.time()
        while time.time() - start < self.runtime_limit - t:
            if verbose:
                print('Fitting with k={}, t={}'.format(k, t))
            self.fit(x_tr, y_tr, rank=k, runtime_limit=t)
            v_opt.append(self.v_opt)
            e_hat.append(self.new_row)
            best_idx.append(self.best_indices)
            loss = util.error(y_va, self.ensemble.predict(x_va), self.p_type)
            losses.append(loss)

            if loss < min(losses):
                ranks.append(k+1)
                best_new_row, best_ensemble = np.copy(self.new_row), copy.deepcopy(self.ensemble)
            else:
                ranks.append(k)

            times.append(2*t)
            k = ranks[-1]
            t = times[-1]

        # after all iterations, restore best model
        self.new_row = best_new_row
        self.ensemble = best_ensemble
        return {'k': ranks, 't': times, 'l': losses, 'v': v_opt, 'e': e_hat, 'b': best_idx}

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

