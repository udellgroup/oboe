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
from model import Model, Ensemble, Model_collection
from sklearn.model_selection import train_test_split
import signal
from contextlib import contextmanager

DEFAULTS = pkg_resources.resource_filename(__name__, 'defaults')
ERROR_MATRIX = pd.read_csv(os.path.join(DEFAULTS, 'error_matrix.csv'), index_col=0, header=0)
RUNTIME_MATRIX = pd.read_csv(os.path.join(DEFAULTS, 'runtime_matrix.csv'), index_col=0, header=0)


class AutoLearner:
    """An object representing an automatically tuned machine learning model.

    Attributes:
        p_type (str):                  Problem type. One of {'classification', 'regression'}.
        algorithms (list):             A list of algorithm types to be considered, in strings. (e.g. ['KNN', 'lSVM']).
        hyperparameters (dict):        A nested dict of hyperparameters to be considered.
        verbose (bool):                Whether or not to generate print statements that showcase the progress.
        n_cores (int):                 Maximum number of cores over which to parallelize (None means no limit).
        runtime_limit(int):            Maximum training time for AutoLearner, in seconds.
        selection_method (str):        Method of selecting entries of new row to sample.
        scalarization (str):           Scalarization of the covariance matrix for mininum variance selection. One of {'D', 'A', 'E'}.
        error_matrix (DataFrame):      Error matrix to use for imputation; includes index and headers.
        runtime_matrix (DataFrame):    Runtime matrix to use for runtime prediction; includes index and headers.
        new_row (np.ndarray):          Predicted row of error matrix.
        runtime_predictor (str):       Model for runtime prediction. One of {'LinearRegression', 'KNeighborsRegressor'}.
        solver (str):                  The convex solver for classic experiment design problem. One of {'scipy', 'cvxpy'}.
        ensemble_method (str):         Ensemble method. One of {'greedy', 'stacking'}.
        
    Attributes to be added in the future:
        X, Y (np.ndarray):             PCA decomposition of error matrix.
        sampled_indices (set):         Indices of new row that have been sampled.
        sampled_models (list):         List of models that have been sampled (i.e. k-fold fitted).
        fitted_indices (set):          Indices of new row that have been fitted (i.e. included in ensemble)
        fitted_models (list):          List of models that have been fitted.
    """

    def __init__(self,
                 p_type='classification', algorithms=None, hyperparameters=None, verbose=False,
                 n_cores=mp.cpu_count(), runtime_limit=512,
                 selection_method='min_variance', scalarization='D',
                 error_matrix=None, runtime_matrix=None, new_row=None,
                 build_ensemble=True, ensemble_method='greedy', runtime_predictor='LinearRegression', solver='scipy',
                 **stacking_hyperparams):

        # TODO: check if arguments to constructor are valid; set to defaults if not specified
        assert selection_method in {'qr', 'min_variance', 'random'}, "The method to select entries to sample must be " \
            "either qr (QR decomposition), min_variance (minimize variance with time constraints), or random (time-constrained random selection, for testing purpose)."

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
        self.error_matrix = util.extract_columns(ERROR_MATRIX, self.algorithms, self.hyperparameters) if error_matrix is None else error_matrix
        self.runtime_matrix = util.extract_columns(RUNTIME_MATRIX, self.algorithms, self.hyperparameters) if runtime_matrix is None else runtime_matrix
        assert util.check_dataframes(self.error_matrix, self.runtime_matrix)
        self.column_headings = np.array([eval(heading) for heading in list(self.error_matrix)])
        self.X, self.Y, _ = linalg.pca(self.error_matrix.values, rank=min(self.error_matrix.shape)-1)

        # sampled & fitted models
        self.new_row = new_row or np.zeros((1, self.error_matrix.shape[1]))
        self.sampled_indices = set()
        self.sampled_models = [None] * self.error_matrix.shape[1]
        self.fitted_indices = set()
        self.fitted_models = [None] * self.error_matrix.shape[1]

        # ensemble attributes
        self.build_ensemble = build_ensemble
        self.ensemble_method = ensemble_method
        self.stacking_hyperparams = stacking_hyperparams
        if self.build_ensemble:
            self.ensemble = Ensemble(self.p_type, self.ensemble_method, self.stacking_hyperparams)
        else:
            self.ensemble = Model_collection(self.p_type)
        
        # runtime predictor
        self.runtime_predictor = runtime_predictor
        
        # convex solver
        self.solver = solver

    def _fit(self, x_train, y_train, rank=None, runtime_limit=None):
        """This is a single round of the doubling process. It fits an AutoLearner object on a new dataset. This will sample the performance of several algorithms on the new dataset, predict performance on the rest, then construct an optimal ensemble model.

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

        if self.selection_method == 'qr':
            to_sample = linalg.pivot_columns(self.error_matrix)
        elif self.selection_method == 'min_variance':
            # select algorithms to sample only from subset of algorithms that will run in allocated time
            valid = np.where(t_predicted <= self.n_cores * runtime_limit/2)[0]
            Y = self.Y[:rank, valid]
            # TODO: check if Y is rank-deficient, i.e. will ED problem fail?
            v_opt = convex_opt.solve(t_predicted[valid], runtime_limit/4, self.n_cores, Y, self.scalarization, self.solver)
            to_sample = valid[np.where(v_opt > 0.9)[0]]
            if np.isnan(to_sample).any():
                to_sample = np.argsort(t_predicted)[:rank]
        
        elif self.selection_method == 'random':
            to_sample = []
            # set of algorithms that are predicted to finish within given budget
            to_sample_candidates = np.where(t_predicted <= runtime_limit/2)[0]
            # remove algorithms that have been sampled already
            to_sample_candidates = list(set(to_sample_candidates) - self.sampled_indices)
            # if the remaining time is not sufficient for random sampling
            if len(to_sample_candidates) == 0:
                to_sample = np.array([np.argmin(t_predicted)])
            else:
                to_sample = np.random.choice(to_sample_candidates, min(self.n_cores, len(to_sample_candidates)), replace=False)
        else:
            to_sample = np.arange(0, self.new_row.shape[1])
        
        if len(to_sample) == 0 and len(self.sampled_indices) == 0:
            # if no columns are selected in first iteration (log det instability), sample n fastest columns
            n = len(np.where(np.cumsum(np.sort(t_predicted)) <= runtime_limit/4)[0])
            if n > 0:
                to_sample = np.argsort(t_predicted)[:n]
            else:
                self.ensemble.fitted = False
                return
            
        start = time.time()        
        if self.selection_method is not 'random':            
            # only need to compute column entry if it has not been computed already
            to_sample = list(set(to_sample) - self.sampled_indices)
            if self.verbose:
                print('Sampling {} entries of new row...'.format(len(to_sample)))
                  
            p1 = mp.Pool(self.n_cores)
            sample_models = [Model(self.p_type, self.column_headings[i]['algorithm'],
                                   self.column_headings[i]['hyperparameters'], self.verbose, i) for i in to_sample]
            sample_model_errors = [p1.apply_async(Model.kfold_fit_validate, args=[m, x_train, y_train, 5])
                                   for m in sample_models]
            p1.close()
            p1.join()

            # update sampled indices
            self.sampled_indices = self.sampled_indices.union(set(to_sample))
            for i, error in enumerate(sample_model_errors):
                cv_error, cv_predictions = error.get()
                sample_models[i].cv_error, sample_models[i].cv_predictions = cv_error.mean(), cv_predictions
                sample_models[i].sampled = True
                self.new_row[:, to_sample[i]] = cv_error.mean()
                self.sampled_models[to_sample[i]] = sample_models[i]
            imputed = linalg.impute(self.error_matrix, self.new_row, list(self.sampled_indices), rank=rank)

            # impute ALL entries
            # unknown = sorted(list(set(range(self.new_row.shape[1])) - self.sampled_indices))
            # self.new_row[:, unknown] = imputed[:, unknown]
            self.new_row = imputed.copy()

            # k-fold fit candidate learners of ensemble
            remaining = (runtime_limit - (time.time()-start)) * self.n_cores
            # add best sampled model to list of candidate learners to avoid empty lists
            best_sampled_idx = list(self.sampled_indices)[int(np.argmin(self.new_row[:, list(self.sampled_indices)]))]
            assert self.sampled_models[best_sampled_idx] is not None
            candidate_indices = [best_sampled_idx]
            self.ensemble.candidate_learners.append(self.sampled_models[best_sampled_idx])
            for i in np.argsort(self.new_row[0]):
                if t_predicted[i] + t_predicted[candidate_indices].sum() <= remaining:
                    last = candidate_indices.pop()
                    assert last == best_sampled_idx
                    candidate_indices.append(i)
                    candidate_indices.append(last)
                    # if model has already been k-fold fitted, immediately add to candidate learners
                    if i in self.sampled_indices:
                        assert self.sampled_models[i] is not None
                        self.ensemble.candidate_learners.append(self.sampled_models[i])
            # candidate learners that need to be k-fold fitted
            to_fit = list(set(candidate_indices) - self.sampled_indices)
        else:
            remaining = (runtime_limit - (time.time()-start)) * self.n_cores
            to_fit = to_sample.copy()

        p2 = mp.Pool(self.n_cores)
        candidate_models = [Model(self.p_type, self.column_headings[i]['algorithm'],
                                  self.column_headings[i]['hyperparameters'], self.verbose, i) for i in to_fit]
        candidate_model_errors = [p2.apply_async(Model.kfold_fit_validate, args=[m, x_train, y_train, 5])
                                  for m in candidate_models]
        p2.close()
        p2.join()

        # update sampled indices
        self.sampled_indices = self.sampled_indices.union(set(to_fit))
        for i, error in enumerate(candidate_model_errors):
            cv_error, cv_predictions = error.get()
            candidate_models[i].cv_error, candidate_models[i].cv_predictions = cv_error.mean(), cv_predictions
            candidate_models[i].sampled = True
            self.new_row[:, to_fit[i]] = cv_error.mean()
            self.sampled_models[to_fit[i]] = candidate_models[i]
            self.ensemble.candidate_learners.append(candidate_models[i])
        # self.new_row = linalg.impute(self.error_matrix, self.new_row, list(self.sampled_indices), rank=rank)

        if self.verbose:
            print('\nFitting ensemble of max. size {}...'.format(len(self.ensemble.candidate_learners)))
        # ensemble selection and fitting in the remaining time budget
        self.ensemble.fit(x_train, y_train, remaining, self.fitted_models)
        for model in self.ensemble.base_learners:
            assert model.index is not None
            self.fitted_indices.add(model.index)
            self.fitted_models[model.index] = model
        self.ensemble.fitted = True

        if self.verbose:
            print('\nAutoLearner fitting complete.')

            
    def fit(self, x_train, y_train, verbose=False):

        """Fit an AutoLearner object, iteratively doubling allowed runtime, and terminate when reaching the time limit."""
        
        num_points, num_features = x_train.shape
        
        # subsample the dataset if it is tall and skinny, in which case runtime predictors have poor empirical extrapolation performance
        if num_points > 10000 and num_points / num_features > self.dataset_ratio_threshold:
            num_points_new = int(min(5000, num_features * self.dataset_ratio_threshold))
            sampling_ratio = num_points_new / num_points
            print(sampling_ratio)
            df_x_train = pd.DataFrame(x_train)
            df_y_train = pd.DataFrame(y_train, columns=['labels'])
            df_resampled = df_x_train.join(df_y_train).groupby('labels').apply(pd.DataFrame.sample, frac=sampling_ratio).reset_index(drop=True)
            x_train = df_resampled.drop(['labels'], axis=1).values
            y_train = df_resampled['labels'].values
        
        t_predicted = convex_opt.predict_runtime(x_train.shape, model_name=self.runtime_predictor)

        # split data into training and validation sets
        try:
            x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train, random_state=0)
        except ValueError:
            x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.15, random_state=0)

        ranks = [linalg.approx_rank(self.error_matrix, threshold=0.05)]
        if self.build_ensemble:
            t_init = 2**np.floor(np.log2(np.sort(t_predicted)[:int(1.1*ranks[0])].sum()))
            t_init = max(1, t_init)
        else:
            t_init = self.runtime_limit / 2
        times = [t_init]
        losses = [0.5]

        e_hat, actual_times, sampled, ensembles = [], [], [], []        

        start = time.time()
        
        def doubling():
            k, t = ranks[0], times[0]
            counter, self.best = 0, 0            
            while time.time() - start < self.runtime_limit - t:
                if verbose:
                    print('Fitting with k={}, t={}'.format(k, t))
                if self.build_ensemble:
                    self.ensemble = Ensemble(self.p_type, self.ensemble_method, self.stacking_hyperparams)
                else:
                    self.ensemble = Model_collection(self.p_type)
                self._fit(x_tr, y_tr, rank=k, runtime_limit=t)
                if self.build_ensemble and self.ensemble.fitted:
                    print("got a new ensemble")
                    loss = util.error(y_va, self.ensemble.predict(x_va), self.p_type)

                    # TEMPORARY: Record intermediate results

                    e_hat.append(np.copy(self.new_row))
                    actual_times.append(time.time() - start)
                    sampled.append(self.sampled_indices)
                    ensembles.append(self.ensemble)
                    losses.append(loss)

                    if loss == min(losses):
                        ranks.append(k+1)
                        self.best = counter
                    else:
                        ranks.append(k)
                    
                    counter += 1

                times.append(2*t)
                k = ranks[-1]
                t = times[-1]
                
        class TimeoutException(Exception): pass

        @contextmanager
        def time_limit(seconds):
            def signal_handler(signum, frame):
                raise TimeoutException("Time limit reached.")
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

        try:
            with time_limit(self.runtime_limit):
                doubling()
        except TimeoutException as e:
            if verbose:
                print("Time limit reached.")

        if self.build_ensemble:
            # after all iterations, restore best model
            self.new_row = e_hat[self.best]
            self.ensemble = ensembles[self.best]

            return {'ranks': ranks[:-1], 'runtime_limits': times[:-1], 'validation_loss': losses,
                'predicted_new_row': e_hat, 'actual_runtimes': actual_times, 'sampled_indices': sampled,
                'models': ensembles}
        else:
            return

    def refit(self, x_train, y_train):
        """Refit an existing AutoLearner object on a new dataset. This will simply retrain the base-learners and
        stacked learner of an existing model, and so algorithm and hyperparameter selection may not be optimal.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        assert self.ensemble.fitted, "Cannot refit unless model has been fit."
        self.ensemble.refit(x_train, y_train)

    def predict(self, x_test):
        """Generate predictions on test data.

        Args:
            x_test (np.ndarray): Features of the test dataset.
        Returns:
            np.ndarray: Predicted labels.
        """
        return self.ensemble.predict(x_test)

    def get_models(self):
        """Get details of the selected machine learning models and the ensemble.
        """
        return self.ensemble.get_models()

    def get_model_accuracy(self, y_test):
        """ Get accuracies of selected models.
        """
        return self.ensemble.get_model_accuracy(y_test)

