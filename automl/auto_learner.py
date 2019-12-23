import convex_opt
import json
import linalg
import multiprocessing as mp
import numpy as np
import scipy as sp
import os
import pickle
import pandas as pd
import pkg_resources
import time
import copy
import util
from pipeline import PipelineObject, Ensemble, Model_collection
import experiment_design as ED
from sklearn.model_selection import train_test_split
import tensorly as tl
import signal
from contextlib import contextmanager

# load default error and runtime tensors

DEFAULTS = pkg_resources.resource_filename(__name__, 'defaults')
ERROR_TENSOR = np.load(os.path.join(DEFAULTS, 'error_tensor.npy'))
RUNTIME_TENSOR = np.load(os.path.join(DEFAULTS, 'runtime_tensor.npy'))
with open(os.path.join(DEFAULTS, 'training_index.pkl'), 'rb') as handle:
    TRAINING_INDEX = pickle.load(handle)
with open(os.path.join(DEFAULTS, 'pipelines.pkl'), 'rb') as handle:
    PIPELINES = pickle.load(handle)
    
    
class AutoLearner:
    """An object that automatically selects pipelines by greedy D-optimal design.

    Attributes:
        p_type (str):                  Problem type. One of {'classification', 'regression'}.
        algorithms (list):             A list of algorithm types to be considered, in strings. (e.g. ['KNN', 'lSVM']).
        hyperparameters (dict):        A nested dict of hyperparameters to be considered.
        n_folds (int):                 Number of cross-validation folds.
        verbose (bool):                Whether or not to generate print statements that showcase the progress.
        n_cores (int):                 Maximum number of cores over which to parallelize (None means no limit).
        runtime_limit (int):           Maximum training time for AutoLearner, in seconds.
        dataset_ratio_threshold(float):The threshold of dataset ratio for dataset subsampling, if the training set is tall and skinny (number of data points much larger than number of features).
        new_row (np.ndarray):          Predicted row of error matrix.
        runtime_predictor (str):       Model for runtime prediction. One of {'LinearRegression', 'KNeighborsRegressor'}.
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
                 n_cores=mp.cpu_count(), n_folds=3, runtime_limit=512, dataset_ratio_threshold=100,
                 new_row=None, load_imputed=True, selection_method='ED', build_ensemble=True, ensemble_method='greedy', runtime_predictor='LinearRegression',
                 **stacking_hyperparams):
        
        self.verbose = verbose
        self.selection_method = selection_method
        # pipeline configurations
        self.p_type = p_type.lower()
        with open(os.path.join(DEFAULTS, self.p_type + '.json')) as file:
            defaults = json.load(file)
        
        # attributes of ML problem        
        self.algorithms = algorithms or defaults['estimator']['algorithms']
        self.hyperparameters = hyperparameters or defaults['estimator']['hyperparameters']
        
        # computational considerations
        self.n_cores = n_cores
        self.runtime_limit = runtime_limit
        self.n_folds = n_folds
        
        # error tensor completion
        ranks_for_imputation = (20, 4, 2, 2, 8, 20)
        if load_imputed:            
            try:
                error_tensor_imputed = np.load(os.path.join(DEFAULTS, 'error_tensor_imputed.npy'))
            except:
                print("no files!")

        else:
            _, _, error_tensor_imputed, _ = util.tucker_on_error_tensor(ERROR_TENSOR, ranks_for_imputation, save_results=True, verbose=self.verbose)
        
        # error tensor factorization
        # TODO: determine whether to generate new error matrix or use default/subset of default        
        
        # the maximum ranks for dataset and estimator dimensions
        k_dataset_for_factorization = 30
        k_estimator_for_factorization = 30
        
        core_tr, factors_tr = tl.decomposition.tucker(error_tensor_imputed, ranks=(k_dataset_for_factorization, 4, 2, 2, 8, k_estimator_for_factorization))
        pipeline_latent_factors = tl.unfold(tl.tenalg.multi_mode_dot(core_tr, factors_tr[1:], modes=[1, 2, 3, 4, 5]), mode=0)
        U_t, S_t, Vt_t = sp.linalg.svd(pipeline_latent_factors, full_matrices=False)
        Y_pca = Vt_t
        self.Y = Y_pca
        
        self.error_tensor_imputed = error_tensor_imputed
        
        # not yet implemented the part of selecting specific algorithms and hyperparameters
        self.error_matrix = tl.unfold(error_tensor_imputed, mode=0)        
        self.runtime_matrix = tl.unfold(RUNTIME_TENSOR, mode=0)        
        
#         training_index = [eval(configs_matrix[i, 0])['dataset'] for i in range(configs_matrix.shape[0])]
        self.training_index = TRAINING_INDEX
        
#         pipelines = [eval(self.configs_matrix[0, i]) for i in range(self.configs_matrix.shape[1])]
#         for p in pipelines:
#             del p['dataset']
        
        pipelines = PIPELINES
        
        self.pipeline_settings = []
        for item in pipelines:
            r = {}
            for key in item:
                r[key] = eval(item[key])            
            self.pipeline_settings.append(r)        

#         # sampled & fitted models
        self.new_row = new_row or np.full((1, len(pipelines)), np.nan)
        self.new_row_pred = new_row or np.full((1, len(pipelines)), np.nan)
        self.sampled_indices = set()
        self.sampled_pipelines = [None] * len(pipelines)
        self.fitted_indices = set()
        self.fitted_pipelines = [None] * len(pipelines)

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
        self.dataset_ratio_threshold = dataset_ratio_threshold

    def _fit(self, x_train, y_train, t_predicted, ranks=None, runtime_limit=None):
        """This private method is a single round of the doubling process. It fits an AutoLearner object on a new dataset.
        This will sample the performance of several algorithms on the new dataset, predict performance on the rest, then construct an optimal ensemble model.

        Args:
            x_train (np.ndarray):  Features of the training dataset.
            y_train (np.ndarray):  Labels of the training dataset.
            t_predicted (np.ndarray): Predicted running time.
            ranks (int):            Rank of error tensor factorization.
            runtime_limit (float): Maximum time to allocate to AutoLearner fitting.
        """
        if self.verbose:
            print("\nSingle round runtime target: {}".format(runtime_limit))
        
        # set to defaults if not provided
        ranks = ranks or linalg.approx_tensor_rank(self.error_tensor_imputed, threshold=0.01)
        runtime_limit = runtime_limit or self.runtime_limit
        
        
        if self.verbose:
            print('Fitting AutoLearner with maximum runtime {} seconds'.format(runtime_limit))

        # cold-start: pick the initial set of models to fit on the new dataset
#         if self.selection_method == 'qr':
#             to_sample = linalg.pivot_columns(self.error_matrix)
#         if self.selection_method == 'min_variance':
            # select algorithms to sample only from subset of algorithms that will run in allocated time
        valid = np.where(t_predicted <= self.n_cores * runtime_limit/2)[0]
        Y = self.Y[:ranks[0], valid]
        
        selected_columns_qr, t_sum, case = ED.pivot_columns_time(Y, t_predicted[valid], runtime_limit/2, 
                                                        columns_to_avoid=None,
                                                        rank=Y.shape[0])
        if case == 'greedy_initialization':
            if self.verbose:
                print(case)
            to_sample = selected_columns_qr

        elif case == 'qr_initialization':        
            to_sample = ED.greedy_stepwise_selection_with_time(Y=Y,
                                                        t=t_predicted[valid],
                                                        initialization=selected_columns_qr,
                                                        t_elapsed=t_sum,
                                                        t_max=runtime_limit/2,
                                                        idx_to_exclude=None)
        # TODO: check if Y is rank-deficient, i.e. will ED problem fail

        
        if np.isnan(to_sample).any():
            to_sample = np.argsort(t_predicted)[:ranks[0]]
        
#         elif self.selection_method == 'random':
#             to_sample = []
#             # set of algorithms that are predicted to finish within given budget
#             to_sample_candidates = np.where(t_predicted <= runtime_limit/2)[0]
#             # remove algorithms that have been sampled already
#             to_sample_candidates = list(set(to_sample_candidates) - self.sampled_indices)
#             # if the remaining time is not sufficient for random sampling
#             if len(to_sample_candidates) == 0:
#                 to_sample = np.array([np.argmin(t_predicted)])
#             else:
#                 to_sample = np.random.choice(to_sample_candidates, min(self.n_cores, len(to_sample_candidates)), replace=False)
#         else:
#             to_sample = np.arange(0, self.new_row.shape[1])
        
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
            # we only need to fit models on the new dataset if it has not been fitted already
            to_sample = list(set(to_sample) - self.sampled_indices)
            if self.verbose:
                print('Sampling {} entries of new row...'.format(len(to_sample)))
                  
            p1 = mp.Pool(self.n_cores)
            sampled_pipelines_single_round = [PipelineObject(p_type=self.p_type, config=self.pipeline_settings_on_dataset[i], index=i, verbose=self.verbose) for i in to_sample]
            sampled_pipeline_errors_single_round = [p1.apply_async(PipelineObject.kfold_fit_validate, args=[p, x_train, y_train, self.n_folds])
                                   for p in sampled_pipelines_single_round]
            p1.close()
            p1.join()

            # predict performance of models not actually fitted on the new dataset
            self.sampled_indices = self.sampled_indices.union(set(to_sample))
            for i, error in enumerate(sampled_pipeline_errors_single_round):
                cv_error, cv_predictions = error.get()
                sampled_pipelines_single_round[i].cv_error, sampled_pipelines_single_round[i].cv_predictions = cv_error, cv_predictions
                sampled_pipelines_single_round[i].sampled = True
                self.new_row[:, to_sample[i]] = cv_error
                self.sampled_pipelines[to_sample[i]] = sampled_pipelines_single_round[i]
            self.new_row_pred = linalg.impute_with_coefficients(self.Y[:ranks[0], :], self.new_row, list(self.sampled_indices))
            
            for idx in np.argsort(self.new_row[0, :])[:5]: # automatically put nans at the end of the list
                print(self.sampled_pipelines[idx])
                self.ensemble.candidate_learners.append(self.sampled_pipelines[idx])

            # impute ALL entries
            # unknown = sorted(list(set(range(self.new_row.shape[1])) - self.sampled_indices))
            # self.new_row[:, unknown] = imputed[:, unknown]

            # k-fold fit candidate learners of ensemble
            remaining = (runtime_limit - (time.time() - start)) * self.n_cores
            
            if remaining > 0:
            # add models predicted to be the best to list of candidate learners to avoid empty lists
            
                print("length of sampled indices: {}".format(len(self.sampled_indices)))

    #             best_sampled_idx = list(self.sampled_indices)[int(np.argmin(self.new_row[:, list(self.sampled_indices)]))]
    #             assert self.sampled_pipelines[best_sampled_idx] is not None
    #             candidate_indices = [best_sampled_idx]
                candidate_indices = []
    #             self.ensemble.candidate_learners.append(self.sampled_pipelines[best_sampled_idx])
                for i in np.argsort(self.new_row_pred[0]):
                    if t_predicted[i] + t_predicted[candidate_indices].sum() <= remaining:
                        candidate_indices.append(i)
                        # last = candidate_indices.pop()
                        # assert last == best_sampled_idx
    #                     candidate_indices.append(i)
                        # if model has already been k-fold fitted, immediately add to candidate learners
                        if i in self.sampled_indices:
                            assert self.sampled_pipelines[i] is not None
                            self.ensemble.candidate_learners.append(self.sampled_models[i])
                # candidate learners that need to be k-fold fitted
                to_fit = list(set(candidate_indices) - self.sampled_indices)
                print("to fit: {}".format(to_fit))
        else:
            remaining = (runtime_limit - (time.time()-start)) * self.n_cores
            to_fit = to_sample.copy()
        
        remaining = (runtime_limit - (time.time() - start)) * self.n_cores
            
        if remaining > 0:
        
            if len(to_fit) > 0:
                # fit models predicted to have good performance and thus going to be added to the ensemble
                p2 = mp.Pool(self.n_cores)
                candidate_pipelines = [PipelineObject(p_type=self.p_type, config=self.pipeline_settings_on_dataset[i], index=i, verbose=self.verbose) for i in to_fit]
                candidate_pipeline_errors = [p2.apply_async(PipelineObject.kfold_fit_validate, args=[p, x_train, y_train, self.n_folds])
                                          for p in candidate_pipelines]
                p2.close()
                p2.join()

                # update sampled indices
                self.sampled_indices = self.sampled_indices.union(set(to_fit))
                for i, error in enumerate(candidate_pipeline_errors):
                    cv_error, cv_predictions = error.get()
                    candidate_pipelines[i].cv_error, candidate_pipelines[i].cv_predictions = cv_error, cv_predictions
                    candidate_pipelines[i].sampled = True
                    self.new_row[:, to_fit[i]] = cv_error
                    self.sampled_pipelines[to_fit[i]] = candidate_pipelines[i]
                    self.ensemble.candidate_learners.append(candidate_pipelines[i])
                # self.new_row = linalg.impute(self.error_matrix, self.new_row, list(self.sampled_indices), rank=rank)

        if self.verbose:
            print('\nFitting ensemble of max. size {}...'.format(len(self.ensemble.candidate_learners)))
        # ensemble selection and fitting in the remaining time budget
        self.ensemble.fit(x_train, y_train, remaining, self.sampled_pipelines)
        for pipeline in self.ensemble.base_learners:
            assert pipeline.index is not None
            self.sampled_indices.add(pipeline.index)
            self.sampled_pipelines[pipeline.index] = pipeline
        self.ensemble.fitted = True

        if self.verbose:
            print('\nAutoLearner fitting complete.')

            
    def fit(self, x_train, y_train, verbose=False):

        """Fit an AutoLearner object, iteratively doubling allowed runtime, and terminate when reaching the time limit."""
        
        num_points, num_features = x_train.shape
        
        if self.verbose:
            print('\nShape of training dataset: {} data points, {} features'.format(num_points, num_features))
        
        if num_points > 10000 and num_points / num_features > self.dataset_ratio_threshold:
            num_points_new = int(min(5000, num_features * self.dataset_ratio_threshold))
            sampling_ratio = num_points_new / num_points
            print(sampling_ratio)
            df_x_train = pd.DataFrame(x_train)
            df_y_train = pd.DataFrame(y_train, columns=['labels'])
            df_resampled = df_x_train.join(df_y_train).groupby('labels').apply(pd.DataFrame.sample, frac=sampling_ratio).reset_index(drop=True)
            x_train = df_resampled.drop(['labels'], axis=1).values
            y_train = df_resampled['labels'].values
            if self.verbose:
                print('\nTraining dataset resampled \nShape of resampled training dataset: {} data points, {} features'.format(x_train.shape[0], x_train.shape[1]))
        
        
        def p2f(x):
            return float(x.strip('%'))/100
        
        # pipeline settings on dataset        
        self.pipeline_settings_on_dataset = []
        
        for item in self.pipeline_settings:
            item_copy = copy.deepcopy(item)
            if item_copy['dim_reducer']['algorithm'] == 'PCA':
                item_copy['dim_reducer']['hyperparameters']['n_components'] = int(p2f(item_copy['dim_reducer']['hyperparameters']['n_components']) * x_train.shape[1])
            elif item_copy['dim_reducer']['algorithm'] == 'SelectKBest':
                item_copy['dim_reducer']['hyperparameters']['k'] = int(p2f(item_copy['dim_reducer']['hyperparameters']['k']) * x_train.shape[1])
                
            self.pipeline_settings_on_dataset.append(item_copy)
        
        
        # predict runtime for the training set of the new dataset.
        t_predicted = convex_opt.predict_runtime(x_train.shape, runtime_matrix=self.runtime_matrix, runtimes_index=self.training_index)

        # split data into training and validation sets
        try:
            x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train, random_state=0)
        except ValueError:
            x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.15, random_state=0)

        ranks = [linalg.approx_tensor_rank(self.error_tensor_imputed, threshold=0.05)]
        if self.build_ensemble:
            t_init = 2**np.floor(np.log2(np.sort(t_predicted)[:int(2*ranks[0][-1])].sum()))
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
                self._fit(x_tr, y_tr, t_predicted, ranks=k, runtime_limit=t)
                if self.build_ensemble and self.ensemble.fitted:
                    if self.verbose:
                        print("\nGot a new ensemble in the round with rumtime target {} seconds".format(t))
                    loss = util.error(y_va, self.ensemble.predict(x_va), self.p_type)

                    # TEMPORARY: Record intermediate results

                    e_hat.append(np.copy(self.new_row))
                    actual_times.append(time.time() - start)
                    sampled.append(self.sampled_indices)
                    ensembles.append(self.ensemble)
                    losses.append(loss)

                    if loss == min(losses):
                        ranks.append((k[0], k[1], k[2], k[3], k[4], k[5]+1))
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
            # set aside 3 seconds for initial and final processing steps
            with time_limit(self.runtime_limit - 3):
                doubling()
        except TimeoutException as e:
            if verbose:
                print("Time limit reached.")

        if self.build_ensemble:
            # after all iterations, restore best model
            
            try:
                self.new_row = e_hat[self.best]
                self.ensemble = ensembles[self.best]

                return {'ranks': ranks[:-1], 'runtime_limits': times[:-1], 'validation_loss': losses,
                    'predicted_new_row': e_hat, 'actual_runtimes': actual_times, 'sampled_indices': sampled,
                    'models': ensembles}
            except IndexError:
                print("No ensemble built within time limit. Please try increasing the time limit or allocate more computational resources.")
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

