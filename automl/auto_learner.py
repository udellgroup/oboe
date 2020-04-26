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
from scipy.stats import mode
from contextlib import contextmanager

# load default error and runtime tensors

DEFAULTS = pkg_resources.resource_filename(__name__, 'defaults')
ERROR_TENSOR = np.load(os.path.join(DEFAULTS, 'error_tensor.npy'))
RUNTIME_TENSOR = np.load(os.path.join(DEFAULTS, 'runtime_tensor.npy'))
with open(os.path.join(DEFAULTS, 'training_index.pkl'), 'rb') as handle:
    TRAINING_INDEX = pickle.load(handle)    
    
class AutoLearner:
    """An object that automatically selects pipelines by greedy D-optimal design.

    Basic attributes:
        p_type (str):                  Problem type. One of {'classification', 'regression'}.
        algorithms (list):             A list of algorithm types to be considered, in strings. (e.g. ['KNN', 'lSVM']).
        hyperparameters (dict):        A nested dict of hyperparameters to be considered.
        n_folds (int):                 Number of cross-validation folds.
        verbose (bool):                Whether or not to generate print statements that showcase the progress.
        n_cores (int):                 Maximum number of cores over which to parallelize (None means no limit).
        runtime_limit (int):           Maximum training time for AutoLearner, in seconds.        
        
    Advanced attributes:
        new_row (np.ndarray):          Predicted row of matricized error tensor, corresponding to the new dataset. Default None. 
        dataset_ratio_threshold(float):The threshold of dataset ratio (number of points / number of features) for dataset subsampling, if the training set is tall and skinny (number of data points much larger than number of features).
        runtime_predictor_algorithm (str):       
                                       Model for runtime prediction. One of {'LinearRegression', 'KNeighborsRegressor'}.
        ensemble_method (str):         Ensemble method. One of {'greedy', 'stacking', 'best_several'}.
        
    Attributes to be added in the future:
        X, Y (np.ndarray):             PCA decomposition of error matrix.
        sampled_indices (set):         Indices of new row that have been sampled.
        sampled_models (list):         List of models that have been sampled (i.e. k-fold fitted).
        fitted_indices (set):          Indices of new row that have been fitted (i.e. included in ensemble)
        fitted_models (list):          List of models that have been fitted.
    """

    def __init__(self,
                 p_type='classification', algorithms=None, hyperparameters=None, verbose=False,
                 n_cores=1, n_folds=3, runtime_limit=512, dataset_ratio_threshold=100,
                 new_row=None, load_imputed_error_tensor=True, path_to_imputed_error_tensor='default', save_imputed_error_tensor=True, 
                 selection_method='ED', build_ensemble=True, load_saved_latent_factors=True, save_latent_factors=True,
                 ensemble_method='best_several', ensemble_max_size=5, runtime_predictor_algorithm='LinearRegression',
                 random_state=0, **stacking_hyperparams):
        
        self.verbose = verbose
        self.random_state = random_state
        self.selection_method = selection_method
        # pipeline configurations
        self.p_type = p_type.lower()
        with open(os.path.join(DEFAULTS, self.p_type + '.json')) as file:
            defaults = json.load(file)
        
        # attributes of ML problem        
        self.algorithms = algorithms or defaults['estimator']['algorithms']
        self.hyperparameters = hyperparameters or defaults['estimator']['hyperparameters']
        
        # computational considerations
        if n_cores is None:
            n_cores = mp.cpu_count()
        self.n_cores = n_cores
        self.runtime_limit = runtime_limit
        self.n_folds = n_folds
        
        # error tensor completion
        ranks_for_imputation = (20, 4, 2, 10, 15, 40)
        if load_imputed_error_tensor:            
            try:
                if path_to_imputed_error_tensor == 'default':
                    error_tensor_imputed = np.load(os.path.join(DEFAULTS, 'error_tensor_imputed.npy'))
                else:
                    if self.verbose:
                        print("loading customized tensor at {} ...".format(path_to_imputed_error_tensor))
                    error_tensor_imputed = np.load(path_to_imputed_error_tensor)
            except:
                print("no files!")
        else:
            _, _, error_tensor_imputed, _ = util.tucker_on_error_tensor(ERROR_TENSOR, ranks_for_imputation, save_results=True, verbose=self.verbose)
            if save_imputed_error_tensor:
                np.save(os.path.join(DEFAULTS, 'error_tensor_imputed.npy'), error_tensor_imputed)
        
        self.error_tensor_imputed = error_tensor_imputed
        
        # error tensor factorization
        # TODO: determine whether to generate new error matrix or use default/subset of default        
        
        # the maximum ranks for dataset and estimator dimensions
        k_dataset_for_factorization = 20
        k_estimator_for_factorization = 40
        
        factorize_error_tensor = True
        if load_saved_latent_factors:
            try:
                U_t = np.load(os.path.join(DEFAULTS, 'error_tensor_U_t.npy'))
                S_t = np.load(os.path.join(DEFAULTS, 'error_tensor_S_t.npy'))
                Vt_t = np.load(os.path.join(DEFAULTS, 'error_tensor_Vt_t.npy'))
                factorize_error_tensor = False
            except:
                if self.verbose:
                    print("No saved latent factors. Factorizing the error tensor now ...")            
        
        if factorize_error_tensor:
            if self.verbose:
                print("Factorizing the error matrix to get latent factors ...")
            core_tr, factors_tr = tl.decomposition.tucker(error_tensor_imputed, ranks=(k_dataset_for_factorization, ranks_for_imputation[1], ranks_for_imputation[2], ranks_for_imputation[3], ranks_for_imputation[4], k_estimator_for_factorization))
            pipeline_latent_factors = tl.unfold(tl.tenalg.multi_mode_dot(core_tr, factors_tr[1:], modes=[1, 2, 3, 4, 5]), mode=0)
            U_t, S_t, Vt_t = sp.linalg.svd(pipeline_latent_factors, full_matrices=False)
            if save_latent_factors:
                if self.verbose:
                    print("Saving latent factors ...")
                np.save(os.path.join(DEFAULTS, 'error_tensor_U_t.npy'), U_t)
                np.save(os.path.join(DEFAULTS, 'error_tensor_S_t.npy'), S_t)
                np.save(os.path.join(DEFAULTS, 'error_tensor_Vt_t.npy'), Vt_t)
        else:
            if self.verbose:
                print("Loading latent factors from storage ...")       
        
        
        # not yet implemented the part of selecting specific algorithms and hyperparameters
        self.error_matrix = tl.unfold(error_tensor_imputed, mode=0)        
        self.runtime_matrix = tl.unfold(RUNTIME_TENSOR, mode=0)        
        self.Y = Vt_t
        
#         training_index = [eval(configs_matrix[i, 0])['dataset'] for i in range(configs_matrix.shape[0])]
        self.training_index = TRAINING_INDEX
        
        with open(os.path.join(DEFAULTS, 'pipeline_settings.pkl'), 'rb') as handle:
            self.pipeline_settings = pickle.load(handle)
        
         # sampled & fitted models
        self.new_row = new_row or np.full((1, len(self.pipeline_settings)), np.nan)
        self.new_row_pred = new_row or np.full((1, len(self.pipeline_settings)), np.nan)
        self.sampled_indices = set()
        self.sampled_pipelines = [None] * len(self.pipeline_settings)
        self.fitted_indices = set()
        self.fitted_pipelines = [None] * len(self.pipeline_settings)

        # ensemble attributes
        self.build_ensemble = build_ensemble
        self.ensemble_method = ensemble_method
        self.stacking_hyperparams = stacking_hyperparams
        if self.build_ensemble:
            self.ensemble = Ensemble(p_type=self.p_type, algorithm=self.ensemble_method, max_size=ensemble_max_size, hyperparameters=self.stacking_hyperparams, verbose=self.verbose)
        else:
            self.ensemble = Model_collection(self.p_type)
        
        # runtime predictor
        self.runtime_predictor_algorithm = runtime_predictor_algorithm
        self.runtime_predictor = convex_opt.initialize_runtime_predictor(runtime_matrix=self.runtime_matrix, runtimes_index=self.training_index, model_name=self.runtime_predictor_algorithm, verbose=self.verbose, load=True, save=False)
        self.dataset_ratio_threshold = dataset_ratio_threshold
        self.columns_to_avoid = list(set(np.arange(len(self.runtime_predictor.models))) - set(np.where(self.runtime_predictor.models)[0])) # not compatible with the valid variable in self.fit() for now

    def _fit(self, x_train, y_train, categorical, t_predicted, ranks=None, runtime_limit=None, remaining_global=None):
        """This private method is a single round of the doubling process. It fits an AutoLearner object on a new dataset.
        This will sample the performance of several algorithms on the new dataset, predict performance on the rest, then construct an optimal ensemble model.

        Args:
            x_train (np.ndarray):  Features of the training dataset.
            y_train (np.ndarray):  Labels of the training dataset.
            t_predicted (np.ndarray): Predicted running time.
            ranks (int):            Rank of error tensor factorization.
            runtime_limit (float): Maximum time to allocate to AutoLearner fitting.
            remaining_global (float): The remaining runtime 
        """
        if self.verbose:
            print("\nSingle round runtime target: {}".format(runtime_limit))
        
        # set to defaults if not provided
        ranks = ranks or linalg.approx_tensor_rank(self.error_tensor_imputed, threshold=0.01)
        runtime_limit = runtime_limit or self.runtime_limit
        
        
        if self.verbose:
            print('Fitting AutoLearner with maximum runtime {} seconds'.format(runtime_limit))

        # cold-start: pick the initial set of models to fit on the new dataset
        valid = np.where(t_predicted <= self.n_cores * runtime_limit/8)[0]
        Y = self.Y[:ranks[0], valid]
        
        if self.verbose:
            print("Selecting an initial set of models to evaluate ...")
            
        selected_columns_qr, t_sum, case = ED.pivot_columns_time(Y, t_predicted[valid], runtime_limit/8, 
                                                        columns_to_avoid=None,
                                                        rank=Y.shape[0])
        if case == 'greedy_initialization':
            if self.verbose:
                print(case)
            to_sample = valid[selected_columns_qr]

        elif case == 'qr_initialization':        
            to_sample = valid[ED.greedy_stepwise_selection_with_time(Y=Y,
                                                        t=t_predicted[valid],
                                                        initialization=selected_columns_qr,
                                                        t_elapsed=t_sum,
                                                        t_max=runtime_limit/8,
                                                        idx_to_exclude=None)]
        # TODO: check if Y is rank-deficient, i.e. will ED problem fail

        if self.verbose:
            print(t_predicted[to_sample])
        
        if np.isnan(to_sample).any():
            to_sample = np.argsort(t_predicted)[:ranks[0]]
        
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
            candidate_indices = []
            # we only need to fit models on the new dataset if it has not been fitted already
            to_sample = list(set(to_sample) - self.sampled_indices)
            if self.verbose:
                print('Sampling {} entries of new row...'.format(len(to_sample)))
                
#             return to_sample
        
            sampled_pipelines_single_round = [PipelineObject(p_type=self.p_type, config=self.pipeline_settings_on_dataset[i], index=i, verbose=self.verbose) for i in to_sample]
                  
            p1 = mp.Pool(self.n_cores)
            sampled_pipeline_errors_single_round = [p1.apply_async(PipelineObject.kfold_fit_validate, args=[p, x_train, y_train, categorical, self.n_folds, runtime_limit/4, self.random_state]) for p in sampled_pipelines_single_round]
            p1.close()
            p1.join()
            
            if self.verbose:
                print("pool fitting completed")

            # predict performance of models not actually fitted on the new dataset

            for i, error in enumerate(sampled_pipeline_errors_single_round):
                cv_error, cv_predictions, t_elapsed = error.get()
                                
                if not np.isnan(cv_error):
                    sampled_pipelines_single_round[i].cv_error, sampled_pipelines_single_round[i].cv_predictions = cv_error, cv_predictions
                    sampled_pipelines_single_round[i].sampled = True
                    self.new_row[:, to_sample[i]] = cv_error
                    self.sampled_pipelines[to_sample[i]] = sampled_pipelines_single_round[i]
                    self.sampled_indices = self.sampled_indices.union(set([to_sample[i]]))
                    candidate_indices.append(to_sample[i])
                    self._t_predicted[to_sample[i]] = t_elapsed
                else:
                    self._t_predicted[to_sample[i]] = max(t_elapsed, self._t_predicted[to_sample[i]])
                    
            self.new_row_pred = linalg.impute_with_coefficients(self.Y[:ranks[0], :], self.new_row, list(self.sampled_indices))
            
            for idx in np.argsort(self.new_row[0, :])[:5]: # np.argsort automatically put nans at the end of the list
                if not np.isnan(self.new_row[0, idx]):
                    if self.verbose:
                        print(self.sampled_pipelines[idx])
                    self.ensemble.candidate_learners.append(self.sampled_pipelines[idx])

            # impute ALL entries
            # unknown = sorted(list(set(range(self.new_row.shape[1])) - self.sampled_indices))
            # self.new_row[:, unknown] = imputed[:, unknown]

            # k-fold fit candidate learners of ensemble
            remaining = (runtime_limit - (time.time() - start)) * self.n_cores            
            first = (case == 'qr_initialization') and not self.ever_once_selected_best
            
            if remaining > 0 or first:
            # add models predicted to be the best to list of candidate learners
                if self.verbose:
                    if remaining < 0 and first:
                        print("Insufficient time in this doubling round, but we add models predicted to be the best at least once.")
                    print("length of sampled indices: {}".format(len(self.sampled_indices)))
                
    #             self.ensemble.candidate_learners.append(self.sampled_pipelines[best_sampled_idx])
                for i in np.argsort(self.new_row_pred[0]):
                    if (first and len(candidate_indices) <= 3) or t_predicted[i] + t_predicted[candidate_indices].sum() <= remaining / 4:                   
#                         if self.verbose:
#                             print("Adding models predicted to be the best to the ensemble ...")
                        candidate_indices.append(i)
                        # if model has already been k-fold fitted, immediately add to candidate learners
                        if i in self.sampled_indices:
                            assert self.sampled_pipelines[i] is not None
                            self.ensemble.candidate_learners.append(self.sampled_pipelines[i])
                # candidate learners that need to be k-fold fitted
                to_fit = list(set(candidate_indices) - self.sampled_indices)
                if self.verbose:
                    print("{} candidate learners need to be k-fold fitted".format(to_fit))
            else:
                if self.verbose:
                    print("Insufficient time in this doubling round.")
        else:
            remaining = (runtime_limit - (time.time()-start)) * self.n_cores
            to_fit = to_sample.copy()
        
        remaining = (runtime_limit - (time.time() - start)) * self.n_cores
            
        if remaining > 0 or first:         
            if len(to_fit) > 0:
                # fit models predicted to have good performance and thus going to be added to the ensemble
                if self.verbose:
                    print("Fitting {} pipelines predicted to be the best ...".format(len(to_fit)))
                    
                candidate_pipelines = [PipelineObject(p_type=self.p_type, config=self.pipeline_settings_on_dataset[i], index=i, verbose=self.verbose) for i in to_fit]
                
#                 print(remaining_global)
                p2 = mp.Pool(self.n_cores)
                candidate_pipeline_errors = [p2.apply_async(PipelineObject.kfold_fit_validate, args=[p, x_train, y_train, categorical, self.n_folds, remaining_global/2, self.random_state]) for p in candidate_pipelines] # set a not-quite-small limit for promising models
                p2.close()
                p2.join()         

                for i, error in enumerate(candidate_pipeline_errors):
                    cv_error, cv_predictions, t_elapsed = error.get()
                    
                    if not np.isnan(cv_error):
                        candidate_pipelines[i].cv_error, candidate_pipelines[i].cv_predictions = cv_error, cv_predictions
                        candidate_pipelines[i].sampled = True
                        self.new_row[:, to_fit[i]] = cv_error
                        self.sampled_pipelines[to_fit[i]] = candidate_pipelines[i]
                        self.ensemble.candidate_learners.append(candidate_pipelines[i])                    
                        # update sampled indices
                        self.sampled_indices = self.sampled_indices.union(set([to_fit[i]]))
                        self._t_predicted[to_fit[i]] = t_elapsed
                    else:
                        self._t_predicted[to_fit[i]] = max(t_elapsed, self._t_predicted[to_fit[i]])
                        
#                 self.new_row = linalg.impute(self.error_matrix, self.new_row, list(self.sampled_indices), rank=rank)
                self.ever_once_selected_best = True
                
        if len(self.ensemble.candidate_learners) > 0:
            self.ensemble.fitted = True
        
            if self.verbose:
                print('\nFitting ensemble of maximum size {}...'.format(len(self.ensemble.candidate_learners)))
            # ensemble selection and fitting in the remaining time budget
            self.ensemble.fit(x_train, y_train, categorical, remaining, self.sampled_pipelines)
            for pipeline in self.ensemble.base_learners:
                assert pipeline.index is not None
                self.sampled_indices.add(pipeline.index)
                self.sampled_pipelines[pipeline.index] = pipeline        

            if self.verbose:
                print('\nAutoLearner fitting complete.')
        else:
            if self.verbose:
                print("Insufficient time in this round.")

            
    def fit(self, x_train, y_train, categorical=None, verbose=False):

        """Fit an AutoLearner object, iteratively doubling allowed runtime, and terminate when reaching the time limit."""
        
        if categorical is None:
            categorical = np.array([False for _ in range(x_train.shape[1])])

        num_points, num_features = x_train.shape
        
        self.ever_once_selected_best = False
        
        if self.verbose:
            print('\nShape of training dataset: {} data points, {} features'.format(num_points, num_features))
        
        if num_points > 10000 and num_points / num_features > self.dataset_ratio_threshold:
            num_points_new = int(min(5000, num_features * self.dataset_ratio_threshold))
            sampling_ratio = num_points_new / num_points
            if self.verbose:
                print("dataset too skewed in having too many data points; sampling data points with sampling ratio {}".format(sampling_ratio))
            df_x_train = pd.DataFrame(x_train)
            df_y_train = pd.DataFrame(y_train, columns=['labels'])
            df_resampled = df_x_train.join(df_y_train).groupby('labels').apply(pd.DataFrame.sample, frac=sampling_ratio).reset_index(drop=True)
            x_train = df_resampled.drop(['labels'], axis=1).values
            y_train = df_resampled['labels'].values
            if self.verbose:
                print('\nTraining dataset resampled \nShape of resampled training dataset: {} data points, {} features'.format(x_train.shape[0], x_train.shape[1]))

        if self.verbose:
            print("Splitting training set into training and validation ..")
            
        # split data into training and validation sets
#         try:
#             x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
#         except ValueError:
#             x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
        
        # for now, do not do validation, but instead do cross-validation
        x_tr = x_train
        x_va = x_train
        y_tr = y_train
        y_va = y_train
        
        num_points_tr, num_features_tr = x_tr.shape

        self._t_predicted = np.maximum(self._predict_runtime(x_tr), 0.01)
        
        def p2f(x):
            return float(x.strip('%'))/100
        
        # pipeline settings on dataset        
        self.pipeline_settings_on_dataset = []
        
        for item in self.pipeline_settings:
            item_copy = copy.deepcopy(item)
#             if item_copy['dim_reducer']['algorithm'] == 'PCA':
#                 item_copy['dim_reducer']['hyperparameters']['n_components'] = int(min((self.n_folds - 1) * num_points_tr/self.n_folds, p2f(item_copy['dim_reducer']['hyperparameters']['n_components']) * num_features_tr))
            if item_copy['dim_reducer']['algorithm'] == 'SelectKBest':
                item_copy['dim_reducer']['hyperparameters']['k'] = int(p2f(item_copy['dim_reducer']['hyperparameters']['k']) * num_features_tr)
                
            self.pipeline_settings_on_dataset.append(item_copy)
        
        
        start = time.time()

#         if num_points / num_features < 2:
#             if self.verbose:
#                 print("dataset too skewed in having too many features ...")
#             self._greedy_initial_selection(x_tr, y_tr, self._t_predicted, self.runtime_limit/4)
        
        
        ranks = [linalg.approx_tensor_rank(self.error_tensor_imputed, threshold=0.05)]
#         if self.build_ensemble:
#             t_init = 2**np.floor(np.log2(np.sort(self._t_predicted)[:int(5*ranks[0][-1])].sum()))
#             t_init = max(1, t_init)
#         else:
#             t_init = self.runtime_limit / 8

        t_init = 2**np.floor(np.log2(np.sort(self._t_predicted)[:int(5*ranks[0][-1])].sum()))
        t_init = max(2**np.floor(np.log2(self.runtime_limit/8)), t_init)
            
        if self.verbose:
            print("Runtime limit of initial round: {}".format(t_init))
        times = [t_init]
        losses = [0.5]
        
#         return

        e_hat, e_hat_pred, actual_times, sampled, ensembles = [], [], [], [], []
        
        if self.verbose:
            print("Doubling process started ...")
        def doubling():
            k, t = ranks[0], times[0]
            counter, self.best = 0, 0            
            while time.time() - start < self.runtime_limit - t:
                if verbose:
                    print('Fitting with ranks={}, t={}'.format(k, t))
#                 if self.build_ensemble:
#                     self.ensemble = Ensemble(self.p_type, self.ensemble_method, self.stacking_hyperparams)
#                 else:
#                     self.ensemble = Model_collection(self.p_type)
                self._fit(x_tr, y_tr, categorical, self._t_predicted, ranks=k, runtime_limit=t, remaining_global=self.runtime_limit-time.time()+start)
                if self.build_ensemble and self.ensemble.fitted:
                    if self.verbose:
                        print("\nGot a new ensemble in the round with runtime target {} seconds".format(t))
#                     loss = util.error(y_va, self.ensemble.predict(x_va), self.p_type)
                    loss = self.ensemble.kfold_fit_validate(x_va, y_va, categorical, n_folds=3, timeout=(self.runtime_limit-time.time()+start)/2)[0]

                    # TEMPORARY: Record intermediate results

                    e_hat.append(np.copy(self.new_row))
                    e_hat_pred.append(np.copy(self.new_row_pred))
                    actual_times.append(time.time() - start)
                    sampled.append(self.sampled_indices)
                    ensembles.append(self.ensemble)
                    losses.append(loss)

                    if loss == min(losses):
#                         ranks.append((k[0], k[1], k[2], k[3], k[4], k[5]+1))
                        ranks.append((k[0], k[1], k[2], k[3], k[4]+1))
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
                self.predict_with_most_common_class = False

                return {'ranks': ranks[:-1], 'runtime_limits': times[:-1], 'validation_loss': losses,
                    'filled_new_row': e_hat, 'predicted_new_row': e_hat_pred, 'actual_runtimes': actual_times, 'sampled_indices': sampled, 'models': ensembles}
            except IndexError:
                self.y_train_mode = mode(y_train.flatten())[0][0]
                self.predict_with_most_common_class = True
                print("No ensemble built within time limit. The auto-learner is set to predict by the mode of class labels in the training set. Please try increasing the time limit or allocate more computational resources.")
        else:
            return

    def refit(self, x_train, y_train, categorical=None):
        """Refit an existing AutoLearner object on a new dataset. This will simply retrain the base-learners and
        stacked learner of an existing model, and so algorithm and hyperparameter selection may not be optimal.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        if categorical is None:
            categorical = np.array([False for _ in range(x_train.shape[1])])
        assert self.ensemble.fitted, "Cannot refit unless model has been fit."
        self.ensemble.refit(x_train, y_train, categorical)

    def predict(self, x_test):
        """Generate predictions on test data.

        Args:
            x_test (np.ndarray): Features of the test dataset.
        Returns:
            np.ndarray: Predicted labels.
        """
        if self.build_ensemble:        
            if self.predict_with_most_common_class:
                return np.full(x_test.shape[0], self.y_train_mode)
            else:
                return self.ensemble.predict(x_test)
        else:
            return 

    def get_models(self):
        """Get details of the selected machine learning models and the ensemble.
        """
        return self.ensemble.get_models()

    def get_model_accuracy(self, y_test):
        """ Get accuracies of selected models.
        """
        return self.ensemble.get_model_accuracy(y_test)    
    
    def _predict_runtime(self, x_train):
        # predict runtime for the training set of the new dataset.
        if self.verbose:
            print("Predicting pipeline running time ..")
        return convex_opt.predict_runtime(x_train.shape, saved_model='Class', model=self.runtime_predictor)
    
    def _greedy_initial_selection(self, x_train, y_train, t_predicted, runtime_limit):
        """
        Not yet updated to incorporate the categorical list.
        """
        if self.verbose:
            print("Fitting fast pipelines that perform well on average.")

        candidate_pipeline_indices = list(set(np.where(t_predicted <= runtime_limit / 8)[0]).intersection(
            set(np.argsort(np.nanmedian(tl.unfold(self.error_tensor_imputed, mode=0), axis=0))[
                :int(len(self.pipeline_settings_on_dataset)/50)])))


        candidate_pipelines = [PipelineObject(p_type=self.p_type, config=self.pipeline_settings_on_dataset[i], index=i, verbose=self.verbose) for i in candidate_pipeline_indices]

        p1 = mp.Pool(self.n_cores)
        candidate_pipeline_errors = [p1.apply_async(PipelineObject.kfold_fit_validate, args=[p, x_train, y_train, categorical, self.n_folds, runtime_limit/4, self.random_state]) for p in candidate_pipelines]
        p1.close()
        p1.join()

        if self.verbose:
            print("Initial greedy fitting completed")


        for i, error in enumerate(candidate_pipeline_errors):
            cv_error, cv_predictions, t_elapsed = error.get()
            if not np.isnan(cv_error):
                candidate_pipelines[i].cv_error, candidate_pipelines[i].cv_predictions = cv_error, cv_predictions
                candidate_pipelines[i].sampled = True
                self.new_row[:, candidate_pipeline_indices[i]] = cv_error
                self.sampled_pipelines[candidate_pipeline_indices[i]] = candidate_pipelines[i]
                self.ensemble.candidate_learners.append(candidate_pipelines[i])                    
                # update sampled indices
                self.sampled_indices = self.sampled_indices.union(set([candidate_pipeline_indices[i]]))
                self._t_predicted[candidate_pipeline_indices[i]] = t_elapsed
            else:
                self._t_predicted[candidate_pipeline_indices[i]] = max(t_elapsed, self._t_predicted[candidate_pipeline_indices[i]])

        if len(self.ensemble.candidate_learners) > 0:
            self.ensemble.fitted = True        
            self.ensemble.fit(x_train, y_train)
        else:
            if self.verbose:
                print("Insufficient time to fit fast and on average best-performing pipelines.")

