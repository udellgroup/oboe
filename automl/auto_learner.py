"""
Automatically tuned scikit-learn model.
"""

import numpy as np
import multiprocessing as mp
import pandas as pd
import pkg_resources
import subprocess
import linalg
import util
import convex_opt_c
import convex_opt_s
import openml
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
    def __init__(self, p_type, algorithms=None, hyperparameters=None, n_cores=None, verbose=False, selection_method='qr', runtime_limit=None, cvxopt_package='cvxpy', transform_error_matrix=False, bayes_opt=False, debug_mode=True, stacking_alg='Logit', **stacking_hyperparams):

        assert selection_method in ['qr', 'min_variance'], "The method to select entries to actually \
        compute must be either qr (QR decomposition) or min_variance (minimize variance with time constraints)."

        # check if arguments to constructor are valid; set to defaults if not specified
        default, new = util.check_arguments(p_type, algorithms, hyperparameters)
        self.p_type = p_type.lower()
        self.algorithms = algorithms
        self.hyperparameters = hyperparameters
        self.n_cores = n_cores
        self.verbose = verbose
        self.selection_method = selection_method
        self.runtime_limit = runtime_limit
        self.transform_error_matrix = transform_error_matrix
        self.n_cores = n_cores
        self.cvxopt_package = cvxopt_package

#        if len(new) > 0:
#            # if selected hyperparameters contain model configurations not included in default
#            proceed = input("Your selected hyperparameters contain some not included in the default error matrix. \n"
#                            "Do you want to generate your own error matrix? [yes/no]")
#            if proceed == 'yes':
#                subprocess.call(['./generate_matrix.sh'])
#                # TODO: load newly generated error matrix file
#            else:
#                return
#        else:
        if True:
            # use default error matrix (or subset of)
            error_matrix_path = pkg_resources.resource_filename(__name__, 'defaults/error_matrix.csv')
            runtime_matrix_path = pkg_resources.resource_filename(__name__, 'defaults/runtime_matrix.csv')
            default_error_matrix = pd.read_csv(error_matrix_path, index_col=0)
            default_runtime_matrix = pd.read_csv(runtime_matrix_path, index_col=0)
            assert set(default_error_matrix.index.tolist()) == set(default_runtime_matrix.index.tolist()), "Indices of error and runtime matrices must match."
            column_headings = np.array([eval(heading) for heading in list(default_error_matrix)])
            selected_indices = np.full(len(column_headings), True)
#            selected_indices = np.array([heading in column_headings for heading in default])
            self.error_matrix = default_error_matrix.values[:, selected_indices]
            self.error_index = default_error_matrix.index.tolist()
            self.runtime_index = default_runtime_matrix.index.tolist()
            self.runtime_matrix = default_runtime_matrix.values[:, selected_indices]
            if self.cvxopt_package == 'cvxpy':
                self.dataset_sizes = convex_opt_c.get_dataset_sizes(default_error_matrix)
#            self.column_headings = sorted(default, key=lambda d: d['algorithm'])
            self.column_headings = column_headings
            self.bayes_opt = bayes_opt
            self.debug_mode = debug_mode

        self.ensemble = Ensemble(self.p_type, stacking_alg, **stacking_hyperparams)
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
            if self.cvxopt_package == 'cvxpy':
                log_runtime_matrix = np.log(self.runtime_matrix)
                log_runtime_predict = convex_opt_c.runtime_prediction_via_poly_fitting(self.dataset_sizes, 3, log_runtime_matrix, x_train, self.runtime_index)
                runtime_predict = np.exp(log_runtime_predict)
                if self.transform_error_matrix:
                    error_matrix_transformed = convex_opt_c.inv_sigmoid(convex_opt.truncate(self.error_matrix))
                    known_indices = convex_opt_c.min_variance_model_selection(self.runtime_limit, runtime_predict, error_matrix_transformed, n_cores=self.n_cores)
                else:
                    known_indices = convex_opt_c.min_variance_model_selection(self.runtime_limit, runtime_predict, self.error_matrix, n_cores=self.n_cores)
            elif self.cvxopt_package == 'scipy':
                runtime_predict = convex_opt_s.predict_runtime(x_train.shape)
                X,Y,Vt = linalg.pca(self.error_matrix, threshold=0.03)
                v_opt_x = convex_opt_s.solve(runtime_predict, self.runtime_limit, Y)
                known_indices = np.where(v_opt_x>0.8)[0]

        if self.debug_mode:
            self.num_known_indices = len(known_indices)
            pivot_columns_one_percent = linalg.pivot_columns(self.error_matrix, threshold=0.01)
            self.num_pivots_one_percent = len(pivot_columns_one_percent)
            self.num_overlap_with_pivots = len(set(known_indices).intersection(set(pivot_columns_one_percent)))
        
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
            # TODO: add predictions to second layer matrix?
        self.new_row = linalg.impute(self.error_matrix, self.new_row, known_indices)

        # Add new row to error matrix at the end (might be incorrect?)
        # self.error_matrix = np.vstack((self.error_matrix, self.new_row))

        # TODO: Fit ensemble candidates (?)
        
        n_models = 3

        bayesian_opt_models = [Model(self.p_type, self.column_headings[i]['algorithm'],
                                     self.column_headings[i]['hyperparameters'], verbose=self.verbose)
                               for i in np.argsort(self.new_row.flatten())[:n_models]]
        if self.bayes_opt:
            if self.verbose:
                print('\nConducting Bayesian optimization...')
            pool2 = Pool(self.n_cores)
            optimized_hyperparams = pool2.map(Model.bayesian_optimize, bayesian_opt_models)
            pool2.close()
            pool2.join()
        else:
            optimized_hyperparams = [m.hyperparameters for m in bayesian_opt_models]

        for i, params in enumerate(optimized_hyperparams):
            bayesian_opt_models[i].hyperparameters = params
            self.ensemble.add_base_learner(bayesian_opt_models[i])
            self.optimized_settings.append({'algorithm': bayesian_opt_models[i].algorithm,
                                            'hyperparameters': bayesian_opt_models[i].hyperparameters})

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

