"""
Parent class for all ML models.
"""

import numpy as np
from .util import error
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, train_test_split
from .model import Model
from sklearn.pipeline import Pipeline
import time

# timeout module
import signal
from contextlib import contextmanager

# data cleaning
from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA

# encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# standardizer
from sklearn.preprocessing import StandardScaler

# dimensionality reducer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2

# classifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier as ExtraTrees
from sklearn.ensemble import GradientBoostingClassifier as GBT
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.svm import LinearSVC as lSVM
from sklearn.svm import SVC as kSVM
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLP


RANDOM_STATE = 0

class PipelineObject:
    """An object representing a machine learning pipeline.

    Attributes:
        p_type (str):           Either 'classification' or 'regression'.
        model (object):         A scikit-learn pipeline object.
        fitted (bool):          Whether or not the model has been trained.
        verbose (bool):         Whether or not to generate print statements when fitting complete.
    """

    def __init__(self, config, index=None, p_type='classification', verbose=False):
        self.p_type = p_type
        
        # this ensures the order of pipeline steps is correct
        self.pipeline_steps = ['imputer', 'encoder', 'standardizer', 'dim_reducer', 'estimator']
        assert set(config.keys()) == set(self.pipeline_steps), "Pipeline steps not correct!"
        
        self.index = index
        self.config = config
#         self.model = self._instantiate()
        self.cv_error = np.nan
        self.cv_predictions = None
        self.sampled = False
        self.fitted = False
        self.verbose = verbose

    def _instantiate(self, categorical, columns_to_keep):
        """
        Creates a scikit-learn object of the corresponding pipeline.
        """
        categorical_columns = np.where(categorical)[0]
        pipeline = []
        for step in self.pipeline_steps:
            alg = self.config[step]['algorithm']
            if alg is not None:
                hyperparameters = self.config[step]['hyperparameters']
                if step == 'encoder' and alg == 'OneHotEncoder':
                    step_object = ColumnTransformer(
                        [('one_hot_encoder', OneHotEncoder(**hyperparameters), categorical_columns)], 
                        remainder='passthrough', sparse_threshold=0)
                elif step == 'dim_reducer' and alg == 'SelectFromModel':
                    step_object = eval(alg)(**hyperparameters)
                else:                    
                    step_object = eval(alg)(**hyperparameters)
                    if step == 'imputer' and hyperparameters['strategy'] != 'constant':
                        categorical = np.array(categorical)[columns_to_keep]
                        categorical_columns = np.where(categorical)[0]
            else:
                step_object = None
            
            pipeline.append((step, step_object))
        
        pipeline = Pipeline(pipeline)
        return pipeline

    def fit(self, x_train, y_train, categorical=None):
        """Fits the pipeline on training data.

        Args:
            x_train (np.ndarray):   Features of the training dataset.
            y_train (np.ndarray):   Labels of the training dataset.
        """
        columns_to_keep = np.where(np.invert(np.all(np.isnan(x_train), axis=0)))[0]
        if categorical is None:
            categorical = np.array([False for _ in range(x_train.shape[1])])
        self.model = self._instantiate(categorical, columns_to_keep)
        self.model.fit(x_train, y_train)
        self.fitted = True
        if self.verbose:
            print("Pipeline fitting completed.")

    def predict(self, x_test):
        """Predicts labels on a new dataset.

        Args:
            x_test (np.ndarray): Features of the test dataset.

        Returns:
            np.array: Predicted features of the test dataset.
        """
        assert self.fitted, "Pipeline not fitted!"
        return self.model.predict(x_test)

    def kfold_fit_validate(self, x_train, y_train, categorical=None, n_folds=5, timeout=1e5, random_state=None):
        """Performs k-fold cross validation on a training dataset. 
        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
            n_folds (int):        Number of folds to use for cross validation.
        Returns:
            float: Mean of k-fold cross validation error.
            np.ndarray: Predictions on the training dataset from cross validation.
        """
        timeout = int(max(1, timeout))
        
        if timeout < 1e5:
            if self.verbose:
                print("having a capped running time of {} seconds".format(timeout))
        if categorical is None:
            categorical = [False for _ in range(x_train.shape[1])]
        y_predicted = np.empty(y_train.shape)
        cv_errors = np.empty(n_folds)
        kf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)
        
        start = time.time()
        
#         for pipeline in self.base_learners:            
            
        def _kfold_fit_validate():            
            for i, (train_idx, test_idx) in enumerate(kf.split(x_train, y_train)):
                x_tr = x_train[train_idx, :]
                y_tr = y_train[train_idx]
                x_te = x_train[test_idx, :]
                y_te = y_train[test_idx]

                columns_to_keep = np.where(np.invert(np.all(np.isnan(x_tr), axis=0)))[0]
                model = self._instantiate(categorical, columns_to_keep)
                if len(np.unique(y_tr)) > 1:
                    model.fit(x_tr, y_tr)
                    y_predicted[test_idx] = model.predict(x_te)
                else:
                    y_predicted[test_idx] = y_tr[0]
                cv_errors[i] = error(y_te, y_predicted[test_idx], p_type=self.p_type)

            self.cv_error = cv_errors.mean()
            self.cv_predictions = y_predicted
            self.sampled = True          

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
            with time_limit(timeout):
                _kfold_fit_validate()
        except TimeoutException:
            self.cv_error = -1
            self.cv_predictions = None
            self.sampled = False
            if self.verbose:
                print("Pipeline fitting time limit reached.")        
        except ValueError:
            self.cv_error = -2
            self.cv_predictions = None
            self.sampled = False
            if self.verbose:
                print("Not a valid pipeline.")            
             
        t_elapsed = time.time() - start
        return self.cv_error, self.cv_predictions, t_elapsed

    












