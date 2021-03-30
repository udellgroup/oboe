"""
Parent class for all ML models.
"""

import numpy as np
from . import util
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, train_test_split


RANDOM_STATE = 0


class Model:
    """An object representing a machine learning model.

    Attributes:
        p_type (str):           Either 'classification' or 'regression'.
        algorithm (str):        Algorithm type (e.g. 'KNN').
        hyperparameters (dict): Hyperparameters (e.g. {'n_neighbors': 5}).
        model (object):         A scikit-learn object for the model.
        fitted (bool):          Whether or not the model has been trained.
        verbose (bool):         Whether or not to generate print statements when fitting complete.
    """

    def __init__(self, p_type, algorithm, hyperparameters={}, verbose=False, index=None):
        self.p_type = p_type
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.model = self.instantiate()
        self.cv_error = np.nan
        self.cv_predictions = None
        self.sampled = False
        self.fitted = False
        self.verbose = verbose
        self.index = index

    def instantiate(self):
        """Creates a scikit-learn object of specified algorithm type and with specified hyperparameters.

        Returns:
            object: A scikit-learn object.
        """
        if self.algorithm == None or self.algorithm.lower() == 'greedy' or self.algorithm == 'best_several':
            return None
        try:
            return getattr(util, self.algorithm)(random_state=RANDOM_STATE, **self.hyperparameters)
        except TypeError:
            return getattr(util, self.algorithm)(**self.hyperparameters)

    def fit(self, x_train, y_train, runtime_limit=None):
        """Fits the model on training data. Note that this function is only used once a model has been identified as a
        model to be included in the final ensemble.

        Args:
            x_train (np.ndarray):   Features of the training dataset.
            y_train (np.ndarray):   Labels of the training dataset.
            runtime_limit (float):  Maximum amount of time to allocate to fitting.
        """
        self.model.fit(x_train, y_train)
        self.fitted = True
        if self.verbose:
            print("{} {} complete.".format(self.algorithm, self.hyperparameters))

    def predict(self, x_test):
        """Predicts labels on a new dataset.

        Args:
            x_test (np.ndarray): Features of the test dataset.

        Returns:
            np.array: Predicted features of the test dataset.
        """
        return self.model.predict(x_test)

    def kfold_fit_validate(self, x_train, y_train, n_folds, random_state=None):
        """Performs k-fold cross validation on a training dataset. Note that this is the function used to fill entries
        of the error matrix.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
            n_folds (int):        Number of folds to use for cross validation.

        Returns:
            float: Mean of k-fold cross validation error.
            np.ndarray: Predictions on the training dataset from cross validation.
        """
        y_predicted = np.empty(y_train.shape)
        cv_errors = np.empty(n_folds)
        kf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)

        for i, (train_idx, test_idx) in enumerate(kf.split(x_train, y_train)):
            x_tr = x_train[train_idx, :]
            y_tr = y_train[train_idx]
            x_te = x_train[test_idx, :]
            y_te = y_train[test_idx]

            model = self.instantiate()
            if len(np.unique(y_tr)) > 1:
                model.fit(x_tr, y_tr)
                y_predicted[test_idx] = model.predict(x_te)
            else:
                y_predicted[test_idx] = y_tr[0]
            cv_errors[i] = self.error(y_te, y_predicted[test_idx])

        self.cv_error = cv_errors.mean()
        self.cv_predictions = y_predicted
        self.sampled = True
        if self.verbose:
            print("{} {} complete.".format(self.algorithm, self.hyperparameters))

        return cv_errors, y_predicted

    def kfold_fit_validate_testing(self, x_train, y_train, n_folds, random_state=None):
        """Performs k-fold cross validation on a training dataset, with fitting on a portion of the training fold and testing on the test fold. 
        
        Args:
        x_train (np.ndarray): Features of the training dataset.
        y_train (np.ndarray): Labels of the training dataset.
        n_folds (int):        Number of folds to use for cross validation.
        
        Returns:
        float: Mean of k-fold cross validation error.
        np.ndarray: Predictions on the training dataset from cross validation.
        """
        y_predicted = np.empty(y_train.shape)
        cv_errors = np.empty(n_folds)
        kf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)
            
        for i, (train_idx, test_idx) in enumerate(kf.split(x_train, y_train)):
            x_tr_val = x_train[train_idx, :]
            y_tr_val = y_train[train_idx]
            x_te = x_train[test_idx, :]
            y_te = y_train[test_idx]
            # split data into training and validation sets
            try:
                x_tr, x_va, y_tr, y_va = train_test_split(x_tr_val, y_tr_val, test_size=0.15, stratify=y_tr_val, random_state=random_state)
            except ValueError:
                x_tr, x_va, y_tr, y_va = train_test_split(x_tr_val, y_tr_val, test_size=0.15, random_state=random_state)
                
            model = self.instantiate()
            if len(np.unique(y_tr)) > 1:
                model.fit(x_tr, y_tr)
                y_predicted[test_idx] = model.predict(x_te)
            else:
                y_predicted[test_idx] = y_tr[0]
            cv_errors[i] = self.error(y_te, y_predicted[test_idx])
                
        self.cv_error = cv_errors.mean()
        self.cv_predictions = y_predicted
        self.sampled = True
        if self.verbose:
            print("{} {} complete.".format(self.algorithm, self.hyperparameters))

        return cv_errors, y_predicted


    def error(self, y_true, y_predicted):
        """Compute error metric for the model.

        Args:
            y_true (np.ndarray):      Observed labels.
            y_predicted (np.ndarray): Predicted labels.
        Returns:
            float: Error metric
        """
        return util.error(y_true, y_predicted, self.p_type)


