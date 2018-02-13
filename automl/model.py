"""
Parent class for all ML models.
"""

from abc import abstractclassmethod
import numpy as np
import classification
import regression
from sklearn.model_selection import KFold

RANDOM_STATE = 0


class Model:
    """An object representing a machine learning model.

    Attributes:
        type (str): Either 'classification' or 'regression'.
        algorithm (str): Algorithm type (e.g. 'KNN').
        hyperparameters (dict): Hyperparameters (e.g. {'k': 5}).
        model (object): A scikit-learn object for the model.
    """

    def __init__(self, type, algorithm, hyperparameters):
        self.type = type
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.model = self.instantiate()

    def instantiate(self):
        """Creates a scikit-learn object of specified algorithm type and with specified hyperparameters.

        Returns:
            object: A scikit-learn object.
        """
        return getattr(eval(self.type), self.algorithm)(**self.hyperparameters)

    def fit(self, x_train, y_train):
        """Fits the model on training data. Note that this function is only used once a model has been identified as a
        model to be included in the final ensemble.

        Args:
            x_train (np.ndarray): Features of training dataset.
            y_train (np.ndarray): Labels of training dataset.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """Predicts labels on a new dataset.

        Args:
            x_test (np.ndarray): Features of test dataset.
        """
        return self.model.predict(x_test)

    def kfold_fit_validate(self, x_train, y_train, n_folds):
        """Performs k-fold cross validation on a training dataset. Note that this is the function used to fill entries
        of the error matrix.

        Args:
            x_train (np.ndarray): Features of training dataset.
            y_train (np.ndarray): Labels of training dataset.
            n_folds (int): Number of folds to use for cross validation.

        Returns:
            float: Mean of k-fold cross validation error.
            np.ndarray: Predictions on training dataset from cross validation.
        """
        y_predicted = np.empty(y_train.shape)
        cv_errors = np.empty(n_folds)
        kf = KFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

        for i, (train_idx, test_idx) in enumerate(kf.split(x_train)):
            x_tr = x_train[train_idx, :]
            y_tr = y_train[train_idx]
            x_te = x_train[test_idx, :]
            y_te = y_train[test_idx]

            model = self.instantiate()
            model.fit(x_tr, y_tr)
            y_predicted[test_idx] = model.predict(x_te)
            cv_errors[i] = self.error(y_predicted[test_idx], y_te)

        return cv_errors.mean(), y_predicted

    @abstractclassmethod
    def error(self, y_predicted, y_observed):
        """Compute error metric for the model; varies based on classification/regression and algorithm type.
        This method must be overwritten in the child class.

        Args:
            y_predicted (np.ndarray): Predicted labels.
            y_observed (np.ndarray): Observed labels.

        Returns:
            float: Error metric.
        """
        pass
