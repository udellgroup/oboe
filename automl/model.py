"""
Parent class for all ML models.
"""

import auto_learner
import numpy as np
import util
from sklearn.model_selection import KFold


RANDOM_STATE = 0


class Model:
    """An object representing a machine learning model.

    Attributes:
        type (str): Either 'classification' or 'regression'.
        algorithm (str): Algorithm type (e.g. 'KNN').
        hyperparameters (dict): Hyperparameters (e.g. {'n_neighbors': 5}).
        model (object): A scikit-learn object for the model.
        verbose (bool): Whether or not to generate print statements when fitting complete.
    """

    def __init__(self, type, algorithm, hyperparameters, verbose=False):
        self.type = type
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.model = self.instantiate()
        self.verbose = verbose

    def instantiate(self):
        """Creates a scikit-learn object of specified algorithm type and with specified hyperparameters.

        Returns:
            object: A scikit-learn object.
        """
        return getattr(auto_learner, self.algorithm)(**self.hyperparameters)

    def fit(self, x_train, y_train):
        """Fits the model on training data. Note that this function is only used once a model has been identified as a
        model to be included in the final ensemble.

        Args:
            x_train (np.ndarray): Features of training dataset.
            y_train (np.ndarray): Labels of training dataset.
        """
        self.model.fit(x_train, y_train)
        if self.verbose:
            print("{} ({}) complete. Data = {}".format(self.algorithm, self.hyperparameters, x_train.shape))

    def predict(self, x_test):
        """Predicts labels on a new dataset.

        Args:
            x_test (np.ndarray): Features of test dataset.

        Returns:
            np.array: Predicted features of test dataset.
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

    def bayesian_optimize(self):
        """Conducts Bayesian optimization of hyperparameters.
        """
        # TODO: implement Bayesian optimization
        pass

    def error(self, y_observed, y_predicted):
        """Compute error metric for the model.

        Args:
            y_observed (np.ndarray): Observed labels.
            y_predicted (np.ndarray): Predicted labels.

        Returns:
            float: Error metric
        """
        return util.error(y_observed, y_predicted, self.type)


class Ensemble(Model):
    """An object representing an ensemble of machine learning models.

    Attributes:
        type (str): Either 'classification' or 'regression'.
        algorithm (str): Algorithm type (e.g. 'Logit').
        hyperparameters (dict): Hyperparameters (e.g. {'C': 1.0}).
        model (object): A scikit-learn object for the model.
    """

    def __init__(self, type, algorithm, hyperparameters):
        super().__init__(type, algorithm, hyperparameters)
        self.base_learners = []

    def predict(self, x_test):
        """Fits the ensemble model on training data.

        Args:
            x_test (np.ndarray): Features of test dataset.

        Returns:
            np.array: Predicted features of test dataset.
        """

        assert len(self.base_learners) > 0, "Ensemble size must be greater than zero."

        base_learner_predictions = ()
        for model in self.base_learners:
            y_predicted = np.reshape(model.predict(x_test), [-1, 1])
            base_learner_predictions += (y_predicted, )

        x_te = np.hstack(base_learner_predictions)
        return self.model.predict(x_te)
