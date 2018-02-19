"""
Parent class for all ML models.
"""

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
        fitted (bool): Whether or not the model has been trained.
        verbose (bool): Whether or not to generate print statements when fitting complete.
    """

    def __init__(self, p_type, algorithm, hyperparameters={}, verbose=False):
        self.p_type = p_type
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.model = self.instantiate()
        self.fitted = False
        self.verbose = verbose

    def instantiate(self):
        """Creates a scikit-learn object of specified algorithm type and with specified hyperparameters.

        Returns:
            object: A scikit-learn object.
        """
        # TODO: is random state necessary (?)
        try:
            return getattr(util, self.algorithm)(random_state=0, **self.hyperparameters)
        except ValueError:
            return getattr(util, self.algorithm)(**self.hyperparameters)

    def fit(self, x_train, y_train):
        """Fits the model on training data. Note that this function is only used once a model has been identified as a
        model to be included in the final ensemble.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
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

    def kfold_fit_validate(self, x_train, y_train, n_folds):
        """Performs k-fold cross validation on a training dataset. Note that this is the function used to fill entries
        of the error matrix.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
            n_folds (int): Number of folds to use for cross validation.

        Returns:
            float: Mean of k-fold cross validation error.
            np.ndarray: Predictions on the training dataset from cross validation.
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

        if self.verbose:
            print("{} {} complete.".format(self.algorithm, self.hyperparameters))

        return cv_errors, y_predicted

    def bayesian_optimize(self):
        """Conducts Bayesian optimization of hyperparameters.
        """
        # TODO: implement Bayesian optimization
        return self.hyperparameters

    def error(self, y_observed, y_predicted):
        """Compute error metric for the model.

        Args:
            y_observed (np.ndarray): Observed labels.
            y_predicted (np.ndarray): Predicted labels.

        Returns:
            float: Error metric
        """
        return util.error(y_observed, y_predicted, self.p_type)


class Ensemble(Model):
    """An object representing an ensemble of machine learning models.

    Attributes:
        p_type (str): Either 'classification' or 'regression'.
        algorithm (str): Algorithm type (e.g. 'Logit').
        hyperparameters (dict): Hyperparameters (e.g. {'C': 1.0}).
        model (object): A scikit-learn object for the model.
    """

    def __init__(self, p_type, algorithm, hyperparameters={}):
        super().__init__(p_type, algorithm, hyperparameters)
        self.base_learners = []

    def add_base_learner(self, model):
        """Add weak learner to ensemble.

        Args:
            model (Model): Model object to be added to the ensemble.
        """
        self.base_learners.append(model)

    def fit(self, x_train, y_train):
        """Fit ensemble model on training data.

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
        """
        assert len(self.base_learners) > 0, "Ensemble size must be greater than zero."

        base_learner_predictions = ()
        for model in self.base_learners:
            _, y_predicted = model.kfold_fit_validate(x_train, y_train, n_folds=3)
            base_learner_predictions += (np.reshape(y_predicted, [-1, 1]), )
            model.fit(x_train, y_train)

        x_tr = np.hstack(base_learner_predictions)
        self.model.fit(x_tr, y_train)
        self.fitted = True

    def predict(self, x_test):
        """Generate predictions of the ensemble model on test data.

        Args:
            x_test (np.ndarray): Features of the test dataset.

        Returns:
            np.array: Predicted features of the test dataset.
        """
        assert len(self.base_learners) > 0, "Ensemble size must be greater than zero."

        base_learner_predictions = ()
        for model in self.base_learners:
            y_predicted = np.reshape(model.predict(x_test), [-1, 1])
            base_learner_predictions += (y_predicted, )

        x_te = np.hstack(base_learner_predictions)
        return self.model.predict(x_te)
