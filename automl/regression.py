"""
Regression algorithms.
"""

from model import Model
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


class Regressor(Model):
    """An object representing a classification algorithm.

    """

    def __init__(self, algorithm, hyperparameters):
        super().__init__('classification', algorithm, hyperparameters)

    def error(self, y_predicted, y_observed):
        """Compute mean squared error.

        Args:
            y_predicted (np.ndarray): Predicted target values.
            y_observed (np.ndarray): Observed target values.

        Returns:
            float: Mean squared error.
        """
        return mean_squared_error(y_observed, y_predicted)