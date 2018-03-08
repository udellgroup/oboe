"""
Find columns of error matrix to minimize variance of predicted latent features.
Solves convex optimization problem as described in chapter 7.5 in https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
"""

import numpy as np
import os
import pandas as pd
import pickle
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def solve(t_predicted, t_max, Y):
    """Solve the following optimization problem:
    minimize -log(det(sum_i v[i]*Y[:, i]*Y[:, i].T)) subject to 0 <= v[i] <= 1 and t_predicted.T * v <= t_max
    The optimal vector v is an approximation of a boolean vector indicating which entries to sample.

    Args:
         t_predicted (np.ndarray): 1-d array specifying predicted runtime for each model setting
         t_max (float):            maximum runtime of sampled model
         Y (np.ndarray):           matrix representing latent variable weights of error matrix
    Returns:
        np.ndarray:                optimal vector v (not truncated to binary values)
    """
    def objective(v):
        sign, log_det = np.linalg.slogdet(Y @ np.diag(v) @ Y.T)
        return -1 * sign * log_det

    def constraint(v):
        return t_max - t_predicted @ v

    n = len(t_predicted)
    v0 = np.full((n, ), 0.5)
    constraints = {'type': 'ineq', 'fun': constraint}
    v_opt = minimize(objective, v0, method='SLSQP', bounds=[(0, 1)] * n, constraints=constraints)
    return v_opt


def predict_runtime(size, saved_model=None):
    """Predict the runtime for each model setting on a dataset with given shape.

    Args:
        size (np.ndarray): 1-d array specifying dataset size as [n_rows, n_columns]
        saved_model (str): path to pre-trained model; defaults to None
    Returns:
        np.ndarray:        1-d array of predicted runtimes
    """
    assert len(size) == 2, "Dataset must be 2-dimensional."
    shape = np.array(size)

    if saved_model:
        with open(saved_model, 'rb') as file:
            model = pickle.load(file)
        return model.predict(shape)

    defaults_path = os.path.join(os.path.realpath(__file__), 'defaults')
    x_train = pd.read_csv(os.path.join(defaults_path, 'dataset_sizes.csv'))
    y_train = pd.read_csv(os.path.join(defaults_path, 'runtime_matrix.csv'))
    model = RuntimePredictor(3, x_train, y_train)
    with open(os.path.join(defaults_path, 'runtime_predictor.pkl'), 'wb') as file:
        pickle.dump(model, file)

    return model.predict(shape)


class RuntimePredictor:
    """Model that predicts the runtime for each model setting on a dataset with given shape. Performs polynomial
    regression on n (# samples), p (# features), and log(n).

    Attributes:
        degree (int):   degree of polynomial basis function
        n_models (int): number of model settings
        models (list):  list of scikit-learn LinearRegression models
    """
    def __init__(self, degree, sizes, runtimes):
        self.degree = degree
        self.n_models = runtimes.shape[1]
        self.models = (None, ) * self.n_models
        self.fit(sizes, runtimes)

    def fit(self, sizes, runtimes):
        """Fit polynomial regression on pre-recorded runtimes on datasets."""
        assert sizes.shape[0] == runtimes.shape[0], "Dataset sizes and runtimes must be recorded on same datasets."
        x_raw = np.concatenate((sizes, np.log(sizes[:, 0]).reshape(-1, 1)), axis=1)
        x_train = PolynomialFeatures(self.degree).fit_transform(x_raw)

        # train independent regression model to predict each runtime of each model setting
        for i in range(self.n_models):
            y_train = runtimes[:, i]
            self.models[i] = LinearRegression().fit(x_train, y_train)

    def predict(self, size):
        """Predict runtime of all model settings on a dataset of given size."""
        x_test = np.append(size, np.log(size[0]))
        predictions = np.zeros(self.n_models)
        for i in range(self.n_models):
            predictions[i] = self.models[i].predict(x_test)
        return predictions
