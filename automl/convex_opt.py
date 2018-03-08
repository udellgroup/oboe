"""
Find columns of error matrix to minimize variance of predicted latent features.
Solves convex optimization problem as described in chapter 7.5 in https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
"""

import numpy as np
from scipy.optimize import minimize


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

