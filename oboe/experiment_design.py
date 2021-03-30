"""
Find columns of error matrix to minimize variance of predicted latent features.
Solves convex optimization problem as described in chapter 7.5 in https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
"""

import numpy as np
import os
from . import linalg
import pandas as pd
import pickle
import openml
from scipy.linalg import qr
import subprocess
from scipy.optimize import minimize


def pivot_columns_time(A, t, t_max, rank=None, columns_to_avoid=None, threshold=None, column_picking_threshold="rank"):
    """
    Only for TensorOboe. Column_picking_threshold: "rank" or "size".
    """
    if columns_to_avoid is not None:
        set_of_columns_to_avoid = set(columns_to_avoid)
    else:
        set_of_columns_to_avoid = set()
    
    t_sum = 0
    assert (threshold is None) != (rank is None), "Exactly one of threshold and rank should be specified."
    if threshold is not None:
        rank = linalg.approx_matrix_rank(A, threshold)
    
    if column_picking_threshold == "rank":
        valid = np.where(t <= t_max / (2 * rank))[0]
    elif column_picking_threshold == "size":
        valid = np.where(t <= t_max / (2 * A.shape[1]))[0]
    
    r = []
    i = 0
    if len(valid) < rank: # there aren't enough pipeline to finish in the time budget, then we need to do greedy
        case = 'greedy_initialization'
        idx_sorted_by_time = np.argsort(t)
        while(t_sum < t_max):
            idx = idx_sorted_by_time[i]
            if t_sum + t[idx] <= t_max:
                r.append(idx)
                t_sum = t_sum + t[idx]
                i += 1
            else:
                break
        
    else:
        case = 'qr_initialization'
#     print("{} valid in {}".format(valid, A.shape))
        qr_columns = qr(A[:, valid], pivoting=True)[2]        
        
        while(len(r) < rank):
            qr_column = qr_columns[i]
            if valid[qr_column] not in set_of_columns_to_avoid:
                r.append(valid[qr_column])
                t_sum += t[valid[qr_column]]
            i += 1
    return r, t_sum, case


def greedy_stepwise_selection_with_time(Y, t, initialization, t_elapsed, t_max, idx_to_exclude=None, verbose=False):
    """
    Only for TensorOboe. D-optimal experiment design.
    """
    n_cols = Y.shape[1]
    assert n_cols == len(t), "Dimensionality mismatch!"
    if t_elapsed >= t_max:
        print("time limit exceeded!")
        return []
    
    t_sum = t_elapsed
    X = Y[:, initialization] @ Y[:, initialization].T
    selected = list(initialization)
    
    if idx_to_exclude is not None:
        set_of_indices_to_avoid = set(idx_to_exclude)
    else:
        set_of_indices_to_avoid = set()
    
    while(t_sum <= t_max):
        if verbose and not iteration % 10:
            print("Iteration {}".format(iteration))
        to_select = list(set(range(n_cols)).difference(set(selected)).difference(set_of_indices_to_avoid))
        try:
            X_inv = np.linalg.inv(X)
        except:
            break
        obj_all = np.array([(Y[:, i] @ X_inv @ Y[:, i])/t[i] for i in to_select])
        
        if not all(np.isnan(obj_all)):
            valid = np.where(t_sum + t[to_select] <= t_max)[0]
            if len(valid) == 0:
                break
            to_add = to_select[valid[np.nanargmax(obj_all[valid])]]
        else:
            break        
            
#         print(to_add)

        X = X + np.outer(Y[:, to_add], Y[:, to_add])
        selected.append(to_add)
        t_sum = t_sum + t[to_add]
#         print("total time: {}".format(t_sum))
    
    if verbose:
        print("number of selected design elements: {}".format(len(selected)))
        print("condition number of final design matrix: {}".format(np.linalg.cond(X)))
    return selected


def solve(t_predicted, t_max, n_cores, Y, scalarization='D'):
    """
    Only for Oboe.
    Solve the following optimization problem:
    minimize -log(det(sum_i v[i]*Y[:, i]*Y[:, i].T)) subject to 0 <= v[i] <= 1 and t_predicted.T * v <= t_max
    The optimal vector v is an approximation of a boolean vector indicating which entries to sample.

    Args:
         t_predicted (np.ndarray): 1-d array specifying predicted runtime for each model setting
         t_max (float):            maximum runtime of sampled model
         n_cores (int):            number of cores to use
         Y (np.ndarray):           matrix representing latent variable weights of error matrix
         scalarization (str):      scalarization method in experimental design.
    Returns:
        np.ndarray:                optimal vector v (not truncated to binary values)
    """

    n = len(t_predicted)

    if scalarization == 'D':
        def objective(v):
            sign, log_det = np.linalg.slogdet(Y @ np.diag(v) @ Y.T)
            return -1 * sign * log_det
    elif scalarization == 'A':
        def objective(v):
            return np.trace(np.linalg.pinv(Y @ np.diag(v) @ Y.T))
    elif scalarization == 'E':
        def objective(v):
            return np.linalg.norm(np.linalg.pinv(Y @ np.diag(v) @ Y.T), ord=2)
    def constraint(v):
        return t_max * n_cores- t_predicted @ v
    v0 = np.full((n, ), 0.5)
    constraints = {'type': 'ineq', 'fun': constraint}
    v_opt = minimize(objective, v0, method='SLSQP', bounds=[(0, 1)] * n, options={'maxiter': 30},
                     constraints=constraints)
    
    return v_opt.x


