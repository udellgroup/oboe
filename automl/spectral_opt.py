"""
Spectral approach to hyperparameter optimization.
Implementation of techniques described in: https://arxiv.org/abs/1706.00764
"""

import numpy as np
from itertools import combinations
from sklearn.linear_model import Lasso


def parity(s, x):
    """The parity function X_S(x) = prod_{i in S} x_i; equivalent to whether the number of occurences of False is even.

    Args:
        s (list):       a subset of the variables, i.e. a subset of [0, 1, ..., n-1]
        x (np.ndarray): a boolean input vector
    Returns:
        bool: the output of the parity function
    """
    assert len(x.shape) == 1, "Input vector must be 1-dimensional"
    return np.invert(x[s]).sum() % 2 == 0


def parity_basis(x, d):
    """Applies parity function for all subsets of [0, 1, ..., n-1] that are of size d.

    Args:
        x (np.ndarray): a boolean input vector
        d (int):        size of subsets
    Returns:
        np.ndarray:     a vector of length (n choose d) where the ith element is parity(s_i, x)
    """
    assert d <= len(x), "Subsets cannot be larger than length of input"
    transformed = []

    # all combinations of d elements of [0, 1, ..., n-1] where n is the length of the input vector
    subsets = list(combinations(np.arange(len(x)), d))
    subsets = [list(subset) for subset in subsets]
    for subset in subsets:
        transformed.append(parity(subset, x))
    return np.array(transformed)


def design_matrix(X, basis=parity_basis, *args):
    """Creates design matrix, i.e. the basis functions applied to each sample.

    Args:
        X (np.ndarray):   an array of boolean vectors (i.e. 2d ndarray)
        basis (function): basis function to apply
        *args (tuple):    arguments to the basis function
    Returns:
        np.ndarray:       the design matrix
    """
    return np.apply_along_axis(basis, 1, X, *args)


def psr(error_function, n_samples, sparsity, degree, lmbda, *args):
    """Conducts the Polynomial Sparse Recovery procedure as defined by Hazan et. al

    Args:
        error_function (function): function to minimize; i.e. from hyperparameter space to error of model
        n_samples (int):           number of samples to query from error function
        sparsity (int):            sparsity of error function
        degree (int):              degree of error function
        lmbda (float):             regularization parameter of lasso
        *args (tuple):             arguments to error function
    Returns:
        function:                  d-degree s-sparse fourier domain approximation of error function
        list:                      indices of relevant hyperparameters
    """
    p = 0.5                                                 # hyperparameter controlling probability of True/False
    X = np.random.rand(n_samples, error_function.dim) > p   # assume error_function has attribute specifying dimensions
    Phi = design_matrix(X, parity_basis, *(degree, ))       # TODO: does 0/1 -1/1 encoding make a difference?
    y = np.apply_along_axis(error_function, 1, X, *args)

    clf = Lasso(alpha=lmbda)
    clf.fit(Phi, y)

    alpha = clf.coef_                                       # fitted weight vector
    j = np.argsort(-1 * alpha)[:sparsity]                   # s largest coefficients of alpha

    def fourier_approximation(x):
        return sum([alpha[j[i]] * parity(s, x)] for i, s in enumerate(j))

    return fourier_approximation, j
