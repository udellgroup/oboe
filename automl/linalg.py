"""
Linear algebra helper functions.
"""

import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import qr


def approx_rank(a, threshold=0.03):
    """Compute approximate rank of a matrix.

    Args:
        a (np.ndarray):    Matrix for which to compute rank.
        threshold (float): All singular values less than threshold * (largest singular value) will be set to 0
    Returns:
        int: The approximate rank of a.
    """
    s = np.linalg.svd(a, compute_uv=False)
    rank = s[s >= threshold * s[0]]
    return len(rank)


def pivot_columns(a, rank=None, threshold=None):
    """Computes the QR decomposition of a matrix with column pivoting, i.e. solves the equation AP=QR such that Q is
    orthogonal, R is upper triangular, and P is a permutation matrix.

    Args:
        a (np.ndarray):    Matrix for which to compute QR decomposition.
        threshold (float): Threshold specifying approximate rank of a. All singular values less than threshold * (largest singular value) will be set to 0
        rank (int):        The approximate rank.
    Returns:
        np.array: The permutation p.
    """
    assert (threshold is None) != (rank is None), "Exactly one of threshold and rank should be specified."
    if threshold is not None:
        rank = approx_rank(a, threshold)
    return qr(a, pivoting=True)[2][:rank]


def pca(a, rank=None, threshold=None):
    """Solves: minimize ||A_XY||^2 where ||.|| is the Frobenius norm.

    Args:
        a (np.ndarray):    Matrix for which to compute PCA.
        threshold (float): Threshold specifying approximate rank of a.
        rank (int):        The approximate rank.
    Returns:
        x, y (np.ndarray): The solutions to the PCA problem.
        vt (np.ndarray):   Transpose of V as specified in the singular value decomposition.
    """
    assert (threshold is None) != (rank is None), "Exactly one of threshold and rank should be specified."
    if threshold is not None:
        rank = approx_rank(a, threshold)
    # std = np.std(a, axis=0)
    u, s, vt = svds(a, k=rank)

    nonzero_pos = np.where(s > 0)[0]
    s = s[nonzero_pos]
    u = u[:, nonzero_pos]
    vt = vt[nonzero_pos, :]

    u = np.fliplr(u)
    s = np.flipud(s)
    vt = np.flipud(vt)
    # sigma_sqrt = np.diag(np.sqrt(s))
    # x = np.dot(u, sigma_sqrt).T
    # # y = np.dot(np.dot(sigma_sqrt, vt), np.diag(std))
    # y = np.dot(sigma_sqrt, vt)

    sigma = np.diag(s)
    x = np.dot(u, sigma).T
    y = vt
    return x, y, vt


def impute(A, a, known_indices, rank=None):
    """Imputes the missing entries of a vector a, given a fully observed matrix A of which a forms a new row.

    Args:
        A (np.ndarray):           Fully observed matrix.
        a (np.ndarray):           1xn partially observed array.
        known_indices (np.array): Array of observed entries; from the set {1,...,n}
        rank (int):               Approximate rank of A.
    Returns:
        np.ndarray: 1xn imputed array.
    """
    rank = rank or len(known_indices)
    x, y, _ = pca(A, rank=rank)
    # find x using matrix division using known portion of a, corresponding columns of A
    x = np.linalg.lstsq(y[:, known_indices].T, a[:, known_indices].T, rcond=None)[0].T
    # approximate full a as x*Y
    return np.dot(x, y)

