"""
Linear algebra helper functions.
"""

import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import qr


def approx_rank(a, threshold=0.03):
    """Compute approximate rank of a matrix.

    Args:
        a (np.ndarray): Matrix for which to compute rank.
        threshold (float): All singular values less than threshold * (largest singular value) will be set to 0
    Returns:
        int: The approximate rank of a.
    """
    s = np.linalg.svd(a, compute_uv=False)
    rank = s[s >= threshold * s[0]]
    return len(rank)


def pivot_columns(a, threshold=0.03):
    """Computes the QR decomposition of a matrix with column pivoting, i.e. solves the equation AP=QR such that Q is
    orthogonal, R is upper triangular, and P is a permutation matrix.

    Args:
        a (np.ndarray): Matrix for which to compute QR decomposition.
        threshold (float): All singular values less than threshold * (largest singular value) will be set to 0
    Returns:
        np.array: The permutation p.
    """
    rank = approx_rank(a, threshold=threshold)
    return qr(a, pivoting=True)[2][:rank]


def pca(a, threshold=0.03):
    """Solves: minimize ||A_XY||^2 where ||.|| is the Frobenius norm.

    Args:
        a (np.ndarray): Matrix for which to compute PCA.
        threshold (float): Threshold specifying approximate rank of a.
    Returns:
        x, y (np.ndarray): The solutions to the PCA problem.
        vt (np.ndarray): Transpose of V as specified in the singular value decomposition.
    """
    rank = approx_rank(a, threshold)
    std = np.std(a, axis=0)
    u, s, vt = svds(a, k=rank)
    sigma_sqrt = np.diag(np.sqrt(s))
    x = np.dot(u, sigma_sqrt).T
    y = np.dot(np.dot(sigma_sqrt, vt), np.diag(std))
    return x, y, vt


def impute(A, a, known_indices, threshold=0.03):
    """Imputes the missing entries of a vector a, given a fully observed matrix A of which a forms a new row.

    Args:
        A (np.ndarray): Fully observed matrix.
        a (np.ndarray): 1xn partially observed array.
        known_indices (np.array): Array of observed entries; from the set {1,...,n}
        threshold (float): Threshold specifying the approximate rank of A.
    Returns:
        np.ndarray: 1xn imputed array.
    """
    x, y, _ = pca(A, threshold=threshold)

    # find x using matrix division using known portion of a, corresponding columns of A
    x = np.linalg.lstsq(y[:, known_indices].T, a[:, known_indices].T, rcond=None)[0].T

    # approximate full a as x*Y
    return np.dot(x, y)

