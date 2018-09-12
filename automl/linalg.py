"""
Linear algebra helper functions.
"""

import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import qr
from scipy.stats import norm


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


def pivot_columns(a, threshold=0.03):
    """Computes the QR decomposition of a matrix with column pivoting, i.e. solves the equation AP=QR such that Q is
    orthogonal, R is upper triangular, and P is a permutation matrix.

    Args:
        a (np.ndarray):    Matrix for which to compute QR decomposition.
        threshold (float): All singular values less than threshold * (largest singular value) will be set to 0
    Returns:
        np.array: The permutation p.
    """
    rank = approx_rank(a, threshold=threshold)
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
    std = np.std(a, axis=0)
    u, s, vt = svds(a, k=rank)
    u = np.fliplr(u)
    s = np.flipud(s)
    vt = np.flipud(vt)
    sigma_sqrt = np.diag(np.sqrt(s))
    x = np.dot(u, sigma_sqrt).T
    y = np.dot(np.dot(sigma_sqrt, vt), np.diag(std))
    return x, y, vt

def EDF(item, array):
    n = len(array)
    array_unique = np.unique(array)
    ranking_dict = {}
    for unique_item in array_unique:
        ranking_dict[unique_item] = len(np.where(array == unique_item)[0])
    
    cumulative_ranking = np.sum([ranking_dict[unique_item] for unique_item in array_unique if unique_item < item])
    #set cumulative ranking to be number of smaller items + half the number of equal items (if there exist)
    try:
        cumulative_ranking += np.ceil(ranking_dict[item] / 2)
    except KeyError:
        pass
    return cumulative_ranking / (n + 1)

def inv_EDF(y, array):
    n = len(array)
    ranking = y * (n + 1)
    ranking_int = int(np.ceil(ranking)) - 1
    if ranking_int >= n:
        ranking_int = n-1
    item = np.sort(array)[ranking_int]
    return item


def coca(a, rank):
    """
        a: fully observed matrix.
        Not yet optimized for storage and speed.
    """
    m, n = a.shape
    Y_coca = np.zeros((m, n))
    Z_coca = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            Y_coca[i, j] = EDF(a[i, j], a[:, j])
            Z_coca[i, j] = norm.ppf(Y_coca[i, j])
    X, Y, Vt = pca(Z_coca, rank=rank)
    return X, Y, Vt

def impute_coca(A, a, known_indices, rank=None):
    """Imputes the missing entries of a vector a, given a fully observed matrix A of which a forms a new row.

    Args:
        A (np.ndarray):           Fully observed matrix.
        a (np.ndarray):           1xn partially observed array.
        known_indices (np.array): Array of observed entries; from the set {1,...,n}
        rank (int):               Approximate rank of A.
    Returns:
        np.ndarray: 1xn imputed array.
    """
    #this function now only supports the case when a is a 1-by-n numpy.ndarray (i.e. only one test dataset)
    a = a.reshape(1, -1)
    rank = rank or len(known_indices)
    z = np.zeros((1, A.shape[1]))
    X, Y, _ = coca(A, rank=rank)
    for observed_idx in known_indices:
        z[:, observed_idx] = norm.ppf(EDF(a[0, observed_idx], A[:, observed_idx]))
    # find x using matrix division using known portion of a, corresponding columns of A
    x = np.linalg.lstsq(Y[:, known_indices].T, z[:, known_indices].T, rcond=None)[0].T
    # approximate full a as x*Y
    z_coca_estimated = np.dot(x, Y)
    y_coca_estimated = norm.cdf(z_coca_estimated)
    x_coca_estimated = [inv_EDF(y_coca_estimated[0, j], A[:, j]) for j in range(A.shape[1])]
    return np.array(x_coca_estimated).reshape(1, -1)


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

