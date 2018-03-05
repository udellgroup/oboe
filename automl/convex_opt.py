"""
Code for solving convex optimization problems to minimize variance of entries selected.
"""

#import libraries
import numpy as np
import pandas as pd
from cvxpy import *
import openml
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds
from scipy.linalg import qr
import glob
from os.path import basename
import os

#import scikit-learn modules
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

#import lowrank-automl modules
import util
import linalg as lrm

def get_dataset_sizes(default_error_matrix):
    """
    Get the numbers of data points and features of the datasets.
    
    Args:
        default_error_matrix (pandas.core.frame.DataFrame): The default error matrix DataFrame.
    Returns:
        dataset_sizes (np.ndarray): The dataset sizes; each row is [dataset_index, number_of_data_points, number_of_features].
        
    """
    openml_datasets = openml.datasets.list_datasets()
    openml_datasets = pd.DataFrame.from_dict(openml_datasets, orient='index')
    dataset_sizes = openml_datasets[['NumberOfInstances', 'NumberOfFeatures']]
    dataset_sizes = np.concatenate((np.array([dataset_sizes.index]).T, dataset_sizes.values), axis=1)
    indices = default_error_matrix.index.tolist()
    for i in set(indices).difference(set(dataset_sizes[:, 0])):
        dataset=openml.datasets.get_dataset(i)
        data_numeric, data_labels, categorical = dataset.get_data(target=dataset.default_target_attribute,return_categorical_indicator=True)
        dataset_sizes = np.concatenate((dataset_sizes, np.array([[i, data_numeric.shape[0], data_numeric.shape[1]]])))
    return dataset_sizes


def runtime_prediction_via_poly_fitting(dataset_sizes, poly_order, runtime_train, x_train, runtime_index, bool_log=True, bool_return_coefs=False):
    """
    Based on dataset sizes and model runtimes, predict the runtime of models on a new dataset.
    
    Args:
        dataset_sizes (np.ndarray): Matrix for dataset sizes. Each row is [dataset_index, number_of_data_points, number_of_features].
        poly_order (int): The order of polynomials used in runtime fitting.
        runtime_train (np.ndarray): A matrix containing runtime of models (or their logarithms) on training datasets. The first column is dataset indices.
        x_train (np.ndarray): Features of the training dataset.
        bool_log (Boolean): Whether to take into log(n) in polynomial fitting.
        bool_return_coefs (Boolean): Whether to return coefficients of linear fitting.
        
    Returns:
        runtime_predict (np.ndarray): Predicted runtime of models or their logarithms, depending on the input. The first entry is dataset indices.
        coefs (np.ndarray): Optional, only exist when bool_return_coefs==True. Coefficients of polynomial fittings.
        
    """
    num_training_sets, num_models = runtime_train.shape
    indices = np.array([np.where(dataset_sizes[:, 0]==runtime_index[i])[0][0] for i in range(len(runtime_train[:, 0]))])
    X = dataset_sizes[indices, 1:]
    if bool_log:
        transformer = FunctionTransformer(np.log)
        X_logn = transformer.transform(np.array([X[:, 0]]).T)
        X_combined = np.concatenate((X, X_logn), axis=1)
    else:
        X_combined = X

    x = np.array([x_train.shape])

    if bool_log:
        x_logn = transformer.transform(x[0, 0])
        x_combined = np.concatenate((x, x_logn), axis=1)
    else:
        x_combined = x

    runtime_predict = []
    num_large_error_terms = []
    coefs = []

    for iter_model in range(num_models):
        Y = runtime_train[:, iter_model]
        model = Pipeline([('poly', PolynomialFeatures(degree=poly_order)), ('linear', LinearRegression(fit_intercept=False))])
        model = model.fit(X_combined, Y)
        coefs.append(model.named_steps['linear'].coef_)
        y = model.predict(x_combined)[0]
        runtime_predict.append(y)

    runtime_predict = np.array(runtime_predict)

    if bool_return_coefs:
        return runtime_predict, coefs
    else:
        return runtime_predict


def proj_to_0_to_1(x, eps=1e-5):
    """
    Project all entries in matrix x into region [eps, 1 - eps] by keeping entries between [-eps, eps] unchanged, replacing entries less than eps by eps, and entries larger than (1 - eps) by (1 - eps). This is to preprocess the error matrix before applying inverse sigmoid function to convert all the [0, 1] entries to (-∞, +∞).
    
    Args:
        x (np.ndarray): The error matrix to convert.
        eps (float): the epsilon value.
        
    Returns:
        np.ndarray: The projected error matrix.
        
    """
    return x + np.multiply(x < eps, eps) - np.multiply(x > 1 - eps, eps)

def sigmoid(x):
    """
    Sigmoid function.
    
    Args:
        x (np.ndarray): Entries within (0, 1).
        
    Returns:
        np.ndarray: Entries within (-∞, +∞).
        
    """
    
    return 1/(1+np.exp(-x))

def inv_sigmoid(x):
    """
    Inverse sigmoid function.
    
    Args:
        x (np.ndarray): Entries within (-∞, +∞).
    
    Returns:
        np.ndarray: Entries within (0, 1).
        
    """
    
    return -np.log(1/x - 1)


def transform_and_keep_indices(numpy_array, operator):
    """
    Transform all the columns other than the first of the given numpy array using the given operator.
    
    Args:
        numpy_array(np.ndarray): Input numpy array; can be the runtime matrix.
        operator(function): The transformation to be applied to all columns other than the first.
        
    Returns:
        transformed_numpy_array(np.ndarray): the transformed numpy array.
        
    """
    dim = len(numpy_array.shape)
    assert dim == 1 or dim == 2, "Input must be a 1-D or 2-D numpy array."
    if dim == 1:
        numpy_array = np.array([numpy_array])
    transformed_numpy_array = np.concatenate((np.array([numpy_array[:, 0]]).T,
                                              operator(numpy_array[:, 1:])), axis=1)
    if dim == 1:
        transformed_numpy_array = transformed_numpy_array[0, :]
    return transformed_numpy_array


def min_variance_model_selection(runtime_limit,
                                 runtime_predict, error_matrix, threshold=0.03,
                                 bool_plot_solution_quality=False, relaxation_threshold=0.8):
    
    """
    Select models with the assumption of i.i.d. random Gaussian noise on each observation, and aims to minimize the variance of latent representation of the new row in the error matrix.
    
    Args:
        runtime_limit (float): The user-specified time limit for running selected models.
        error_matrix (np.ndarray): The error matrix in use.
        threshold (float): The threshold for truncating singular values.
        bool_plot_solution_quality (Boolean): Whether to plot the λ values of the relaxed integer programming problem.
        relaxation_threshold (float): The threshold for truncating λ values to get an approximate solution for the original integer programming problem.
        
        
    Returns:
        λ_indices_selected (np.ndarray): The indices of selected columns.
        
    """
    
    assert threshold>0 and threshold<1, "The threshold for truncating singular values should be a float between 0 and 1."
    assert relaxation_threshold>0 and relaxation_threshold<1, "The threshold for converting relaxed integer programming problem back to the integer version should be a float between 0 and 1."
    
    
#    #In package testing, delete the corresponding rows in runtime matrix and error matrix if the rows corresponding to test_dataset_index appears in them.
#
#    indices_runtime = runtime_limit[:, 0]
#    indices_em = error_matrix[:, 0]
#    if test_dataset_index in indices_runtime:
#        runtime_limit = np.delete(runtime_limit, list(indices_runtime).index(test_dataset_index), 0)
#    if test_dataset_index in indices_em:
#        error_matrix = np.delete(error_matrix, list(indices_em).index(test_dataset_index), 0)

    _, num_models = error_matrix.shape

    #low rank approximation

    rank = lrm.approx_rank(error_matrix, threshold=threshold)
    rank_one_percent = lrm.approx_rank(error_matrix, threshold=0.01)
    X,Y,Vt = lrm.pca(error_matrix, threshold=threshold)
    num_pivots_selected = rank
    
    #model selection for variance minimization via D-optimal design
    λ = Variable(num_models)
    objective = Minimize(-log_det(sum([λ[i]*np.outer(Y[:, i], Y[:, i]) for i in range(num_models)])))
    constraints = [0 <= λ, λ <= 1]
    # time constraint
    constraints += [runtime_predict*λ <= runtime_limit]
    prob = Problem(objective, constraints)
    result = prob.solve()
    #the solution to λ in numpy array format
    λ_sol = np.array(λ.value).T[0]

    if bool_plot_solution_quality:
        f = plt.figure()
        plt.hist(λ_sol)
        plt.title('Distribution of λ Values when runtime='+str(runtime_limit))
        f.savefig('runtime='+str(runtime_limit)+'.png', dpi=250, bbox_inches='tight', format = 'png')
    
    λ_indices_selected = np.where(λ_sol>relaxation_threshold)[0]

    return λ_indices_selected



