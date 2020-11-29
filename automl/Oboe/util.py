#requires a log file in the folder that contains csv files.

"""
Miscellaneous helper functions.
"""

import inspect
import itertools
import json
import numpy as np
import os
import pandas as pd
import pkg_resources
import re
import sys
import glob
from math import isclose
from sklearn.metrics import mean_squared_error

# Classification algorithms
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import ExtraTreesClassifier as ExtraTrees
from sklearn.ensemble import GradientBoostingClassifier as GBT
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.svm import LinearSVC as lSVM
from sklearn.svm import SVC as kSVM
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLP

# Regression algorithms
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
# TODO: include more regression algorithms


defaults_path = pkg_resources.resource_filename(__name__, 'defaults')
with open(os.path.join(defaults_path, 'classification.json'), 'r') as f:
    CLS = json.load(f)
with open(os.path.join(defaults_path, 'regression.json'), 'r') as f:
    REG = json.load(f)

ALGORITHMS_C = dict(zip(CLS['algorithms'], list(map(lambda name: eval(name), CLS['algorithms']))))
ALGORITHMS_R = dict(zip(REG['algorithms'], list(map(lambda name: eval(name), REG['algorithms']))))

DEFAULTS = {'algorithms':       {'classification': ALGORITHMS_C,           'regression': ALGORITHMS_R},
            'hyperparameters': {'classification': CLS['hyperparameters'],  'regression': REG['hyperparameters']}}


def extract_columns(df, algorithms=None, hyperparameters=None):
    """
    Extract certain columns of the error matrix.
    
    Args:
        error_matrix (DataFrame):    The error matrix to be extracted.
        algorithms (string or list): One or a list of algorithms as search space.
        
    Args to be implemented:
        hyperparameters (list):      A list of hyperparameters as search space.
        
    Returns:
        DataFrame:                   A DataFrame consisting of corresponding columns.
    """
    assert algorithms is not None or hyperparameters is not None, \
    "At least one of the 'algorithms' and 'hyperparameters' need to be specified!"
    sampled_columns = []
    for item in list(df):
        to_sample_this_column = False
        if algorithms is None:
            to_sample_this_column = True
        elif eval(item)['algorithm'] in algorithms:
            if hyperparameters is None:
                to_sample_this_column = True
            else:
                to_sample_this_column = True
                hyperparameter_column = eval(item)['algorithm']
                hyperparameter_allowed = hyperparameters[eval(item)['algorithm']]
                for key in hyperparameter_column:
                    if not key in hyperparameter_allowed.keys():
                        continue
                    else:
                        if hyperparameter_column[key] in hyperparameter_allowed[key]:
                            continue
                        else:
                            to_sample_this_column = False
                            break        
        if to_sample_this_column == True:
            sampled_columns.append(item)
    return df[sampled_columns]

def extract_column_names(df, algorithms=None, hyperparameters=None):
    """
    Extract names of certain columns of the error matrix.
    
    Args:
        error_matrix (DataFrame):    The error matrix to be extracted.
        algorithms (string or list): One or a list of algorithms as search space.
        
    Args to be implemented:
        hyperparameters (list):      A list of hyperparameters as search space.
        
    Returns:
        list:                        A list of column names.
    """
    return list(extract_columns(df, algorithms=algorithms, hyperparameters=hyperparameters))

def error(y_true, y_predicted, p_type):
    """Compute error metric for the model; varies based on classification/regression and algorithm type.
    BER (Balanced Error Rate): For classification.
                              1/n * sum (0.5*(true positives/predicted positives + true negatives/predicted negatives))
    MSE (Mean Squared Error): For regression. 1/n * sum(||y_pred - y_obs||^2).

    Args:
        y_true (np.ndarray):      Observed labels.
        y_predicted (np.ndarray): Predicted labels.
        p_type (str):             Type of problem. One of {'classification', 'regression'}
    Returns:
        float: Error metric.
    """

    assert p_type in {'classification', 'regression'}, "Please specify a valid type."
    y_true = np.squeeze(y_true)
    y_predicted = np.squeeze(y_predicted)

    if p_type == 'classification':
        errors = []
        epsilon = 1e-15
        for i in np.unique(y_true):
            tp = ((y_true == i) & (y_predicted == i)).sum()
            tn = ((y_true != i) & (y_predicted != i)).sum()
            fp = ((y_true != i) & (y_predicted == i)).sum()
            fn = ((y_true == i) & (y_predicted != i)).sum()
            errors.append(1 - 0.5*(tp / np.maximum(tp + fn, epsilon)) - 0.5*(tn / np.maximum(tn + fp, epsilon)))
        return np.mean(errors)

    elif p_type == 'regression':
        return mean_squared_error(y_true, y_predicted)


def invalid_args(func, arglist):
    """Check if args is a valid list of arguments to be passed to the function func.

    Args:
        func (function): Function to check arguments for
        arglist (list):  Proposed arguments
    Returns:
        set: Set of arguments in args that are invalid (returns empty set if there are none).
    """
    args = inspect.getfullargspec(func)[0]
    return set(arglist) - set(args)


def check_arguments(p_type, algorithms, hyperparameters, defaults=DEFAULTS):
    """Check if arguments to constructor of AutoLearner object are valid, and default error matrix can be used.

    Args:
        p_type (str):           Problem type. One of {'classification', 'regression'}
        algorithms (list):      List of selected algorithms as strings. (e.g. ['KNN', 'lSVM', 'kSVM']
        hyperparameters (dict): Nested dict of selected hyperparameters.
        defaults (dict):        Nested dict of default algorithms & hyperparameters.
    Returns:
        bool: Whether or not the default error matrix can be used.
    """
    # check if valid problem type
    assert p_type.lower() in ['classification', 'regression'], "Please specify a valid type."

    # set selected algorithms to default set if not specified
    all_algs = list(defaults['algorithms'][p_type].keys())
    if algorithms is None:
        algorithms = all_algs

    # check if selected algorithms are a subset of supported algorithms for given problem type
    assert set(algorithms).issubset(set(all_algs)), \
        "Unsupported algorithm(s) {}.".format(set(algorithms) - set(all_algs))

    # set selected hyperparameters to default set if not specified
    all_hyp = defaults['hyperparameters'][p_type]
    if hyperparameters is None:
        hyperparameters = all_hyp

    # check if selected hyperparameters are valid arguments to scikit-learn models
    invalid = [invalid_args(defaults['algorithms'][p_type][alg], hyperparameters[alg].keys())
               for alg in hyperparameters.keys()]
    for i, args in enumerate(invalid):
        assert len(args) == 0, "Unsupported hyperparameter(s) {} for algorithm {}" \
            .format(args, list(hyperparameters.keys())[i])

    # check if it is necessary to generate new error matrix, i.e. are all hyperparameters in default error matrix
    compatible_columns = []
    new_columns = []
    default_settings = generate_settings(defaults['algorithms'][p_type].keys(), defaults['hyperparameters'][p_type])
    for alg in hyperparameters.keys():
        for values in itertools.product(*hyperparameters[alg].values()):
            setting = {'algorithm': alg, 'hyperparameters': dict(zip(hyperparameters[alg].keys(), list(values)))}
            if setting in default_settings:
                compatible_columns.append(setting)
            else:
                new_columns.append(setting)
    return compatible_columns, new_columns


def knapsack(weights, values, capacity):
    """Solve the knapsack problem; maximize sum_i v[i]*x[i] subject to sum_i w[i]*x[i] <= W and x[i] in {0, 1}

    Args:
        weights (np.ndarray): "weights" of each item
        values (np.ndarray):  "values" of each item
        capacity (int):       maximum "weight" allowed
    Returns:
        set: list of selected indices
    """
    assert len(weights) == len(values), "Weights & values must have same shape."
    assert type(capacity) == int, "Capacity must be an integer."
    n = len(weights)
    m = np.zeros((n+1, capacity+1)).astype(int)

    for i in range(n+1):
        for w in range(capacity+1):
            if i == 0 or w == 0:
                pass
            elif weights[i-1] <= w:
                m[i, w] = max(values[i-1] + m[i-1, w-weights[i-1]], m[i-1, w])
            else:
                m[i, w] = m[i-1, w]

    def find_selected(j, v):
        if j == 0:
            return set()
        if m[j, v] > m[j-1, v]:
            return {j-1}.union(find_selected(j-1, v - weights[j-1]))
        else:
            return find_selected(j-1, v)

    return find_selected(n, capacity)


def check_dataframes(m1, m2):
    """Check if 2 dataframes have the same shape and share the same index column.

    Args:
        m1 (DataFrame): first dataframe
        m2 (DataFrame): second dataframe
    Returns:
        bool:           Whether the conditions are satisfied
    """
    assert m1.shape == m2.shape
    assert set(m1.index) == set(m2.index)
    return True


def generate_settings(algorithms, hyperparameters, sort=True):
    """Generate column headings of error matrix.

    Args:
        algorithms (list):      A list of algorithms in strings (e.g. ['KNN', 'RF', 'lSVM'])
        hyperparameters (dict): A nested dictionary of hyperparameters. First key is algorithm type (str), second key
                                is hyperparameter name (str); argument to pass to scikit-learn constructor with array
                                of values
                                (e.g. {'KNN': {'n_neighbors': np.array([1, 3, 5, 7]),
                                               'p':           np.array([1, 2])}}).
        sort (bool):            Whether to sort settings in alphabetical order with respect to algorithm name.
    Returns:
        list: List of nested dictionaries, one entry for each model setting.
              (e.g. [{'algorithm': 'KNN',  'hyperparameters': {'n_neighbors': 1, 'p': 1}},
                     {'algorithm': 'lSVM', 'hyperparameters': {'C': 1.0}}])
    """
    settings = []
    for alg in algorithms:
        hyperparams = hyperparameters[alg]
        for values in itertools.product(*hyperparams.values()):
            configs = dict(zip(hyperparams.keys(), list(values)))
            for key, val in configs.items():
                if isinstance(val, (int, float)):
                    if isclose(val, round(val)):
                        configs[key] = int(round(val))
            settings.append({'algorithm': alg, 'hyperparameters': configs})
    if sort:
        settings = sorted(settings, key=lambda k: k['algorithm'])
    return settings


def merge_rows(save_dir):
    """Merge rows of error matrix. Creates two CSV files: one error matrix and one runtime matrix.

    Args:
        save_dir (str): Directory containing per-dataset CSV files of cross-validation errors & time for each model.
    """
    if not os.path.isdir(save_dir):
        print('Invalid path.')
        return

    # find files to concatenate (all .csv files; may contain previously merged results)
    files = [file for file in os.listdir(save_dir) if file.endswith('.csv') and 'sizes' not in file]
    em, rm = 'error_matrix.csv', 'runtime_matrix.csv'
    headers, ids, error_matrix_rows, runtime_matrix_rows = None, [], (), ()

    if (em in files) and (rm in files):
        errors = pd.read_csv(os.path.join(save_dir, files.pop(files.index(em))), index_col=0)
        runtimes = pd.read_csv(os.path.join(save_dir, files.pop(files.index(rm))), index_col=0)
        assert set(errors.index) == set(runtimes.index), "Previous results must share index column."
        assert set(list(errors)) == set(list(runtimes)), "Previous results must share headers."
        ids += list(errors.index)
        headers = list(errors)
        error_matrix_rows += (errors.values, )
        runtime_matrix_rows += (runtimes.values, )

    # concatenate new results
    # TODO: only load files corresponding to completed files in log.txt
    for file in files:
        file_path = os.path.join(save_dir, file)
        dataframe = pd.read_csv(file_path, index_col=0)
        if headers is None:
            headers = list(dataframe)
        else:
            assert set(headers) == set(list(dataframe)), "All results must share same headers."
        if np.isnan(dataframe.values).any():
            # if values contain NaNs, generation has not yet finished
            pass
        else:
            permutation = [headers.index(h) for h in list(dataframe)]
            error_matrix_rows += (np.expand_dims(dataframe.values[0, permutation], 0), )
            runtime_matrix_rows += (np.expand_dims(dataframe.values[1, permutation], 0), )
            ids.append(file.split('.')[0])
            try:
                os.mkdir(os.path.join(save_dir, "merged_csv_files"))
            except:
                pass
            os.rename(file_path, os.path.join(save_dir, "merged_csv_files", file))
#             os.remove(file_path)
            if len(error_matrix_rows) % 50 == 0:
                print('Merging {} files...'.format(len(error_matrix_rows)))

    # get dataset sizes
    # openml_datasets = openml.datasets.list_datasets()
    # openml_datasets = pd.DataFrame.from_dict(openml_datasets, orient='index')
    # dataset_sizes = openml_datasets[['NumberOfInstances', 'NumberOfFeatures']]

#     #find the log file
#     for f in glob.glob('{}/log*.txt'.format(save_dir)):
#          log_path = f
#     # save dataset sizes
#     with open(log_path, 'r') as file:
#         lines = file.readlines()
#     dataset_ids, sizes = [], []
#     for line in lines:
#         if 'Size' in line:
#             log_ids = [int(n) for n in re.findall(r'ID=(\d+)', line)]
#             size = [eval(n) for n in re.findall(r'Size=\((\d+, \d+)\)', line)]
#             if len(log_ids) == 1 and len(size) == 1:
#                 dataset_ids.append(log_ids[0])
#                 sizes.append(size[0])

    # save results
    pd.DataFrame(np.vstack(error_matrix_rows), index=ids, columns=headers).to_csv(os.path.join(save_dir, em))
    pd.DataFrame(np.vstack(runtime_matrix_rows), index=ids, columns=headers).to_csv(os.path.join(save_dir, rm))
#     pd.DataFrame(np.vstack(sizes), index=dataset_ids).to_csv(os.path.join(save_dir, 'dataset_sizes.csv'))    
    # dataset_sizes.to_csv(os.path.join(save_dir, 'dataset_sizes.csv'))

    
if __name__ == '__main__':
    merge_rows(sys.argv[1])
