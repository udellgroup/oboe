"""
Miscellaneous helper functions.
"""

import numpy as np
import pandas as pd
import re
import inspect
import itertools
import os
import sys
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


CLS = ['KNN', 'DT', 'RF', 'GBT', 'AB', 'lSVM', 'kSVM', 'Logit', 'Perceptron', 'GNB']
REG = ['Lasso', 'Ridge', 'ElasticNet']

CLASSIFICATION = dict(zip(CLS, list(map(lambda name: eval(name), CLS))))
REGRESSION = dict(zip(REG, list(map(lambda name: eval(name), REG))))

HYPERPARAMETERS_C = {'KNN':        {'n_neighbors':       np.arange(1, 17, 2, dtype=int)},
                     'DT':         {'min_samples_split': np.geomspace(0.01, 0.00001, 4)},
                     'RF':         {'min_samples_split': np.geomspace(0.01, 0.00001, 4)},
                     'GBT':        {'learning_rate':     np.geomspace(0.1, 0.001, 3)},
                     'AB':         {'n_estimators':      np.array([50, 100]),
                                    'learning_rate':     np.array([1.0, 2.0])},
                     'lSVM':       {'C':                 np.array([0.25, 0.5, 0.75, 1.0, 2.0])},
                     'kSVM':       {'C':                 np.array([0.25, 0.5, 0.75, 1.0, 2.0])},
                     'Logit':      {'C':                 np.array([0.25, 0.5, 0.75, 1.0, 2.0])},
                     'Perceptron': {},
                     'GNB':        {}
                     }
HYPERPARAMETERS_R = {}

DEFAULTS = {'algorithms':      {'classification': CLASSIFICATION,    'regression': REGRESSION},
            'hyperparameters': {'classification': HYPERPARAMETERS_C, 'regression': HYPERPARAMETERS_R}}


def error(y_observed, y_predicted, p_type):
    """Compute error metric for the model; varies based on classification/regression and algorithm type.
    BER (Balanced Error Rate): For classification.
                              1/n * sum (0.5*(true positives/predicted positives + true negatives/predicted negatives))
    MSE (Mean Squared Error): For regression. 1/n * sum(||y_pred - y_obs||^2).

    Args:
        y_observed (np.ndarray): Observed labels.
        y_predicted (np.ndarray): Predicted labels.
        p_type (str): Type of problem. One of {'classification', 'regression'}

    Returns:
        float: Error metric.
    """
    assert p_type in ['classification', 'regression'], "Please specify a valid type."

    if p_type == 'classification':
        errors = []
        epsilon = 1e-15
        for i in np.unique(y_observed):
            tp = ((y_observed == i) & (y_predicted == i)).sum()
            tn = ((y_observed != i) & (y_predicted != i)).sum()
            fp = ((y_observed != i) & (y_predicted == i)).sum()
            fn = ((y_observed == i) & (y_predicted != i)).sum()
            errors.append(1 - 0.5*(tp / np.maximum(tp + fn, epsilon)) - 0.5*(tn / np.maximum(tn + fp, epsilon)))
        return np.mean(errors)

    elif p_type == 'regression':
        return mean_squared_error(y_observed, y_predicted)


def invalid_args(func, arglist):
    """Check if args is a valid list of arguments to be passed to the function func.

    Args:
        func (function): Function to check arguments for.
        arglist (list): Proposed arguments

    Returns:
        set: Set of arguments in args that are invalid (returns empty set if there are none).
    """
    args = inspect.getfullargspec(func)[0]
    return set(arglist) - set(args)


def check_arguments(p_type, algorithms, hyperparameters, defaults=DEFAULTS):
    """Check if arguments to constructor of AutoLearner object are valid, and default error matrix can be used.

    Args:
        p_type (str): Problem type. One of {'classification', 'regression'}
        algorithms (list): List of selected algorithms as strings. (e.g. ['KNN', 'lSVM', 'kSVM']
        hyperparameters (dict): Nested dict of selected hyperparameters.
        defaults (dict): Nested dict of default algorithms & hyperparameters.

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


def generate_settings(algorithms, hyperparameters, sort=True):
    """Generate column headings of error matrix.

    Args:
        algorithms (list): A list of algorithms in strings (e.g. ['KNN', 'RF', 'lSVM'])
        hyperparameters (dict): A nested dictionary of hyperparameters. First key is algorithm type (str), second key
        is hyperparameter name (str); argument to pass to scikit-learn constructor with array of values
        (e.g. {'KNN': {'n_neighbors': np.array([1, 3, 5, 7]),
                       'p': np.array([1, 2])}}).
        sort (bool): Whether to sort settings in alphabetical order with respect to algorithm name.

    Returns:
        list: List of nested dictionaries, one entry for each model setting.
              (e.g. [{'algorithm': 'KNN', 'hyperparameters': {'n_neighbors': 1, 'p': 1}},
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
    files = [file for file in os.listdir(save_dir) if file.endswith('.csv')]
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
    for file in files:
        file_path = os.path.join(save_dir, file)
        dataframe = pd.read_csv(file_path, index_col=0)
        if headers is None:
            headers = list(dataframe)
        else:
            assert set(headers) == set(list(dataframe)), "All results must share same headers."
        if np.isnan(dataframe.values).any():
            # if values contain NaNs, generations has not yet finished
            pass
        else:
            permutation = [headers.index(h) for h in list(dataframe)]
            error_matrix_rows += (np.expand_dims(dataframe.values[0, permutation], 0), )
            runtime_matrix_rows += (np.expand_dims(dataframe.values[1, permutation], 0), )
            ids.append(file.split('.')[0])
            # os.remove(file_path)

    # save results
    pd.DataFrame(np.vstack(error_matrix_rows), index=ids, columns=headers).to_csv(os.path.join(save_dir, em))
    pd.DataFrame(np.vstack(runtime_matrix_rows), index=ids, columns=headers).to_csv(os.path.join(save_dir, rm))


if __name__ == '__main__':
    merge_rows(sys.argv[1])
