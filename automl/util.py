"""
Miscellaneous helper functions.
"""

import numpy as np
import inspect
from sklearn.metrics import mean_squared_error


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
            pp = y_predicted == i
            pn = np.invert(pp)
            tp = pp & (y_observed == i)
            tn = pn & (y_observed != i)
            errors.append(0.5 * tp.sum()/np.max(pp.sum(), epsilon) + tn.sum()/np.max(pn.sum(), epsilon))

        return np.mean(errors)

    elif p_type == 'regression':

        return mean_squared_error(y_observed, y_predicted)


def invalid_args(func, arglist):
    """Check if args is a valid list of arguments to be passed to the function func.

    Args:
        func (function): Function to check arguments for.
        arglist (list): Proposed arguments

    Returns:
        set: Set of arguments in args that are invalid (if any).
    """

    args = inspect.getfullargspec(func)[0]
    return set(arglist) - set(args)


def check_arguments(p_type, algorithms, hyperparameters, defaults):
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
    assert p_type in ['classification', 'regression'], "Please specify a valid type."

    # set selected algorithms to default set if not specified
    all_algs = list(defaults['algorithms'][p_type].keys())
    if algorithms is None:
        algorithms = all_algs

    # check if selected algorithms are a subset of supported algorithms for given problem type
    assert set(algorithms).issubset(set(all_algs)), \
        "Unsupported algorithm(s) {}.".format(set(algorithms) - set(all_algs))

    # set selected hyperparameters to default set if not specified
    all_hyp = list(defaults['hyperparameters'][p_type])
    if hyperparameters is None:
        hyperparameters = all_hyp

    # check if selected hyperparameters are valid arguments to scikit-learn models
    invalid = [invalid_args(defaults['algorithms'][p_type][alg], hyperparameters[alg].keys())
               for alg in hyperparameters.keys()]
    for i, args in enumerate(invalid):
        assert len(args) == 0, "Unsupported hyperparameter(s) {} for algorithm {}" \
            .format(args, list(hyperparameters.keys())[i])

    # TODO: check if necessary to generate new error matrix
