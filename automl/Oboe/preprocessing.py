"""
Pre-process datasets.
"""

import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer


def pre_process(raw_data, categorical, impute=True, standardize=True, one_hot_encode=True):
    """
    Pre-process one dataset.

    Args:
        raw_data (np.ndarray):    raw features of the n-by-d dataset, without indices and headings.
        categorical (list):       a boolean list of length d indicating whether each raw feature is categorical.
        impute (bool):            whether to impute missing entries or not.
        standardize (bool):       whether to standardize each feature or not.
        one_hot_encode (bool):    whether to use one hot encoding to pre-process categorical features or not.
    Returns:
        np.ndarray:               pre-processed dataset.
    """
    # list of pre-processed arrays (sub-portions of dataset)
    processed = []

    # whether to impute missing entries
    if impute:
        # if there are any categorical features
        if np.array(categorical).any():
            raw_categorical = raw_data[:, categorical]
            # impute missing entries in categorical features using the most frequent number
            imp_categorical = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=False)
            processed.append(imp_categorical.fit_transform(raw_categorical))

        # if there are any numeric features
        if np.invert(categorical).any():
            raw_numeric = raw_data[:, np.invert(categorical)]
            # impute missing entries in non-categorical features using mean
            imp_numeric = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
            processed.append(imp_numeric.fit_transform(raw_numeric))

        # data has now been re-ordered so all categorical features appear first
        categorical = np.array(sorted(categorical, reverse=True))
        processed_data = np.hstack(tuple(processed))

    else:
        processed_data = raw_data

    # one-hot encoding for categorical features (only if there exist any)
    if one_hot_encode and np.array(categorical).any():
        encoder = OneHotEncoder(categorical_features=categorical)
        processed_data = encoder.fit_transform(processed_data).toarray()
        categorical = np.zeros(processed_data.shape[1], dtype=bool)
            
    # standardize all numeric and one-hot encoded categorical features
    if standardize:
        processed_data[:, np.invert(categorical)] = scale(processed_data[:, np.invert(categorical)])
        
    print('Data pre-processing finished')
    return processed_data, categorical
