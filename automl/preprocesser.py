"""
Preprocess datasets.
"""

import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer


def DataPreprocessing(data_numeric, categorical, bool_imputer=True, bool_standardization=True, bool_onehotencoder=True):
    """
    Preprocess one dataset.

    Args:
        data_numeric (np.ndarray): Raw features of the n-by-d dataset, without indices and headings.
        categorical (np.ndarray): A d-dimensional Boolean array indicating whether each raw feature is categorical or not.
        bool_imputer (Boolean): whether to impute missing entries or not.
        bool_standardization (Boolean): whether to standardize each feature or not.
        bool_onehotencoder (Boolean): whether to use One Hot Encoding to preprocess categorical features or not.

    """
    #whether to impute missing entries
    if bool_imputer:
        # whether there exist categorical features
        bool_cat = bool(np.sum(np.isfinite(np.where(np.asarray(categorical)==True))))
        # whether there exist noncategorical features
        bool_noncat = bool(np.sum(np.isfinite(np.where(np.asarray(categorical)==False))))
        
        
        if bool_cat:
            # categorical features
            data_numeric_cat = data_numeric[:, categorical]
            # impute missing entries in categorical features using the most frequent number
            imp_cat = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=False)
            imp_cat.fit(data_numeric_cat)
            data_numeric_cat = imp_cat.transform(data_numeric_cat)
            # number of categorical features
            num_cat = data_numeric_cat.shape[1]
        
        
        if bool_noncat:            
            #noncategorical features
            data_numeric_noncat = data_numeric[:,np.invert(categorical)]
            #impute missing entries in non-categorical features using mean
            imp_noncat = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
            imp_noncat.fit(data_numeric_noncat)
            data_numeric_noncat = imp_noncat.transform(data_numeric_noncat)
            #number of noncategorical features
            num_noncat = data_numeric_noncat.shape[1]
        
        #whether there exist both categorical and noncategorical features
        if bool_cat*bool_noncat:
            
            data_numeric = np.concatenate((data_numeric_cat, data_numeric_noncat), axis=1)
            categorical = [True for i in range(num_cat)] + [False for i in range(num_noncat)]
        
        #whether there are only categorical features
        elif bool_cat*(not bool_noncat):
            data_numeric = data_numeric_cat
            categorical = [True for i in range(num_cat)]
        
        #whether there are only noncategorical features
        elif (not bool_cat)*bool_noncat:
            data_numeric = data_numeric_noncat
            categorical = [False for i in range(num_noncat)]

    # OneHotEncoding for categorical features
    if bool_onehotencoder:
        
        #check if there exist categorical features
        if np.sum(np.isfinite(np.where(np.asarray(categorical) == True))):
            enc=OneHotEncoder(categorical_features = categorical)
            enc.fit(data_numeric)
            data_numeric = enc.transform(data_numeric).toarray()
            
    # Standardization of all features
    if bool_standardization:
        if bool_OneHotEncoder:
            data_numeric = scale(data_numeric)
        
        #check if there exist numerical features
        elif np.sum(np.isfinite(np.where(np.asarray(categorical) == False))):
            data_numeric[:,np.invert(categorical)] = scale(data_numeric[:,np.invert(categorical)])

    print("DataPreprocessing finished")
    return data_numeric, categorical
