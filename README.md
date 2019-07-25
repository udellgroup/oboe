# Oboe

In an orchestra, the oboe plays an initial note which the other instruments use to tune to the right frequency before the performance begins; this package, Oboe, is an automated machine learning/model selection system that uses collaborative filtering to find good models for supervised learning tasks within a user-specified time limit. Further hyperparameter tuning can be performed afterwards.

Oboe is based on matrix factorization and classical experiment design. For a complete description, refer to our paper at KDD 2019: [OBOE: Collaborative Filtering for AutoML Model Selection](https://arxiv.org/abs/1808.03233).

This system is still under developement and subjects to change.

## Installation

#### Dependencies
oboe requires:
* Python (>= 3.5)
* numpy  (>= 1.8.2)
* scipy  (>= 0.13.3)
* pandas (>=0.22.0)
* scikit-learn  (>= 0.18)
* multiprocessing (>=0.70.5)
* OpenML (>=0.7.0)
* mkl (>=1.0.0)
* re
* os
* json
* util

#### User Installation
This part is currently under development; an example for code usage is in the `example` folder. The package will be pip installable in the future.

## Usage

### Online Phase (AutoML model selection)
Given a new dataset, we want to select promising models and hyperparameters. Denote features and labels of the training set as `x_train` and `y_train`, and features of the test set as `x_test`, a short example of training and testing is
```
from auto_learner import AutoLearner
m = AutoLearner(runtime_limit=20) #set the time limit for model fitting to be 20 seconds
m.fit(x_train, y_train)
m.predict(x_test)
```
Additional arguments can be applied to customize the `AutoLearner` instance, including:
1. Basics
* p_type (str): Problem type, which is one of {'classification', 'regression'}. By default, 'classification'.
* verbose (Boolean): Whether or not to generate print statements that showcase the progress. By default, false.
* n_cores (int): Maximum number of CPU cores to use. The default value 'None' means no limit, i.e., up to all the CPU cores of the machine.
* runtime_limit (int): Maximum runtime for AutoLearner fitting, in seconds. By default, 512 seconds as the timeout limit.
* scalarization (str): Scalarization of the covariance matrix for mininum variance selection. One of {'D', 'A', 'E'}. 'D', as default, enjoys best performance and fastest speed in practice.
* build_ensemble (Boolean): Whether to build an ensemble of promising models.
* stacking_alg (str): The method used for ensemble construction. One of {'greedy', 'stacking'}. By default, 'greedy'.
* dataset_ratio_threshold (float): The threshold of dataset ratio for dataset subsampling, if the training set is tall and skinny (i.e., number of data points much larger than number of features).

2. Advanced customization
* algorithms (list): A list of algorithm types to be considered, in strings, e.g. ['KNN', 'lSVM']. By default, all the algorithms in the error matrix. The supported classification algorithms are: 'AB' (Adaboost), 'DT' (decision tree), 'ExtraTrees' (extra trees), 'GBT' (gradient boosting), 'GNB' (Gaussian naive Bayes), 'KNN' (kNN), 'Logit' (logistic regression), 'MLP' (multilayer perceptron), 'Perceptron' (perceptron), 'RF' (random forest), 'kSVM' (kernel SVM), 'lSVM' (linear SVM).
* hyperparameters (dict): A nested dict of hyperparameters to be considered. By default, all the model hyperparameters in the error matrix.
* error_matrix (DataFrame): Error matrix to use for imputation, includes indices and headers. The one in `defaults` folder is used by default.
* runtime_matrix (DataFrame): Runtime matrix to use for runtime prediction, includes indices and headers. The one in `defaults` folder is used by default.
* new_row (np.ndarray): Predicted row of error matrix; corresponds to the new dataset. By default, 'None'.
* selection_method (str): Method of selecting entries of new row to sample. One of {'min_variance', 'qr'}. 'min_variance' corresponds to the selection approach via classic experiment design; 'qr' selects the pivot columns in the error matrix and thus does not provide the functionality of maximizing performance within given runtime budget. By default, 'min_variance'.
* runtime_predictor (str): Model for runtime prediction. One of {'LinearRegression', 'KNeighborsRegressor'}. By default, 'LinearRegression'. Dataset sizes (number of data points and number of features) are used as feature vectors for both runtime predictor models.

For executable and more detailed examples, please refer to the `example` folder.

### Offline Phase

##### Error Matrix Generation
Please refer to `examples/error_matrix_generation` for an error matrix generation example.