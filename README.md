# Oboe

In an orchestra, the oboe plays an initial note which the other instruments use to tune to the right frequency before the performance begins; this package, Oboe, can be used to provide an initialization with which to tune the hyperparameters of machine learning algorithms.

Oboe is a data-driven Python algorithmic system for automated machine
learning, and is based on matrix factorization and classical experiment design. For a complete description, refer to [our paper](https://arxiv.org/abs/1808.03233).

This system is still under developement and subjects to change.

## Installation

#### Dependencies
oboe requires:
* Python (>= 3.5)
* numpy  (>= 1.8.2)
* scipy  (>= 0.13.3)
* pandas (>=0.22.0)
* scikit-learn  (>= 0.18)
* multiprocessing
* OpenML (>=0.7.0)
* mkl (>=1.0.0)
* re
* os
* json
* util

`cvxpy` is required unless `scipy.optimize` is used as the convex solver, which is default.

#### User Installation
This part is currently under development; an example for code usage is in the `example` folder. The package will be `setup.py` and pip installable in the future.

## Usage

### Online Phase (AutoML initialization)
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
* verbose (bool): Whether or not to generate print statements that showcase the progress. By default, false.
* n_cores (int): Maximum number of CPU cores to use. The default value 'None' means no limit.
* runtime_limit (int): Maximum runtime for AutoLearner fitting, in seconds. By default, 512 seconds as the timeout limit.
* scalarization (str): Scalarization of the covariance matrix for mininum variance selection. One of {'D', 'A', 'E'}. 'D', as the default value, is the best-performing and fastest scalarization method in practice.
* stacking_alg(str): The method used for ensemble construction. One of {'greedy', 'stacking'}. By default, 'greedy'.
2. Further customization
* algorithms (list): A list of algorithm types to be considered, in strings, e.g. ['KNN', 'lSVM']. By default, all the algorithms in the error matrix. 
* hyperparameters (dict): A nested dict of hyperparameters to be considered. By default, all the model hyperparameters in the error matrix.
* error_matrix (DataFrame): Error matrix to use for imputation, includes indices and headers. The one in `defaults` folder is used by default.
* runtime_matrix (DataFrame): Runtime matrix to use for runtime prediction, includes indices and headers. The one in `defaults` folder is used by default.
* new_row (np.ndarray): Predicted row of error matrix; corresponds to the new dataset. By default, 'None'.
* selection_method (str): Method of selecting entries of new row to sample. One of {'min_variance', 'qr'}. 'min_variance' corresponds to the selection approach via classic experiment design; 'qr' selects the pivot columns in the error matrix and thus does not provide the functionality of maximizing performance within given runtime budget. By default, 'min_variance'.

For executable and more detailed examples, please refer to the `example` folder.

### Offline Phase

##### Error Matrix Generation
The `generate_matrix.sh` in `automl` folder is a bash script intended for the error matrix generation. Please refer to its usage message for configurations. As an example,
```
./generate_matrix.sh -m generate results selected_datasets defaults/classification.json 90 False False
```
will run in parallel with no more than 90 processes to generate the error and runtime matrices' corresponding rows, each corresponds to a single pre-processed classification dataset in `selected_datasets` folder, and then output results into a subfolder in `results`. The models in use are specified in `defaults/classification.json`.

Then to get the error matrix in use,
```
./generate_matrix.sh -m merge <directory_of_error_matrix_rows>
```
will concatenate the rows of error and runtime matrices, respectively, into `error_matrix.csv` and `runtime_matrix.csv`.




