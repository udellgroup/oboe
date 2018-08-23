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

`cvxpy` is required unless `scipy.optimize` is used as convex

#### User Installation
This part is currently under development; an example for code usage is in the `example` folder. The package will be `setup.py` installable in the future.

## Usage

### Online Phase (AutoML initialization)
Given a new dataset, we want to select promising models and hyperparameters. Denote features and labels of the training set as `x_train` and `y_train`, and features of the test set as `x_test`, a short example of training and testing is
```
from auto_learner import AutoLearner
m = AutoLearner()
m.fit(x_train, y_train)
m.predict(x_test)
```
Additional parameters can be applied to customize `AutoLearner`. For executable and more detailed examples, please refer to the `example` folder.

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




