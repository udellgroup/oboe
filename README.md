# Oboe

In an orchestra, the oboe plays an initial note which the other instruments use to tune to the right frequency before the performance begins; this package, Oboe, can be used to provide an initialization with which to tune the hyperparameters of machine learning algorithms.

Oboe is a data-driven Python algorithmic system for automated machine
learning, and is based on matrix factorization and classical experiment design.

This system is still under developement and subjects to change.

## Installation

#### Dependencies
oboe requires:
* Python (>= 3.5)
* NumPy  (>= 1.8.2)
* SciPy  (>= 0.13.3)
* Scikit-Learn  (>= 0.18)
* multiprocessing

#### User Installation
This part is currently under developement; an example for code usage is in the `example` folder.

## Usage

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


### Online Phase
Please refer to the example in the `example` folder.

