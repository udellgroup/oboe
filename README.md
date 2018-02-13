# lowrank-automl
lowrank-automl is a data-driven approach to automated machine 
learning based on matrix factorization and Bayesian optimization.
It is written in Python and distributed under the MIT license.

### Installation

##### Dependencies
lowrank-automl requires:
* Python (>= 3.5)
* NumPy  (>= 1.8.2)
* SciPy  (>= 0.13.3)
* Scikit-Learn  (>= 0.18)
* SMAC
* Pathos

##### User Installation
lowrank-automl currently only supports building from source:
```
git clone https://github.com/udellgroup/lowrank-automl.git
cd lowrank-automl
python setup.py install
```

### Framework
lowrank-automl conducts automated model selection and 
hyperparameter tuning in two phases:

##### Offline Phase

###### Error Matrix Generation
To make use of similarities across datasets, we construct
an *error matrix* $E$ that records the performance of various
models (each column corresponds to an algorithm & hyperparameter
combination) on training datasets (each row corresponds to
a dataset). 

###### Low Rank Approximation
We summarize this error matrix by a low rank approximation
$E \approx XY$ using PCA.

##### Online Phase

###### Performance Sampling & Prediction
Given a new dataset, we sample the performance of several
models that are indicative of the performance of others. We
then use our low rank approximation to predict the performance
of other models in this dataset.

###### Hyperparameter Optimization
Once we have identified several model configurations that
we predict will perform well, we perform fine-grained 
hyperparameter optimization using Bayesian optimization.

###### Ensemble Construction
The final machine learning model is an ensemble of the
best performing models.