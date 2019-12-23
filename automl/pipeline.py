"""
Parent class for all ML models.
"""

import numpy as np
import util
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, train_test_split
from model import Model
from sklearn.pipeline import Pipeline

# data cleaning
from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA

# encoder
from sklearn.preprocessing import OneHotEncoder

# standardizer
from sklearn.preprocessing import StandardScaler

# dimensionality reducer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2

# classifier
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


RANDOM_STATE = 0

class PipelineObject:
    """An object representing a machine learning pipeline.

    Attributes:
        p_type (str):           Either 'classification' or 'regression'.
        model (object):         A scikit-learn pipeline object.
        fitted (bool):          Whether or not the model has been trained.
        verbose (bool):         Whether or not to generate print statements when fitting complete.
    """

    def __init__(self, config, index=None, p_type='classification', verbose=False):
        self.p_type = p_type
        
        # this ensures the order of pipeline steps is correct
        self.pipeline_steps = ['imputer', 'encoder', 'standardizer', 'dim_reducer', 'estimator']
        assert set(config.keys()) == set(self.pipeline_steps), "Pipeline steps not correct!"
        
        self.index = index
        self.config = config
        self.model = self._instantiate()
        self.cv_error = np.nan
        self.cv_predictions = None
        self.sampled = False
        self.fitted = False
        self.verbose = verbose

    def _instantiate(self):
        """
        Creates a scikit-learn object of the corresponding pipeline.
        """
        pipeline = []
        for step in self.pipeline_steps:
            alg = self.config[step]['algorithm']
            if alg is not None:
                step_object = eval(alg)(**self.config[step]['hyperparameters'])
            else:
                step_object = None
            
            pipeline.append((step, step_object))
        
        pipeline = Pipeline(pipeline)
        return pipeline

    def fit(self, x_train, y_train):
        """Fits the pipeline on training data.

        Args:
            x_train (np.ndarray):   Features of the training dataset.
            y_train (np.ndarray):   Labels of the training dataset.
        """
        
        self.model.fit(x_train, y_train)
        self.fitted = True
        if self.verbose:
            print("Pipeline fitting completed.")

    def predict(self, x_test):
        """Predicts labels on a new dataset.

        Args:
            x_test (np.ndarray): Features of the test dataset.

        Returns:
            np.array: Predicted features of the test dataset.
        """
        return self.model.predict(x_test)

    def kfold_fit_validate(self, x_train, y_train, n_folds=5, random_state=None):
        """Performs k-fold cross validation on a training dataset. 

        Args:
            x_train (np.ndarray): Features of the training dataset.
            y_train (np.ndarray): Labels of the training dataset.
            n_folds (int):        Number of folds to use for cross validation.

        Returns:
            float: Mean of k-fold cross validation error.
            np.ndarray: Predictions on the training dataset from cross validation.
        """
        y_predicted = np.empty(y_train.shape)
        cv_errors = np.empty(n_folds)
        kf = StratifiedKFold(n_folds, shuffle=True, random_state=random_state)

        for i, (train_idx, test_idx) in enumerate(kf.split(x_train, y_train)):
            x_tr = x_train[train_idx, :]
            y_tr = y_train[train_idx]
            x_te = x_train[test_idx, :]
            y_te = y_train[test_idx]

            model = self._instantiate()
            if len(np.unique(y_tr)) > 1:
                model.fit(x_tr, y_tr)
                y_predicted[test_idx] = model.predict(x_te)
            else:
                y_predicted[test_idx] = y_tr[0]
            cv_errors[i] = util.error(y_te, y_predicted[test_idx], p_type=self.p_type)

        self.cv_error = cv_errors.mean()
        self.cv_predictions = y_predicted
        self.sampled = True
        if self.verbose:
            print("{}-fold cross-validation completed.".format(n_folds))
        return self.cv_error, self.cv_predictions

    

class Ensemble(Model):
    """An object representing an ensemble of machine learning pipelines.

    Attributes:
        p_type (str):           Either 'classification' or 'regression'.
        algorithm (str):        Algorithm type (e.g. 'Logit').
        hyperparameters (dict): Hyperparameters (e.g. {'C': 1.0}).
        model (object):         A scikit-learn object for the model.
    """

    def __init__(self, p_type, algorithm, hyperparameters={}):
        super().__init__(p_type, algorithm, hyperparameters)
        self.candidate_learners = []
        self.base_learners = []
        self.second_layer_features = None

    def select_base_learners(self, y_train, fitted_base_learners):
        """Select base learners from candidate learners based on ensembling algorithm.
        """
        cv_errors = np.array([p.cv_error for p in self.candidate_learners])
        print("cv errors: {}".format(cv_errors))
        
        # greedy ensemble forward selection
        assert self.algorithm in {'greedy', 'stacking'}, "The ensemble selection method must be either greedy forward selection (by Caruana et al.) or stacking."
        if self.algorithm == 'greedy':
            x_tr = ()
            # initial number of models in ensemble
            n_initial = 3
            for i in np.argsort(cv_errors)[:n_initial]:
                x_tr += (self.candidate_learners[i].cv_predictions.reshape(-1, 1), )
                if fitted_base_learners is None:
                    pre_fitted = None
                else:
                    pre_fitted = fitted_base_learners[self.candidate_learners[i].index]
                if pre_fitted is not None:
                    self.base_learners.append(pre_fitted)
                else:
                    self.base_learners.append(self.candidate_learners[i])

            x_tr = np.hstack(x_tr)
            candidates = list(np.argsort(cv_errors))
            error = util.error(y_train, mode(x_tr, axis=1)[0], self.p_type)

            while True:
                looped = True
                for i, idx in enumerate(candidates):
                    slm = np.hstack((x_tr, self.candidate_learners[i].cv_predictions.reshape(-1, 1)))
                    err = util.error(y_train, mode(slm, axis=1)[0], self.p_type)
                    if err < error:
                        error = err
                        x_tr = slm
                        if fitted_base_learners is None:
                            pre_fitted = None
                        else:
                            pre_fitted = fitted_base_learners[self.candidate_learners[i].index]
                        if pre_fitted is not None:
                            self.base_learners.append(pre_fitted)
                        else:
                            self.base_learners.append(self.candidate_learners[i])
                        looped = False
                        break
                if looped:
                    break
            self.second_layer_features = x_tr
        elif self.algorithm == 'stacking':
            self.base_learners = self.candidate_learners
            x_tr = [p.cv_predictions.reshape(-1, 1) for p in self.candidate_learners]
            self.second_layer_features = np.hstack(tuple(x_tr))

    def fit(self, x_train, y_train, runtime_limit=None, fitted_base_learners=None):
        """Add pipelines to the ensemble and fit the ensemble on training data.

        Args:
            x_train (np.ndarray):        Features of the training dataset.
            y_train (np.ndarray):        Labels of the training dataset.
            fitted_base_learners (list): A list of already fitted models.
            
        Args to be implemented:
            runtime_limit (float):       Maximum runtime to be allocated to fitting.
        """
        self.select_base_learners(y_train, fitted_base_learners)
        # TODO: parallelize training over base learners
        for pipeline in self.base_learners:
            if not pipeline.fitted:
                pipeline.fit(x_train, y_train)
        if self.algorithm != 'greedy':
            self.model.fit(self.second_layer_features, y_train)
        self.fitted = True

    def refit(self, x_train, y_train):
        """Fit ensemble model on training data with base learners already added and unchanged.
            
        Args:
            x_train (np.ndarray):        Features of the training dataset.
            y_train (np.ndarray):        Labels of the training dataset.
            
        Args to be implemented:
            runtime_limit (float):       Maximum runtime to be allocated to fitting.
        """
        # TODO: parallelize training over base learners
        for pipeline in self.base_learners:
            if not pipeline.fitted:
                pipeline.fit(x_train, y_train)
        if self.algorithm == 'stacking':
            self.model.fit(self.second_layer_features, y_train)

    def predict(self, x_test):
        """Generate predictions of the ensemble model on test data.

        Args:
            x_test (np.ndarray): Features of the test dataset.
        Returns:
            np.array: Predicted labels of the test dataset.
        """
        assert len(self.base_learners) > 0, "Ensemble size must be greater than zero."

        base_learner_predictions = ()
        for pipeline in self.base_learners:
            y_predicted = np.reshape(pipeline.predict(x_test), [-1, 1])
            base_learner_predictions += (y_predicted, )
        self.x_te = np.hstack(base_learner_predictions)
        if self.algorithm == 'greedy':
            return mode(self.x_te, axis=1)[0].reshape((1, -1))
        else:
            return self.model.predict(self.x_te)

    def get_models(self):
        """Get details of the selected machine learning pipelines and the ensemble.
        """
#         base_learner_names = {}
#         for pipeline in self.base_learners:
#             if pipeline.algorithm in base_learner_names.keys():
#                 base_learner_names[model.algorithm].append(model.hyperparameters)
#             else:
#                 base_learner_names[model.algorithm] = [model.hyperparameters]
        base_learner_names = [p.config for p in self.base_learners]
    
        if self.algorithm == 'greedy':
            return {'ensemble method': 'greedy selection', 'base learners': base_learner_names}
        elif self.algorithm == 'stacking':
            ensemble_learner_name = {}
            ensemble_learner_name[self.model.algorithm] = self.model.hyperparameters
            return {'ensemble method': 'stacking', 'ensemble learner': ensemble_learner_name, 'base learners': base_learner_names}

    def get_pipeline_accuracies(self, y_test):
        """ Get prediction accuracies of each base learner when the true test labels are provided.
            
            Args:
                y_test (np.array):      True labels of the test set.
                
            Returns:
                accuracies (list):      A numerical list of individual model accuracies on the test set.
        """
        accuracies = []
        for i in range(self.x_te.shape[1]):
            accuracies.append(util.error(y_test, self.x_te[:, i], self.p_type))
        return accuracies


class Model_collection(Ensemble):
    """An object representing a collection of individual machine learning pipelines.
        
        Attributes:
            p_type (str):           Either 'classification' or 'regression'.
    """
    def __init__(self, p_type):
        super().__init__(p_type=p_type, algorithm=None, hyperparameters=None)

    def select_base_learners(self):
        """ Set inidividual learners to be all the learners added to the collection.
        """
        self.base_learners = self.candidate_learners

    def fit(self, x_train, y_train, runtime_limit=None, fitted_base_learners=None):
        """ Fit inidividual learners in the pipeline collection on training dataset.
        
        Args:
            x_train (np.ndarray):        Features of the training dataset.
            y_train (np.ndarray):        Labels of the training dataset.
        """
        self.select_base_learners()
        super().refit(x_train=x_train, y_train=y_train)
        self.fitted = True

    def predict(self, x_test):
        """Generate predictions of the individual learners on test data.
            
        Args:
            x_test (np.ndarray): Features of the test dataset.
            
        Returns:
            np.ndarray: A 2-dimensional array containing predicted labels of the test dataset. Each column corresponds to the predictions of one single base learner.
        """
        assert len(self.base_learners) > 0, "Ensemble size must be greater than zero."
        
        base_learner_predictions = ()
        for pipeline in self.base_learners:
            y_predicted = np.reshape(pipeline.predict(x_test), [-1, 1])
            base_learner_predictions += (y_predicted, )
        # concatenation of predictions of each base learner
        self.x_te = np.hstack(base_learner_predictions)
        return self.x_te

    def get_models(self):
        """Get details of the selected machine learning pipelines.
        """
        return [p.config for p in self.base_learners]











