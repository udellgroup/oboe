import numpy as np
import os
import pandas as pd
import pickle
import tensorly as tl
import openml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def initialize_runtime_predictor(runtime_matrix, runtimes_index, model_name='LinearRegression', load=True, save=False, method='tensoroboe', verbose=False):
    assert load != save, "Runtime predictor error: must either load previously initialized runtime predictors, or save the initialized runtime predictors at this time!"
    
    if method == 'oboe':
        defaults_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'defaults/Oboe')
    elif method == 'tensoroboe':
        defaults_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'defaults/TensorOboe')
    
    if load:
        if verbose:
            print("Loading saved runtime predictors ...")
            
        with open(os.path.join(defaults_path, 'runtime_predictors.pkl'), 'rb') as handle:
            model = pickle.load(handle)
            
    elif save: 
        if verbose:
            print("Initializing runtime predictors and save to the defaults path ...")            
        
        try:
            dataset_sizes = pd.read_csv(os.path.join(defaults_path, 'dataset_sizes.csv'), index_col=0)
            sizes_index = np.array(dataset_sizes.index)
            sizes = dataset_sizes.values
        except FileNotFoundError:
            sizes_index = []
            sizes = []
        if runtime_matrix is None:
            runtime_tensor = np.float64(np.load(os.path.join(defaults_path, 'runtime_tensor_f16_compressed.npz'))['a'])
            runtime_matrix = tl.unfold(runtime_tensor, mode=0)

        if runtimes_index is None:
            with open(os.path.join(defaults_path, 'training_index.pkl'), 'rb') as handle:
                runtimes_index = pickle.load(handle)
#             with open(os.path.join(defaults_path, 'configs_tensor.pkl'), 'rb') as handle:
#                 configs_tensor = pickle.load(handle)
#             configs = tl.unfold(configs_tensor, mode=0)           
#             runtimes_index = [eval(configs[i, 0])['dataset'] for i in range(configs.shape[0])]

        model = RuntimePredictor(3, sizes, sizes_index, np.log(runtime_matrix), runtimes_index, model_name=model_name)

        with open(os.path.join(defaults_path, 'runtime_predictors.pkl'), 'wb') as file:
            pickle.dump(model, file)
    return model   
    
    
def predict_runtime(size, runtime_matrix=None, runtimes_index=None, saved_model=None, model=None, model_name='LinearRegression', save=False, method='tensoroboe'):
    """Predict the runtime for each model setting on a dataset with given shape.

    Args:
        size (tuple):               tuple specifying dataset size as [n_rows, n_columns]
        runtime_tensor (np.ndarray):the numpy array containing runtime.
        runtimes_index (list):      dataset indices of runtime tensor.
        saved_model (str):          path to pre-trained model; defaults to None. One of {'File', 'Class', None}.
        model (class):              saved runtime predictor; valid only when saved_model == 'Class'.
        save (bool):                whether to save pre-trained model
    Returns:
        np.ndarray:        1-d array of predicted runtimes
    """
    assert len(size) == 2, "Dataset must be 2-dimensional."
    shape = np.array(size)

    if saved_model == 'File':
        with open(saved_model, 'rb') as file:
            model = pickle.load(file)
    elif saved_model == 'Class':
        model = model
    elif method == 'tensoroboe':
        model = initialize_runtime_predictor(runtime_matrix=runtime_matrix, runtimes_index=runtimes_index, model_name=model_name, method=method, save=save)
    elif method == 'oboe':
        defaults_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'defaults/Oboe')
        try:
            dataset_sizes = pd.read_csv(os.path.join(defaults_path, 'dataset_sizes.csv'), index_col=0)
            sizes_index = np.array(dataset_sizes.index)
            sizes = dataset_sizes.values
        except FileNotFoundError:
            sizes_index = []
            sizes = []
        if runtime_matrix is None:
            runtime_matrix = pd.read_csv(os.path.join(defaults_path, 'runtime_matrix.csv'), index_col=0)
        runtimes_index = np.array(runtime_matrix.index)
        runtimes = runtime_matrix.values
        model = RuntimePredictor(3, sizes, sizes_index, np.log(runtimes), runtimes_index, model_name=model_name)
        if save:
            with open(os.path.join(defaults_path, 'runtime_predictor.pkl'), 'wb') as file:
                pickle.dump(model, file)
                
    return np.exp(model.predict(shape))


class RuntimePredictor:
    def __init__(self, degree, sizes, sizes_index, runtimes, runtimes_index, model_name='LinearRegression'):
        self.degree = degree
        self.n_models = runtimes.shape[1]
        self.model_name = model_name
        self.models = [None] * self.n_models
        self.fit(sizes, sizes_index, runtimes, runtimes_index)

    def fit(self, sizes, sizes_index, runtimes, runtimes_index):
        """Fit polynomial regression on pre-recorded runtimes on datasets."""
        # assert sizes.shape[0] == runtimes.shape[0], "Dataset sizes and runtimes must be recorded on same datasets."
        for i in set(runtimes_index).difference(set(sizes_index)):
            dataset = openml.datasets.get_dataset(int(i))
            data_numeric, data_labels, categorical, _ = dataset.get_data(target=dataset.default_target_attribute)
            if len(sizes) == 0:
                sizes = np.array([data_numeric.shape])
                sizes_index = np.array(i)
            else:
                sizes = np.concatenate((sizes, np.array([data_numeric.shape])))
                sizes_index = np.append(sizes_index, i)

        sizes_train = np.array([sizes[list(sizes_index).index(i), :] for i in runtimes_index])
        sizes_log = np.concatenate((sizes_train, np.log(sizes_train[:, 0]).reshape(-1, 1)), axis=1)
        sizes_train_poly = PolynomialFeatures(self.degree).fit_transform(sizes_log)

        # train independent regression model to predict each runtime of each model setting
        for i in range(self.n_models):
            runtime = runtimes[:, i]
            no_nan_indices = np.where(np.invert(np.isnan(runtime)))[0]
            runtime_no_nan = runtime[no_nan_indices]            
            
            if self.model_name == 'LinearRegression':
                sizes_train_poly_no_nan = sizes_train_poly[no_nan_indices]
                self.models[i] = LinearRegression().fit(sizes_train_poly_no_nan, runtime_no_nan)
            elif self.model_name == 'KNeighborsRegressor':
                sizes_train_no_nan = sizes_train[no_nan_indices]
                def metric(a, b):
                    coefficients = [1, 100]
                    return np.sum(np.multiply((a - b) ** 2, coefficients))
                        
                def weights(distances):
                    return distances

                neigh = KNeighborsRegressor(n_neighbors=5, metric=metric, weights=weights)
                self.models[i] = neigh.fit(sizes_train_no_nan, runtime_no_nan)
            # self.models[i] = Lasso().fit(sizes_train_poly, runtime)

    def predict(self, size):
        """Predict runtime of all model settings on a dataset of given size.
        
        Args:
            size(np.array): Size of the dataset to fit runtime onto.
        Returns:
            predictions (np.array): The predicted runtime.
        """
        if self.model_name == 'LinearRegression':
            size_test = np.append(size, np.log(size[0]))
            size_test_poly = PolynomialFeatures(self.degree).fit_transform([size_test])
            predictions = np.zeros(self.n_models)
            for i in range(self.n_models):
                predictions[i] = self.models[i].predict(size_test_poly)[0]
    
        elif self.model_name == 'KNeighborsRegressor':
            predictions = np.zeros(self.n_models)
            for i in range(self.n_models):
                predictions[i] = self.models[i].predict(np.array(size).reshape(1, -1))[0]
        
#        # TO BE REMOVED: sanity check
#
#        size_check = (1000, 10)
#        size_check = np.append(size, np.log(size[0]))
#        size_check_poly = PolynomialFeatures(self.degree).fit_transform([size_check])
#        print(size_check_poly)
#        for i in range(self.n_models):
#            print(self.models[i].predict(size_check_poly)[0])

        return predictions