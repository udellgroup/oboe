from auto_learner import AutoLearner

# compute the latent factors
m = AutoLearner(p_type='classification', runtime_limit=100, load_imputed_error_tensor=False, load_saved_latent_factors=False, 
                load_saved_runtime_predictors=False, save_fitted_runtime_predictors=True, verbose=True)
