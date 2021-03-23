# Examples
Examples of how to use the Oboe systems.

1. `error_matrix_generation`

This directory has an example of the offline error matrix generation in Oboe. More details will follow on how to collect pipeline performance for TensorOboe.

2. `classification_by_Oboe.ipynb`

This Jupyter notebook shows examples of using Oboe to do classification. The schematic way to initialize an Oboe classification `AutoLearner` is `m = AutoLearner(p_type='classification', method='Oboe', **kwargs)`.

3. `classification_by_TensorOboe.ipynb`

This Jupyter notebook shows examples of using TensorOboe to do classification. The schematic way to initialize a TensorOboe classification `AutoLearner` is `m = AutoLearner(p_type='classification', method='TensorOboe', **kwargs)`.