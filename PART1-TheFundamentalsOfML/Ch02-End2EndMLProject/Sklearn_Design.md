# Scikit-Learn API Design
A remarkably well designed API. The main <a href="https://arxiv.org/pdf/1309.0238v1.pdf" target="_blank">design principles</a> are:

## Consistency
All objects share a consistent and simple interface:

### Estimators
* Object that can estimate some parameters based on a dataset
* Example, imputer
* Estimation is performed by `fit()` method
* Any other param other than dataset, used to guide the estimation, is called hyperparameter. They are set as instance variables.

### Transformers
* Estimators that can also transform a dataset.
* Example, imputer.
* Transformation is performed by `transform()` method with the dataset to be transformed as parameter.
* Transformation relies on learned params from call to `fit()` method.
* Perform param estimation and transformation in one step on dataset using - `fit_transform()` method

### Predictors
* Estimators that are capable of making predictions.
* Example, LinearRegression
* It has a `predict()` method that takes new instances and returns predictions.
* It also has a `score()` method that calculates score of the prediction based on type of task - Regression or Classification.

## Inspection
* All hyperparameters of the estimators are accessible directly via public instance variables.
* Example, imputer.strategy
* All the estimator's learned params are also accessible via public variables with an underscore suffix.
* Example, imputer.statistics_

## Nonproliferation of classes
* Datasets are represented as numPy arrays or SciPy sparse matrices, instead of homemade classes.
* Hyperparameters are just regular Python strings or numbers.

## Composition
* Existing building blocks are re-used as much as possible
* Example, It is easy to create `Pipeline` from a long sequence of transformers followed by a final estimator.

## Sensible Defaults
* Reasonable default values are given to params, making it easy to create a baseline working system quickly.
