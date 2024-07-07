"""
mightypy.ml
=============
"""

from mightypy.ml._linear import (
    LinearRegression,
    LassoRegression,
    RidgeRegression,
    LogisticRegression,
    polynomial_regression,
    trend,
)


from mightypy.ml._tree import DecisionTreeClassifier, DecisionTreeRegressor

from mightypy.ml._ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from mightypy.ml._utils import moving_window_matrix, sigmoid
from mightypy.ml._recommender import ALS


__all__ = [
    "LinearRegression",
    "LassoRegression",
    "RidgeRegression",
    "LogisticRegression",
    "polynomial_regression",
    "trend",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "moving_window_matrix",
    "sigmoid",
    "ALS",
]
