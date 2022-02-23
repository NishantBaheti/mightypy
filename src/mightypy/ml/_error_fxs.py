"""
Cost functions of Machine Learning
"""

import numpy as np


def calculate_mse_cost(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Calculate error for regression model with mean squared error

    Args:
        y_pred (np.ndarray): predicted y value, y^.
        y (np.ndarray): actual y value. 

    Returns:
        float: mean squared error cost.
    """
    residual = y_pred - y
    diff_squared = np.square(residual)
    mse_cost = np.mean(diff_squared) / 2
    return float(mse_cost)


def calculate_entropy_cost(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Calculate entropy error for classification model

    Args:
        y_pred (np.ndarray): predicted y value, y^.
        y (np.ndarray): actual y value.

    Returns:
        float: entorpy error cost.
    """

    part_1 = y * np.log(y_pred)

    part_2 = (1 - y) * np.log(1 - y_pred)

    cost = (-1 / y_pred.shape[0]) * np.sum(part_1 + part_2)
    return cost
