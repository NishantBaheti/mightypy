"""
Utiltiy helper functions of Machine Learning
"""

import numpy as np
from typing import Tuple


def sigmoid(val: np.ndarray) -> np.ndarray:
    """Sigmoid function

    .. math::
        f(z) = \\frac{1}{1 + e^{-z}}

    Args:
        val (ndarray): input value

    Returns:
        np.ndarray: sigmoid value
    """
    return 1 / (1 + np.exp(-val))


def moving_window_matrix(arr: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    """Create Moving Window matrix for 1D data.

    More details on this function.
    https://machinelearningexploration.readthedocs.io/en/latest/MathExploration/MovingWindow.html

    Args:
        arr (np.ndarray): input 1D array.
        window (int): window/ number of columns.
        lag (int, optional): lag count for moving. Defaults to 1.

    Returns:
        np.ndarray: transformed matrix.

    Raises:
        AssertionError: input array shape should be 1D like (m,).
        AssertionError: length of array should be greater than window size and lag.

    Example:
        >>> a = np.random.rand(100)
        >>> print(moving_window_matrix(a, 20, 2))
    """

    assert len(np.shape(arr)) == 1, 'input array shape should be 1D like (m,).'
    size = arr.shape[0]
    assert size > window and size > lag, \
        'length of array should be greater than window size and lag.'

    frame_width = size - window + 1

    new_frame_width = int(np.ceil(frame_width / lag))
    new_frame = np.empty(shape=(window, new_frame_width))
    for row in range(0, window):
        new_frame[row] = arr[row: row+frame_width][::lag]

    return new_frame.T


def polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    fit Regression line with polynomial degree.

    Args:
        x (np.ndarray): independent variable.
        y (np.ndarray): dependent variable.
        degree (int): polynomial degree.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: slope, residual, fitline.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(-10, 10)
        >>> y = x**2 + x**3
        >>> s, r, l = polynomial_regression(x, y, 3)
        >>> plt.plot(x, y, 'ko', label='original')
        >>> plt.plot(x, l, '.-',  label='regression line')
        >>> plt.legend()
        >>> plt.show()
    """
    a = np.polyfit(x, y, degree)
    slope = a[:-1]
    resid = a[-1:]
    fit_line = np.array([(x**(degree - i))*slope[i]
                     for i in range(0, degree)]).sum(axis=0) + resid
    return slope, resid, fit_line


def trend(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    get trend of the data.

    Args:
        x (np.ndarray): independent variable.
        y (np.ndarray): dependent variable.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: slope, residual, trendline. 

    Examples;
        >>> import matplotlib.pyplot as plt 
        >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> y = np.array([1, 2, 3, 3, 4, 5, 7, 10])
        >>> s, r, t = trend(x, y)
        >>> plt.plot(x, y, 'o', label='original')
        >>> plt.plot(x, t, '.-',  label='regression line')
        >>> plt.legend()
        >>> plt.show()
    """
    return polynomial_regression(x, y, 1)


if __name__ == "__main__":

    # a = np.random.rand(100)

    # print(moving_window_matrix(a, 20, 2))

    # import matplotlib.pyplot as plt

    # x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # y = np.array([1, 2, 3, 3, 4, 5, 7, 10])
    # s, r, t = trend(x, y)
    # plt.plot(x, y, 'o', label='original')
    # plt.plot(x, t, '.-',  label='regression line')
    # plt.legend()
    # plt.show()


    # x = np.arange(-10, 10)
    # y = x**2 + x**3
    # s, r, l = polynomial_regression(x, y, 3)

    # plt.plot(x, y, 'ko', label='original')
    # plt.plot(x, l, '.-',  label='regression line')
    # plt.legend()
    # plt.show()

    pass
