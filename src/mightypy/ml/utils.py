"""
Utiltiy helper functions of Machine Learning
"""

import numpy as np


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


def moving_window_matrix(arr: np.ndarray, window: int, shift: int = 1)-> np.ndarray:
    """Create Moving Window matrix for 1D data.

    More details on this function.
    https://machinelearningexploration.readthedocs.io/en/latest/MathExploration/MovingWindow.html

    Args:
        arr (np.ndarray): input 1D array.
        window (int): window/ number of columns.
        shift (int, optional): shift count for moving. Defaults to 1.

    Returns:
        np.ndarray: transformed matrix.

    Raises:
        AssertionError: input array shape should be 1D like (m,).
        AssertionError: length of array should be greater than window size and shift.
    """

    assert len(np.shape(arr)) == 1, 'input array shape should be 1D like (m,).'
    assert arr.shape[0] > window and arr.shape[0] > shift, \
        'length of array should be greater than window size and shift.'

    frame_width = arr.shape[0] - window + 1

    new_frame = []
    for i in range(0, window):
        new_frame.append(arr[i: i+frame_width][::shift])

    return np.array(new_frame).T


if __name__ == "__main__":

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print(moving_window_matrix(a, 4, 1))
