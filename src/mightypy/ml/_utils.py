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


def moving_window_matrix(arr: np.ndarray, window: int, lag: int = 1)-> np.ndarray:
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

    new_frame_width = int(np.ceil(frame_width/ lag))
    new_frame = np.empty(shape=(window, new_frame_width))
    for row in range(0, window):
        new_frame[row] = arr[row: row+frame_width][::lag]

    return new_frame.T


if __name__ == "__main__":

    a = np.random.rand(100)

    print(moving_window_matrix(a, 20, 2))
