"""
Dataset generation module.

author : Nishant Baheti <nishantbaheti.it19@gmail.com>
"""

import numpy as np
from typing import Tuple


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Create 2D data rotation matrix.

    Reference article
    ------------------
    
    https://en.wikipedia.org/wiki/Rotation_matrix

    Args:
        theta (float): angle for rotation.

    Returns:
        np.ndarray: rotation matrix.
    """

    mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return mat


def spiral_data(data_limit: int = 30, n_classes: int = 2,
                         n_samples_per_class=300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spiral data for classification problem.

    Args:
        data_limit (int, optional): range of data. Defaults to 30.
        n_classes (int, optional): number of classes for classification. Defaults to 2.
        n_samples_per_class (int, optional): number of samples per classes. Defaults to 300.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X,y.
    """

    theta = np.pi * (2 / n_classes)
    rotation_mat = rotation_matrix_2d(theta=theta)

    features = [np.array([
        [
            (np.cos(i/8) * i) + np.random.randn(), (np.sin(i/8) * i) + np.random.randn()
        ] for i in np.linspace(2, data_limit, n_samples_per_class)
    ])]
    target = [np.ones(shape=(n_samples_per_class, 1), dtype=np.int32) * 0]

    for i in range(0, n_classes-1):
        features.append(features[-1] @ rotation_mat)
        target.append(np.ones(shape=(n_samples_per_class, 1), dtype=np.int32) * i+1)
    X = np.vstack(features)
    y = np.vstack(target)
    return X, y

def sine_wave_from_sample(
        n_samples: int, signal_freq: float, n_cycles: int = 10, amplitude: int = 1,
        amp_shift: int = 0, phase_shift: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sine wave generation with number of samples and signal frequency

    Reference:
        https://machinelearningexploration.readthedocs.io/en/latest/MathExploration/Fourier.html#Sine-wave

    Args:
        n_samples (int): number of samples.
        signal_freq (float): signal frequency.
        n_cycles (int, optional): number of cycles. Defaults to 10.
        amplitude (int, optional): signal amplitude. Defaults to 1.
        amp_shift (int, optional): amplitude shift. Defaults to 0.
        phase_shift (int, optional): phase shift. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: signal wave, time, freq.
    """
    sampling_freq = (n_samples * signal_freq) / n_cycles
    # n_samples = int((sampling_freq / signal_freq) * n_cycles)

    t_step = 1 / sampling_freq
    time = np.linspace(0, (n_samples-1) * t_step, n_samples)

    f_step = sampling_freq / n_samples
    freq = np.linspace(0, (n_samples-1) * f_step, n_samples)

    wave = amp_shift + amplitude * \
        np.sin((2 * np.pi * signal_freq * time) + phase_shift)

    return wave, time, freq

def sine_wave_from_timesteps(
        signal_freq: float, time_step: float, amplitude: int = 1, amp_shift: int = 0,
        phase_shift: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sine wave generation with time steps and signal frequency

    Args:
        signal_freq (float): singal frequency.
        time_step (float): time step.
        amplitude (int, optional): amplitude. Defaults to 1.
        amp_shift (int, optional): amplitude shift. Defaults to 0.
        phase_shift (int, optional): phase shift. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: signal wave, time, freq.
    """
    time = np.arange(0, 1, time_step)
    n = len(time)
    freq = (1/(time_step * n)) * np.arange(n)

    wave = amp_shift + amplitude * \
        np.sin((2 * np.pi * signal_freq * time) + phase_shift)

    return wave, time, freq
        
if __name__ == "__main__":

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # X, y = generate_spiral_data(n_classes=1)

    # sns.scatterplot(x=X[..., 0], y=X[..., 1], hue=y[..., 0], palette='dark')
    # plt.show()

    pass