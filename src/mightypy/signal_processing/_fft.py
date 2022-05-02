"""
Signal Processing module.


author : Nishant Baheti <nishantbaheti.it19@gmail.com>
"""

import numpy as np

class FFTMixins:
    @staticmethod
    def psd(f_hat, l_signal):
        """Power Spectral Density."""
        return ((f_hat * np.conjugate(f_hat)) / l_signal).real  # type: ignore

    @staticmethod
    def magnitude(f_hat, l_signal):
        """Magnitude."""
        return (np.abs(f_hat) / l_signal).real  # type: ignore


class DenoiseFFT(FFTMixins):
    """
    Denoise signals with Fast Fourier method.

    Args:
        method (str): method. accepted values are psd, mag.
        threshold (float): threshold for cleanup.

    References:
        https://machinelearningexploration.readthedocs.io/en/latest/MathExploration/SignalProcessingFFT.html

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from mightypy.make import sine_wave_from_timesteps
        >>> time_step = 0.001
        >>> wave1, time1, freqs1 = sine_wave_from_timesteps(signal_freq=50, time_step=time_step)
        >>> wave2, time2, freqs2 = sine_wave_from_timesteps(signal_freq=70, time_step=time_step)
        >>> original_signal = wave1 + wave2
        >>> N = len(original_signal)
        >>> noisy_signal = original_signal + 2.5 * np.random.randn(N) + 2.8 * np.random.randn(N)  # adding random noise here
        >>> model = DenoiseFFT('psd', 100)
        >>> cleaned_signal = model.transform(noisy_signal)
        >>> plt.plot(original_signal, label='original')
        >>> plt.plot(noisy_signal, label='noisy')
        >>> plt.plot(cleaned_signal, label='cleaned')
        >>> plt.legend(loc='best')
        >>> plt.show()
    """

    def __init__(self, method: str, threshold: float) -> None:
        assert method.lower() in ('psd', 'mag'), "denoise signal method should be in psd, mag."
        self._method = method.lower()
        self._threshold = threshold
        super().__init__()

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Perform denoising operation on signal.

        Args:
            signal (np.ndarray): signal.

        Returns:
            np.ndarray: cleaned signal.
        """
        self.signal = signal
        self._l_signal = len(self.signal)
        self._f_hat = np.fft.fft(self.signal, self._l_signal)
        
        if self._method == 'psd':
            x = super().psd(self._f_hat, self._l_signal)
        else:
            x = super().magnitude(self._f_hat, self._l_signal)

        above_thresh_flag = x > self._threshold
        cleaned_f_hat = self._f_hat * above_thresh_flag

        return np.fft.ifft(cleaned_f_hat).real  # type: ignore


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mightypy.make import sine_wave_from_timesteps

    time_step = 0.001
    wave1, time1, freqs1 = sine_wave_from_timesteps(signal_freq=50, time_step=time_step)
    wave2, time2, freqs2 = sine_wave_from_timesteps(signal_freq=70, time_step=time_step)
    original_signal = wave1 + wave2

    N = len(original_signal)

    noisy_signal = original_signal + 2.5 * np.random.randn(N) + 2.8 * np.random.randn(N)  # adding random noise here

    model = DenoiseFFT('psd', 100)
    cleaned_signal = model.transform(noisy_signal)

    plt.plot(original_signal, label='original')
    plt.plot(noisy_signal, label='noisy')
    plt.plot(cleaned_signal, label='cleaned')
    plt.legend(loc='best')
    plt.show()

    model = DenoiseFFT('mag', 0.2)
    cleaned_signal = model.transform(noisy_signal)

    plt.plot(original_signal, label='original')
    plt.plot(noisy_signal, label='noisy')
    plt.plot(cleaned_signal, label='cleaned')
    plt.legend(loc='best')
    plt.show()
