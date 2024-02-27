from mightypy.signal_processing import PSDDenoiser
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
fs = 10e3
N = 100
amp = 2 * np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
X = amp * np.sin(2 * np.pi * freq * time)
X += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
denoiser = PSDDenoiser()
cleaned_signal = denoiser.transform(X)
plt.plot(X, label="noisy")
plt.plot(cleaned_signal, label="cleaned")
plt.title(f"Threshold : {denoiser.threshold}")
plt.legend(loc="best")
plt.savefig("./test.png")