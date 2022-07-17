import numpy as np

x = np.arange(1, 7, 1)
y = np.random.rand(6)
degree = 1

print(np.polynomial.Polynomial.fit(x, y, deg=degree).coef)