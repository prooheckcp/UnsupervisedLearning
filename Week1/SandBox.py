import numpy as np

y = np.array([1, 1, 1, 1])
p = np.array([1, 0, 1, 1])

z = y[y != p]

print(z)