
from scipy import linalg
import numpy as np
from numpy import *
from numpy.linalg import *

m, n = 9, 6
a = np.random.randn(m, n) + 1.j * np.random.randn(m, n)


U, s, Vh = svd(a)  # U.shape, s.shape, Vh.shape

# print(U.T == inv(U))
# print(U.T)
# print(inv(U))

# Reconstruct the original matrix from the decomposition:

sigma = np.zeros((m, n))

for i in range(min(m, n)):
    sigma[i, i] = s[i]

a1 = np.dot(U, dot(sigma, Vh))
res = np.allclose(a, a1)
print(res)
