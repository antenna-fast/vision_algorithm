from numpy import *
from numpy.linalg import *


a = array([1, 0])
b = array([1, 1])
c = array([-1, 1])
d = array([1, -1])
e = array([-1, -1])

# cos_ang = dot(a, b) / (norm(a)*norm(b))
# cos_ang = dot(a, c) / (norm(a)*norm(c))
cos_ang = dot(a, d) / (norm(a)*norm(d))
# cos_ang = dot(a, e) / (norm(a)*norm(e))

print(cos_ang)
