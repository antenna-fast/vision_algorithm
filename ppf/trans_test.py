from numpy import *
from scipy.spatial.transform import Rotation as R


a = array([1, 0, 0])

r_vec = R.from_rotvec(array([0, 0, 90]) * 180 / pi)
r_mat = r_vec.as_matrix()

b = dot(r_mat, a)
print(b)
