from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R


# 给出新的坐标系三个坐标轴（向量），问：如何求出相对于原坐标系之间的旋转变换
# 答：变换就是正交基

base_1 = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
r_vec = R.from_rotvec(array([0, 0, 90]) * pi / 180)
# r_vec = R.from_rotvec(array([0, 0, 90]))
r_mat = r_vec.as_matrix()

# print('r_mat:\n', r_mat)

base_3 = dot(r_mat, base_1)
print('base_3:\n', base_3)

# 变换后的坐标轴 是在原来坐标系中的表示
# 其他情况下,求出的法向量也是相对于原坐标系

# 要把变换后的坐标系变换到 oxyz坐标系(基础坐标系),只需要求逆
#
r_mat_inv = inv(r_mat)
base_3_inv = dot(r_mat_inv, base_3)
print('bas_3_inv:\n', base_3_inv)  # 又变成了原来的
