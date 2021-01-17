from numpy import *
from numpy.linalg import *


##########
# 绕坐标轴旋转  得到旋转矩阵
# 齐次坐标!
# 这些推到很简单 绕那个轴,哪个轴不变. 极坐标表示出旋转角即可
def rot_x(theta):
    c = cos(theta)
    s = sin(theta)
    rx = array([[1, 0, 0, 0],
                [0, c, -1*s, 0],
                [0, s, c, 0],
                [0, 0, 0, 1]])
    return rx


def rot_y(theta):
    c = cos(theta)
    s = sin(theta)
    ry = array([[c, 0, s, 0],
                [0, 1, 0, 0],
                [-1*s, 0, c, 0],
                [0, 0, 0, 1]])
    return ry


def rot_z(theta):
    c = cos(theta)
    s = sin(theta)
    rz = array([[c, -1*s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    return rz

# 复合旋转,可以完成计算之后使用 cs记号简化三角函数计算


###########
# 绕任意轴旋转
# 这种方法:给定旋转轴与旋转角,得到旋转矩阵?
# 在PPF里面如何实现?


if __name__ == '__main__':
    print()
    ang2rad = pi/180
    a = array([1, 0, 0, 1])  # 齐次坐标
    rz = rot_z(45*ang2rad)
    # print(rz)

    rot_vec = dot(rz, a)
    # print('rot_vec:', rot_vec)

    # 变回来:
    # inv_rz = inv(rz)  # 正交矩阵! 转置=逆矩阵
    inv_rz = rz.T  # 正交矩阵! 转置=逆矩阵  但是损失了一点精度
    inv_rot_vec = dot(inv_rz, rot_vec)
    print('inv_rot_vec:', inv_rot_vec)

    ####### 上面测试了基本的旋转
