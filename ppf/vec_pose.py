from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R
import scipy

from transform import *


# 反对称阵
def get_anti_sym(vec):
    mat = array([[0, -1 * vec[2], vec[1]],
                 [vec[2], 0, -1 * vec[0]],
                 [-1 * vec[1], vec[0], 0]])
    return mat


# 2d里面 叉乘之后 逆时针是正的
def get_ang_2d(vec1, vec2):  # vec1 起始向量，vec2 终止向量
    dist = dot(vec1, vec2) / (norm(vec1) * norm(vec2))  # 小夹角 接近于1
    ang = arccos(dist) * 180 / pi  # 转为角度
    if ang != 180:  # 共线的时候 叉积等于0 无法使用
        ang = ang * cross(vec1, vec2)

    return ang


# a = [0, 1]
# # a = [1, 0]
# b = [1, 0]
# # b = [0, 1]
# print('get_ang_2d:', get_ang_2d(a, b))


# 轴角
# 根据叉乘 得到旋转方向  对于轴角，旋转方向与轴一致
def get_ang_3d(vec1, vec2):
    dist = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    ang = arccos(dist) * 180 / pi

    rot_axis = cross(vec1, vec2)  # 旋转轴  决定旋转方向

    return ang, rot_axis


# # a = [0, 1, 0]
# a = [1, 0, 0]
# # b = [1, 0, 0]
# b = [0, 1, 1]
# print('get_ang_3d:', get_ang_3d(a, b))


# 轴角旋转 方向由转轴方向定义  转角0-pi  转轴和叉积前后对应有关，统一即可
def get_rodrigues(v, ang, k):  # 角度制输入  OK
    """
    推导过程：
    v_p = v*k*k
    v_t = v - v_p
    v_t_rot = v_t*cos(theta) + kxv*sin(theta)
    v_rot = v_p + v_t_rot
    """
    ang = ang * pi / 180
    cos_theta = cos(ang)
    cross_kv = cross(k, v)
    v_rot = v * cos_theta + (1 - cos_theta) * dot(dot(k, v), k) + sin(ang) * cross_kv
    return v_rot


# 从轴角转换成旋转矩阵  旋转轴是单位向量
def from_rodri_to_mat(k, theta):
    theta = theta * pi / 180
    K = get_anti_sym(k)
    R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K ** 2)
    # R = eye(3)*cos(theta) + (1-cos(theta))*dot(k, k.T) + sin(theta)*K
    return R


if __name__ == '__main__':
    # 模型点对
    pt_1 = array([0, 0, 0])
    pt_2 = array([1, 0, 0])
    pt_n_1 = array([0, 1, 1])
    pt_n_1 = pt_n_1 / norm(pt_n_1)  # 法向量都是单位的
    pt_n_2 = array([0, 1, 0])
    # vec_1 = pt_2 - pt_1

    # 定义变换  只加入一些平移 法向量不变
    t_vect = array([150, -2, -8], dtype='float')
    # print('r_mat:\n', r_mat)

    # 场景点对
    # pcd_trans = dot(r_mat, pcd_trans.T).T
    pt_s_1 = pt_1 + t_vect
    pt_s_2 = pt_2 + t_vect
    pt_s_n_1 = pt_n_1
    pt_s_n_s = pt_n_2

    # Tmg:
    Tmg = -pt_1  # 参考点
    Tsg = -pt_s_1
    # print(Tsg - Tmg)

    # 将模型点和场景参考点变换到统一坐标系
    pt_m_1 = pt_1 + Tmg  # 变换后的参考点
    pt_m_2 = pt_2 + Tmg
    # print('pt_m_1, pt_m_2:', pt_m_1, pt_m_2)

    # 平移变换 场景
    pt_s_t_1 = pt_s_1 + Tsg
    pt_s_t_2 = pt_s_2 + Tsg
    # print('pt_s_t_1, pt_s_t_2:', pt_s_t_1, pt_s_t_2)

    # 平移算完了 接下来需要得到旋转
    # 基于alpha进行投票

    x_axis = array([1, 0, 0])  # 目标向量

    # print('pt_n_1:', pt_n_1)
    ############## start 模型点 根据法向量变换
    theta_mg, rot_axis = get_ang_3d(pt_n_1, x_axis)  # 旋转角 旋转轴 1.起始 2.终止  不可反
    # print('theta_mg:', theta_mg)  # 这个角度是从目标到起始!
    # print('rot_axis:', rot_axis)

    # Rodrigues  对平移后的mi进行变换  OK
    # print('pt_m_2 before:', pt_m_2)
    mi_rot = get_rodrigues(pt_m_2, theta_mg, rot_axis)  # 问题:无法根据轴向确定旋转方向
    print('mi_rot after ok：', mi_rot)  #

    # ****************************#  ！！！！！！
    res = from_rodri_to_mat(rot_axis, theta_mg)  # 轴角->mat  有问题
    # print('res:\n', res)  # 这里显然不对了 不是正交
    # 检验是否正确：使用res dot v，看是否等于v_rot  结果不一样！
    mi_rot = dot(res, pt_m_2)  # 不对
    print('mi_rot que:', mi_rot)
    # ****************************# ！！！！！！

    ############# 场景点 根据法向量变换
    theta_mg, rot_axis = get_ang_3d(pt_s_n_1, x_axis)  # 旋转角  根据法向量
    # print('theta_mg_2:', theta_mg)
    # print('rot_axis_2:', rot_axis)

    # Rodrigues  对si进行变换
    # print('pt_s_t_2:', pt_s_t_2)
    si_rot = get_rodrigues(pt_s_t_2, theta_mg, rot_axis)
    # print('si_rot after：', si_rot)

    # 将两个变换后的i点 投影到yoz平面
    mi_rot = mi_rot[1:]
    si_rot = si_rot[1:]

    alpha = get_ang_2d(mi_rot, si_rot)
    print('alpha:', alpha)

    r_mat_alpha = rot_x(alpha)  # OK
    print('r_mat_alpha:\n', r_mat_alpha)  #

    # 由局部坐标恢复全局姿态： Rs = R1s_inv * R2s_inv * R2m * R1m   其中1、2代表第几次旋转
    # 首先把轴角表示转换成旋转矩阵   Q：是不是可以直接用轴角表示？

    # R1s_inv = inv()
    # R_s =

    #############
    # 得到旋转轴  方向判断 右手定则

    # 罗格里德:把轴角旋转 -> 旋转矩阵  然后对mi和si进行变换
    # v_p = dot(dot(pt_n_1, rot_axis), rot_axis)  # x||  水平分量 带方向
    # v_p = dot(pt_n_1, rot_axis)  # x||  水平分量 标量
    # print('v||:', v_p)

    # v_t = pt_n_1 - v_p
    # print('v_t:', v_t)  # 垂直分量
    # theta = 45  # 问题:事先不知道这个旋转角度
    # v_t_rot = v_t * cos(theta) + cross(rot_axis, pt_n_1) * sin(theta)
    # # print('v_t_rot:', v_t_rot)
    #
    # v_rot = v_p + v_t_rot
    # # print('v_rot:', v_rot)
