from numpy import *
import numpy as np
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R
import scipy

from transform import *


# # 定义一些常数
rad2deg = 180 / pi


# 找到二维argmax索引
def get_max_idx_2d(a):
    max_ele = np.max(a)
    idx = np.where(a == max_ele)
    return idx[0][0], idx[1][0]  # 如果有多个相同的最大值，只返回其中的一个


# 反对称阵
def get_anti_sym(vec):
    mat = array([[0, -1 * vec[2], vec[1]],
                 [vec[2], 0, -1 * vec[0]],
                 [-1 * vec[1], vec[0], 0]])
    return mat


# 2d里面 叉乘之后 逆时针是正的
def get_ang_2d(vec2, vec1):  # vec1 起始向量，vec2 终止向量
    dist = dot(vec1, vec2) / (norm(vec1) * norm(vec2))  # 小夹角 接近于1

    if dist > 1:  # 由于精度损失 可能会略微大于1 此时无解
        dist = 1

    # print('dist:', dist)
    ang = arccos(dist) * rad2deg  # 转为角度
    if ang != 180:  # 共线的时候 叉积等于0 无法使用
        ang = ang * sign(cross(vec1, vec2))

    return ang


def get_ang_2d_tan(a, b):  #
    ang = arctan2(a, b)
    ang = ang * rad2deg
    return ang


# test
a = [1, 1]
b = [1, 0]
a = get_ang_2d(a, b)  # 1目标向量  2参考向量  a相对于b的
print('ang:', a)


# 轴角
# 根据叉乘 得到旋转方向  对于轴角，旋转方向与轴一致
def get_ang_3d(vec1, vec2):
    dist = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    ang = arccos(dist) * 180 / pi

    rot_axis = cross(vec1, vec2)  # 旋转轴  决定旋转方向

    return ang, rot_axis


# 轴角旋转 方向由转轴方向定义  转角0-pi  转轴和叉积前后对应有关，统一即可
def get_rodrigues(v, ang, k):  # 角度制输入  OK
    """
    推导过程：
    v_p = v*k*k
    v_t = v - v_p
    v_t_rot = v_t*cos(theta) + kxv*sin(theta)
    v_rot = v_p + v_t_rot
    """
    # 单位化k
    k = k / norm(k)
    ang = ang * pi / 180
    cos_theta = cos(ang)
    cross_kv = cross(k, v)
    v_rot = (1 - cos_theta) * dot(dot(k, v), k) + v * cos_theta + sin(ang) * cross_kv
    return v_rot


# 从轴角转换成旋转矩阵  旋转轴是单位向量   1x3
def from_rodri_to_mat(k, theta):
    theta = theta * pi / 180
    K = get_anti_sym(k)
    # u 代表k
    # R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K ** 2)
    # 参考：CHAPTER 9. ROTATION ABOUT AN ARBITRARY AXIS
    k_T = k[:, newaxis]
    k_vec = k.reshape(1, 3)
    R = eye(3) * cos(theta) + (1 - cos(theta)) * dot(k_T, k_vec) + sin(theta) * K

    return R


def get_alpha(mr, mi, mr_n, sr, si, sr_n):
    # Tmg:
    Tmg = -mr  # 参考点  mr
    Tsg = -sr  # sr
    # print(Tsg - Tmg)

    # 将模型点和场景参考点变换到统一坐标系
    # pt_m_1 = mr + Tmg  # 变换后的参考点  好像没用到  总是0
    pt_m_2 = mi + Tmg
    # print('pt_m_1, pt_m_2:', pt_m_1, pt_m_2)

    # 平移变换 场景
    # pt_s_t_1 = sr + Tsg
    pt_s_t_2 = si + Tsg
    # print('pt_s_t_1, pt_s_t_2:', pt_s_t_1, pt_s_t_2)

    # 平移算完了 接下来需要得到旋转
    # 基于alpha进行投票

    x_axis = array([1, 0, 0])  # 目标向量

    # print('pt_n_1:', pt_n_1)
    ############## start 模型点 根据法向量变换
    theta_mr_1, rot_axis_mr_1 = get_ang_3d(mr_n, x_axis)  # 旋转角 旋转轴 1.起始 2.终止  不可反
    # print('theta_mg:', theta_mg)  # 这个角度是从目标到起始!
    # print('rot_axis_mr_1:', rot_axis_mr_1)

    # Rodrigues  对平移后的mi进行变换  OK
    # print('pt_m_2 before:', pt_m_2)
    mi_rot = get_rodrigues(pt_m_2, theta_mr_1, rot_axis_mr_1)
    # print('mi_rot after ok：', mi_rot)  #

    ############# 场景点 根据法向量变换
    theta_sr_1, rot_axis_sr_1 = get_ang_3d(sr_n, x_axis)  # 旋转角  根据法向量
    # print('theta_sr_1:', theta_sr_1)
    # print('rot_axis_sr_1:', rot_axis_sr_1)

    # Rodrigues  对si进行变换
    # print('pt_s_t_2:', pt_s_t_2)
    si_rot = get_rodrigues(pt_s_t_2, theta_sr_1, rot_axis_sr_1)
    # print('si_rot after：', si_rot)

    # 将两个变换后的i点 投影到yoz平面
    mi_rot = mi_rot[1:] / norm(mi_rot[1:])
    si_rot = si_rot[1:] / norm(si_rot[1:])
    # print('mi_rot:', mi_rot)
    # print('si_rot:', si_rot)

    # 求错了 既然是和x轴的夹角，怎么能再用x轴呢！  此处用y轴
    alpha_ref_axis = array([0, 1])  # alpha的参考轴
    alpha_mi = get_ang_2d(mi_rot, alpha_ref_axis)  # 逆时针为正
    alpha_si = get_ang_2d(si_rot, alpha_ref_axis)  #
    # print('alpha:', alpha)

    alpha = alpha_mi - alpha_si

    return alpha


def get_pose(alpha_mi, rot_axis_mr_1, theta_mr_1, alpha_si, rot_axis_sr_1, theta_sr_1):
    r_mat_alpha_mi = rot_x(alpha_mi)  # OK
    # print('r_mat_alpha:\n', r_mat_alpha_mi)  #

    r_mat_alpha_si = rot_x(alpha_si)  # OK
    # print('r_mat_alpha:\n', r_mat_alpha_si)  #

    # 由局部坐标恢复全局姿态： Rs = R1s_inv * R2s_inv * R2m * R1m   其中1、2代表第几次旋转
    # 首先把轴角表示转换成旋转矩阵   Q：是不是可以直接用轴角表示？
    R_m1 = from_rodri_to_mat(rot_axis_mr_1, theta_mr_1)  # mr_n对齐x轴   # k, theta
    R_m2 = r_mat_alpha_mi[:3, :3]  # Alpha mi
    R_s1 = from_rodri_to_mat(rot_axis_sr_1, theta_sr_1)  # sr_n对齐x轴
    R_s2 = r_mat_alpha_si[:3, :3]  # Alpha si

    R_inv_temp = dot(inv(R_s1), inv(R_s2))
    R_m_temp = dot(R_m2, R_m1)
    pose_s = dot(R_inv_temp, R_m_temp)
    print('pose_s:\n', pose_s)  # 场景相对于模型的位姿
    #############

    return pose_s


if __name__ == '__main__':
    # 模型点对
    pt_1 = array([0, 0, 0])  # mr
    pt_2 = array([1, 0, 0])  # mi
    pt_n_1 = array([0, 1, 1])  # mr_n
    pt_n_1 = pt_n_1 / norm(pt_n_1)  # 法向量都是单位的
    pt_n_2 = array([0, 1, 0])  # mi_n  没用到
    # vec_1 = pt_2 - pt_1

    # 定义变换  只加入一些平移 法向量不变
    t_vect = array([150, -2, -8], dtype='float')
    # print('r_mat:\n', r_mat)

    # 场景点对
    # pcd_trans = dot(r_mat, pcd_trans.T).T
    pt_s_1 = pt_1 + t_vect  # sr
    pt_s_2 = pt_2 + t_vect  # si
    pt_s_n_1 = pt_n_1  # sr_n
    pt_s_n_2 = pt_n_2  # si_n

    a = get_alpha(pt_1, pt_2, pt_n_2, pt_s_1, pt_s_2, pt_s_n_2)
    print('a:', a)

    # Tmg:
    Tmg = -pt_1  # 参考点
    Tsg = -pt_s_1
    # print(Tsg - Tmg)

    # 将模型点和场景参考点变换到统一坐标系
    # pt_m_1 = pt_1 + Tmg  # 变换后的模型点  0 0
    pt_m_2 = pt_2 + Tmg
    # print('pt_m_1, pt_m_2:', pt_m_1, pt_m_2)

    # 平移变换 场景
    # pt_s_t_1 = pt_s_1 + Tsg
    pt_s_t_2 = pt_s_2 + Tsg
    # print('pt_s_t_1, pt_s_t_2:', pt_s_t_1, pt_s_t_2)

    # 平移算完了 接下来需要得到旋转
    # 基于alpha进行投票

    x_axis = array([1, 0, 0])  # 目标向量

    # print('pt_n_1:', pt_n_1)
    ############## start 模型点 根据法向量变换
    theta_mr_1, rot_axis_mr_1 = get_ang_3d(pt_n_1, x_axis)  # 旋转角 旋转轴 1.起始 2.终止  不可反
    # print('theta_mg:', theta_mg)  # 这个角度是从目标到起始!
    # print('rot_axis:', rot_axis)

    # Rodrigues  对平移后的mi进行变换  OK
    # print('pt_m_2 before:', pt_m_2)
    mi_rot = get_rodrigues(pt_m_2, theta_mr_1, rot_axis_mr_1)
    # print('mi_rot after ok：', mi_rot)  #

    # ****************************#  OKOKOKOKOK  轴角转旋转矩阵
    # res = from_rodri_to_mat(rot_axis, theta_mg)  # 轴角->mat  有问题
    # print('res:\n', res)
    # # # 检验是否正确：使用res dot v，看是否等于v_rot 结果一样！  1.20 发现是向量点乘要确定维度！ 从公式出发解决的
    # mi_rot = dot(res, pt_m_2)  # 不对
    # print('mi_rot que:', mi_rot)
    # ****************************#  OKOKOKOKOK

    ############# 场景点 根据法向量变换
    theta_sr_1, rot_axis_sr_1 = get_ang_3d(pt_s_n_1, x_axis)  # 旋转角  根据法向量
    # print('theta_sr_1:', theta_sr_1)
    # print('rot_axis_sr_1:', rot_axis_sr_1)

    # Rodrigues  对si进行变换
    # print('pt_s_t_2:', pt_s_t_2)
    si_rot = get_rodrigues(pt_s_t_2, theta_sr_1, rot_axis_sr_1)
    # print('si_rot after：', si_rot)

    # 将两个变换后的i点 投影到yoz平面
    mi_rot = mi_rot[1:]
    si_rot = si_rot[1:]
    # print('mi_rot:', mi_rot)
    # print('si_rot:', si_rot)

    alpha_ref_axis = array([1, 0])  # alpha的参考轴
    alpha_mi = get_ang_2d(mi_rot, alpha_ref_axis)  # 这里不知道正负？？这里已经知道正负
    alpha_si = get_ang_2d(si_rot, alpha_ref_axis)  # 这里不知道正负？？这里已经知道正负
    # print('alpha:', alpha)
    #
    # r_mat_alpha_mi = rot_x(alpha_mi)  # OK
    # # print('r_mat_alpha:\n', r_mat_alpha_mi)  #
    #
    # r_mat_alpha_si = rot_x(alpha_si)  # OK
    # # print('r_mat_alpha:\n', r_mat_alpha_si)  #
    #
    # # 由局部坐标恢复全局姿态： Rs = R1s_inv * R2s_inv * R2m * R1m   其中1、2代表第几次旋转
    # # 首先把轴角表示转换成旋转矩阵   Q：是不是可以直接用轴角表示？
    # R_m1 = from_rodri_to_mat(rot_axis_mr_1, theta_mr_1)  # mr_n对齐x轴   # k, theta
    # R_m2 = r_mat_alpha_mi[:3, :3]  # Alpha mi
    # R_s1 = from_rodri_to_mat(rot_axis_sr_1, theta_sr_1)  # sr_n对齐x轴
    # R_s2 = r_mat_alpha_si[:3, :3]  # Alpha si
    #
    # R_inv_temp = dot(inv(R_s1), inv(R_s2))
    # R_m_temp = dot(R_m2, R_m1)
    # pose_s = dot(R_inv_temp, R_m_temp)
    # print('pose_s:\n', pose_s)  # 场景相对于模型的位姿
    # #############
