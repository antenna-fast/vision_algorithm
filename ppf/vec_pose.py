from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R

import scipy


# 注意 投影后直接旋转?
def get_rota(vec):
    r_mat = 0
    return r_mat


# 反对称阵
def get_anti_sym(vec):
    mat = array([[0, -1 * vec[2], vec[1]],
                 [vec[2], 0, -1 * vec[0]],
                 [-1 * vec[1], vec[0], 0]])
    return mat


if __name__ == '__main__':
    # 模型点对
    pt_1 = array([0, 0, 0])
    pt_2 = array([1, 0, 0])
    pt_n_1 = array([0, 1, 1])
    pt_n_1 = pt_n_1 / norm(pt_n_1)  # 法向量都是单位的
    pt_n_2 = array([0, 1, 0])
    # vec_1 = pt_2 - pt_1

    # 定义变换  只加入一些平移 法向量不变
    # r = R.from_rotvec(pi / 180 * array([30, 60, 30]))  # 角度->弧度
    # r_mat = r.as_matrix()
    t_vect = array([150, -2, -8], dtype='float')
    # print('r_mat:\n', r_mat)

    # 场景点对
    # pcd_trans = dot(r_mat, pcd_trans.T).T
    pt_s_1 = pt_1 + t_vect
    pt_s_2 = pt_2 + t_vect
    pt_s_n_1 = pt_n_1
    pt_s_n_s = pt_n_2

    # Tmg:
    Tmg = pt_1
    Tsg = pt_s_1

    # print(Tsg - Tmg)
    # 将模型点和场景参考点变换到统一坐标系
    pt_m_1 = pt_1 - Tmg
    pt_m_2 = pt_2 - Tmg
    # print(pt_m_1, pt_m_2)

    # 场景
    pt_s_t_1 = pt_s_1 - Tsg
    pt_s_t_2 = pt_s_2 - Tsg
    # print(pt_s_t_1, pt_s_t_2)

    # 平移算完了 只需要得到旋转
    # 基于alpha进行投票?

    x_axis = array([1, 0, 0])  # 目标向量
    pt_n_1 = array([0, 0, 1])  # 待
    # res = scipy.spatial.transform.Rotation.align_vectors(x_axis, pt_n_1)
    # print(res)

    # 得到旋转轴  注意 np里坐标系的z轴反了
    rot_axis = cross(x_axis, pt_n_1)  # np
    # rot_axis = cross(x_axis, pt_n_1)
    print('rot_axis_1:', rot_axis)
    # 为了方便计算 写成矩阵乘法

    a_mat = get_anti_sym(x_axis)  # 计算叉积时
    rot_axis_2 = dot(a_mat, pt_n_1)
    print('rot_axis_2:', rot_axis_2)

    rot_axis = array([1, 0, 0])  # 手动设定了一个旋转轴
    print('rot_axis:', rot_axis)

    print('pt_n_1:', pt_n_1)

    # 罗格里德:把轴角旋转 -> 旋转矩阵  然后对mi和si进行变换
    v_p = dot(dot(pt_n_1, rot_axis), rot_axis)  # x||  水平分量 带方向
    # v_p = dot(pt_n_1, rot_axis)  # x||  水平分量 标量
    print('v||:', v_p)

    v_t = pt_n_1 - v_p
    print('v_t:', v_t)  # 垂直分量
    theta = 45  # 问题:事先不知道这个旋转角度
    v_t_rot = v_t * cos(theta) + cross(rot_axis, pt_n_1) * sin(theta)
    print('v_t_rot:', v_t_rot)

    v_rot = v_p + v_t_rot
    print('v_rot:', v_rot)

    r_mat_x = get_rota(array([1, 0, 0]))  # x轴在坐标系中的绝对位姿  只能是作为参考姿态 而不能作为转换
    r_mat_n1 = get_rota(pt_n_1)  # 向量的绝对位姿

    # 相对于x轴的变换

    # print('r_mat_x:\n', r_mat_x)
    # 用这个的逆 乘以法向量，即可对齐
    r_mat_x_inv = inv(r_mat_x)
    mat = dot(r_mat_n1, r_mat_x_inv)  # 相对姿态
    n_2 = dot(mat, pt_n_1)
    # print(pt_n_1)
    # print('mat:\n', mat)
    # print('n_2:', n_2)
