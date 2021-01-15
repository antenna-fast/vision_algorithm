from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R


def get_rota(vec):
    # 得到一个向量在global坐标系中的姿态  角度格式
    # 将参考点法向量对齐到x轴  这是一个纯旋转变换
    rad2ang = 180 / pi
    ang_x = arccos(vec[0] / 1) * rad2ang  # 这里其实求得在原坐标系中的姿态
    ang_y = arccos(vec[1] / 1) * rad2ang
    ang_z = arccos(vec[2] / 1) * rad2ang

    print(ang_x, ang_y, ang_z)
    # 用角度构造旋转矩阵
    #  问题：不是直接使用这个角度！
    r = R.from_rotvec(pi / 180 * array([ang_x, ang_y, ang_z]))  # 角度->弧度
    r_mat = r.as_matrix()
    print('r_mat:\n', r_mat)

    return r_mat


if __name__ == '__main__':

    # 模型点对
    pt_1 = array([0, 0, 0])
    pt_2 = array([1, 0, 0])
    pt_n_1 = array([1, 0, 0])
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

    r_mat_x = get_rota(array([1, 0, 0]))  # x轴在坐标系中的绝对位姿  只能是作为参考姿态 而不能作为转换
    r_mat_n1 = get_rota(pt_n_1)  # 向量的绝对位姿

    # 相对于x轴的变换

    # print('r_mat_x:\n', r_mat_x)
    # 用这个的逆 乘以法向量，即可对齐
    r_mat_x_inv = inv(r_mat_x)
    mat = dot(r_mat_n1, r_mat_x_inv)  # 相对姿态
    n_2 = dot(mat, pt_n_1)
    # print(pt_n_1)
    print('mat:\n', mat)
    print('n_2:', n_2)
