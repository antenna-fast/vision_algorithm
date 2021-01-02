from numpy import *
from scipy.spatial.transform import Rotation as R


def get_pose(m_r, m_r_n, s_r, s_r_n):
    # 变换到统一坐标系

    # 对齐m_r与坐标原点
    # 对齐m_r_n与

    return 0


if __name__ == '__main__':
    print()

    # 假设现在已经找到了匹配点对，现在要将他们对齐  找到其中的变换

    # 每个对齐的点对 包含：
    # pt_m_r, pt_m_r_n, pt_m_i, pt_m_i_n
    # pt_s_r, pt_s_r_n, pt_s_i, pt_s_i_n

    m_r = array([0, 0, 0])
    m_r_n = array([0, 1, 0])

    m_i = array([1, 0, 0])
    m_i_n = array([0, 1, 0])

    # 定义变换
    t = array([1, 1, 1])  # 只有平移变换时 法向量不变
    r_vec = R.from_rotvec(array([0, 0, 0]))
    r_mat = r_vec.as_matrix()

    # print(r_mat)

    s_r = m_r + t
    s_r_n = m_r_n

    s_i = m_i + t
    s_i_n = m_i_n

    # 对齐mr sr
    t_mg = m_r
    t_sg = s_r

    # 对齐法向量 者需要旋转矩阵
    

    # 将m_r_n对齐到x轴
    x_axis = array([1, 0, 0])
    ang = arctan(cross(x_axis, m_r_n) / dot(x_axis, m_r_n))
    print(ang)