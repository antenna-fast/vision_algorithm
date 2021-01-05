from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R


if __name__ == '__main__':

    # 模型点
    pt_1 = array([0, 0, 0])
    pt_2 = array([1, 0, 0])
    vec_1 = pt_2 - pt_1

    # 场景点
    pt_3 = array([2, 1, 0])
    pt_4 = array([3, 2, 0])
    vec_2 = pt_4 - pt_3

    # Tmg:
    # 平移mr 到坐标原点
    t_mg = pt_1
