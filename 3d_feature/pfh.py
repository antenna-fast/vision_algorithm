from numpy import *
from numpy.linalg import *


# 得到一个邻域的PFH
# 输入 1.中心点 2.邻域点 3.单位化的法向量
# 返回125维的特征直方图

def get_pfh(now_pt, vicinity, vicinity_n, vic_num):
    # 采样任意两点
    for i in range(vic_num):
        pt_i = vicinity[i]
        pt_i_n = vicinity_n[i]

        for j in range(vic_num):
            pt_j = vicinity[j]
            pt_j_n = vicinity_n[j]

            # 两点的连线  pt_j对应第二个点 p2
            pj_pi = pt_j - pt_i
            pj_pi_unit = pj_pi / norm(pj_pi)

            # 三个单位向量的局部坐标系
            u = pt_i_n
            v = cross(u, pj_pi_unit)
            w = cross(u, v)

            # 三个角度 表达两个法向量之间的差异
            a = dot(v, pt_j_n)
            b = dot(u, pj_pi_unit)
            c = arctan2(dot(w, pt_j_n), dot(u, pt_j_n))

            # 量化 将其放到直方图中
            # 这里得到了Cn2 + 1 组数据
            # 直方图的构建：先将三个角度离散化，然后放到对应的区间里面

    return 0


if __name__ == '__main__':
    print()
    # 读取

    #
