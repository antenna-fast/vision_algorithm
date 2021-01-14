from numpy import *


# 投影点[XYZ] 投影线[ABC]
def pt_to_line(p, l):
    l_n = array([l[0], l[1]])

    # 求参数t
    t = -(l[2] + l[0] * p[0] + l[1] * p[1]) / (l[0] * l_n[0] + l[1] * l_n[1])
    print('t:', t)

    x = p + t * l_n
    print(x)

    return x


# 输入:点 平面 平面法向量
# 输出:投影点
def pt_to_plan(a, p, p_n):
    # 参数t
    t = -1 * (p[0] * a[0] + p[1] * a[1] + p[2] * a[2] + p[3]) / (p[0] * p_n[0] + p[1] * p_n[1] + p[2] * p_n[2])
    # print(t)

    x = a + t * p_n

    return x
