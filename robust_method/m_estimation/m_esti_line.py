from numpy import *

# 使用M-Estimator拟合一条直线

# 基本思想:
# 对残差进一步做处理,如果残差太大,则认为是离群点,要减弱离群点的影响


# 不同形式的函数:

# 做成分段函数
# 残差太大的变成常数,残差小的变成残差的平方
def e_huber(e, threshold):
    if abs(e) < threshold:
        return e**2

    return threshold


# 较为普遍的?
def e_normal(t, sigma):
    t2 = t**2
    p = t2 / (t2 + sigma**2)
    return p
