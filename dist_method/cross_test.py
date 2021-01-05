from numpy import *

# 目的：解决法向量的旋转不变性问题
# 要构建局部坐标系，如何才能保证z轴 也就是法向量方向一致

a = array([1, 0, 0])
b = array([0, 1, 0])

ab_norm = cross(a, b)
# ab_norm = cross(b, a)  # 反过来就是-
print(ab_norm)
