# 使用ransac进行6D位姿估计   测试

from numpy import *
import matplotlib.pyplot as plt

import time

# 框架：
# 加载数据  or 生成
#   采样，生成模型  用对应点  需要*个点确定一个位姿矩阵  *要不要考虑尺度?
#   计算模型在数据上的置信度
# 迭代完成之后，选择置信度最大的
