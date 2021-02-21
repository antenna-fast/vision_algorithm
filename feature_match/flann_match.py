import cv2
from numpy import *


# While using ORB, you can pass the following.
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)  # or pass empty dictionary  这是第二个字典，指定了索引里的树应该被递归遍历的次数

flann = cv2.FlannBasedMatcher(index_params, search_params)


# 制作描述子
des_1 = array([[0, 0], [1, 1], [3, 3], [9, 9], [10, 10]]).astype(uint8)  # 训练 场景的 只能多不能少  模板  只有这个类型怎么可以，浮点数的特征呢！
des_2 = array([[3, 3], [0, 0], [1, 1], [9, 9]]).astype(uint8)  # 查询   应当是模型的

# 输出的形式：遍历des_2，在des_1中找到最近的点
# queryDescriptors, trainDescriptors, k, mask, compactResult
matches = flann.knnMatch(des_2, des_1, k=2)

# print(matches)
for (m, n) in matches:  # m是最小距离  n是次小距离（或者一会加上过滤）
    print(m.distance)
    if m.distance < 0.7 * n.distance:

        print('queryIdx:', m.queryIdx)  # 查询  des2  模型上的
        print('trainIdx:', m.trainIdx)  # 训练集中的索引  场景上的索引
