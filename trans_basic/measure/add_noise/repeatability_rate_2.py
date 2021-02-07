from numpy import *
from numpy.linalg import *
import open3d as o3d

from base_trans import *
from dist import *  # 距离计算


# 关键点用来评估检测算法对变换的鲁棒性

# 给定两个检测出来的关键点，在经过变换之后，看第二个检测出来的是不是在邻域范围之内
# 如果是，就认为是重合的


# rep(X,Y)  如果Y中检测到X中也有的关键点
# (rep(X,Y) + ep(Y,X)) / 2


# 索引一致的情况 (可以通过索引找到关键点)
# 给定两组点 以及变换
def get_repeate_rate(pcd_np_1, pcd_np_2, r_mat, t_vect):
    # 首先将模型变换到场景
    pcd_np_trans = dot(r_mat, pcd_np_1.T).T + t_vect

    repeat_num = 0
    all_repeat = min(len(pcd_np_1), len(pcd_np_2))  # 取出较小的一组

    p_num = len(pcd_np_1)
    for pt_idx in range(p_num):  # 变换后的 参照点
        pt_1 = pcd_np_trans[pt_idx]
        pt_2 = pcd_np_2[pt_idx]

        dist = sqrt(sum((pt_1 - pt_2) ** 2))
        # print(dist)

        if dist < 1:
            repeat_num += 1

    repeat_rate = repeat_num / all_repeat
    # print(repeat_rate)

    return repeat_rate


# 索引不一致
# 搜索得到最近邻
# 输入是关键点
def get_repeate_rate_2(pcd_np_1, pcd_np_2, r_mat, t_vect):
    repeat_num = 0

    vici_num = 2  # 2近邻 包含他自己

    # 首先将模型变换到场景
    pcd_np_trans = dot(r_mat, pcd_np_1.T).T + t_vect

    pcd1_num = len(pcd_np_1)
    pcd2_num = len(pcd_np_2)

    all_repeat = min(pcd1_num, pcd2_num)  # 取出较小的一组

    for pt_idx in range(pcd1_num):  # 根据模型中的关键点进行比较
        pt_1 = pcd_np_trans[pt_idx]

        # print('len(pcd2_np):', len(pcd2_np))

        # 将变换后的点添加到变换后的场景中
        pcd2_np_temp = pcd_np_2  # 每次都更新
        pcd2_np_temp = vstack((pcd2_np_temp, pt_1))  #
        # print('len(pcd2_np_temp):', len(pcd2_np_temp))

        # 找到最近邻点 看距离是否在范围内  其实这种方法也不需要提前知道变换
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd2_np_temp)

        # 构建搜索树
        pcd_tree_2 = o3d.geometry.KDTreeFlann(pcd2)

        [k, idx, _] = pcd_tree_2.search_knn_vector_3d(pcd2.points[pcd2_num], vici_num)  # 最后一个点的近邻
        vici_idx = idx[1:]
        # asarray(pcd.colors)[vici_idx_1, :] = [0, 0, 1]

        vici_pts = array(pcd2.points)[vici_idx]
        # all_pts = array(pcd.points)[idx]

        dist = sqrt(sum((pt_1 - vici_pts)**2))
        print(dist)

        if dist < 1:  # 距离阈值
            repeat_num += 1

        # print(pt_idx / all_repeat)

    repeat_rate = repeat_num / all_repeat
    print(repeat_rate)

    return repeat_rate


# 准确率
# 预测正确的结果所占的比例  TP+TN/TP+TN+FP+FN
def get_accuracy(TP, FP, TN, FN):
    return (TP+TN)/(TP+TN+FP+FN)


# 精确率
# 所有被识别为正类别的样本中，真正为正样本的比例  TP/TP+FP
def get_precision(TP, FP):
    return TP / (TP + FP)


# 召回率
# 所有正样本中，被正确识别为正样本的比例  TP/TP+FN
def get_recall(TP, FN):
    return TP / (TP + FN)

# 对于点云的关键点来说
# 令：
# TP+FN是模型上的关键点
# TP是变换后 实际分类为关键点里面真正关键点的数量
# FN是变换后 真正的关键点，但被分类为N
# FP是变换后 实际是N 被分成了P


if __name__ == '__main__':

    # 先直接使用原始数据
    # 基于索引,然后查找最近点,如果距离小于某个数值就算重合

    # 加载保存的关键点
    key_pts_buff_1 = loadtxt('save_file/key_pts_buff_1.txt')
    key_pts_buff_2 = loadtxt('save_file/key_pts_buff_2.txt')

    # 对第一个变换

    # 定义变换
    r = R.from_rotvec(pi / 180 * array([30, 60, 30]))  # 角度->弧度
    # r = R.from_rotvec(pi / 180 * array([0, 0, 10]))  # 角度->弧度
    r_mat = r.as_matrix()
    t_vect = array([150, -2, -8], dtype='float')
    # print('r_mat:\n', r_mat)

    # 比较
    ra = get_repeate_rate_2(key_pts_buff_1, key_pts_buff_2, r_mat, t_vect)  # pcd_np_1, pcd_np_2, r_mat, t_vect

    print('重复率', ra)
