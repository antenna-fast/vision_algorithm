# 加载GT，用作检测，看IOU几何

from o3d_impl import *
from dist import *  # 距离计算
import os

import scipy.io as scio


# 加载GT

# data_path = out_path + 'bird_3.ply'
# model_name = 'ant'  # 手动实验：为了得到测试结果，不太好批量化
# model_name = 'bird_3'
# model_name = 'armadillo'
# model_name = 'bust'
# model_name = 'girl'
# model_name = 'hand_3'
model_name = 'camel'

gt_root = 'D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/OUTPUT_DATA/GROUND_TRUTH_A/'
gt_path = gt_root + model_name + '.mat'
gt_idx_dic = scio.loadmat(gt_path)
print(gt_idx_dic.keys())
print(gt_idx_dic['GT_MODEL'].shape)

# T x N-1 x 2
sigma = 0.03
i = 3  # T sigma的个数?
n = 11
gt_idx = gt_idx_dic['GT_MODEL'][i, n-1, 0]  # 索引小1
print(gt_idx.shape)
gt_idx = squeeze(gt_idx) - 1  # 注意 要-1  matlab从1开始
print(gt_idx)
