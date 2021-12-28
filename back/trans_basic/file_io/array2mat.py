import scipy.io as io
from numpy import *


# 读取检测的关键点,保存为mat  注意mat从1开始

f_path = '../measure/save_file_kpt_idx/key_pts_buff_idx_0.07.txt'
key_pts_buff_1 = loadtxt(f_path)


s = {'IP_vertex_indices': key_pts_buff_1}
io.savemat('D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/ALGORITHMs_INTEREST_POINTS/Ours/bird_3.mat', s)
