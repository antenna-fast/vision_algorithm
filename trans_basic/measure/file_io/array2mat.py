import scipy.io as io
from numpy import *


f_path = '../save_file_kpt_idx/key_pts_buff_idx_0.07.txt'
key_pts_buff_1 = loadtxt(f_path)

# a = array([1, 2, 3, 4])
s = {'IP_vertex_indices': key_pts_buff_1}
io.savemat('D:/SIA/科研/Benchmark/3DInterestPoint/3DInterestPoint/IP_BENCHMARK/ALGORITHMs_INTEREST_POINTS/Ours/bird_3.mat', s)
