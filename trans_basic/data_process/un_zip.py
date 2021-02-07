import tarfile
import gzip
import os

# SHREC模型

# data_root = 'C:/Users/yaohua-win/Desktop/data/'
data_root = 'D:/SIA/科研/data/'

d = os.listdir(data_root)
print(d)

# 解压所有
# for di in d:
#     d1 = data_root + di
#     print(d1)
#
#     f = tarfile.open(d1)
#     f.extractall(data_root + 'unzip_1')

f2_list = os.listdir(data_root + 'unzip_1')
# f2_path = 'airplanesPly/b1.ply.gz'

for f2_path in f2_list:
    f2_path = data_root + 'unzip_1/' + f2_path

    f3_list = os.listdir(f2_path)
    for f3 in f3_list:
        f3 = f2_path + '/' + f3
        f_name = f3.replace('.gz', '')

        g_file = gzip.GzipFile(f3)
        open(f_name, 'wb+').write(g_file.read())
