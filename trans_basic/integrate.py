import open3d as o3d

from point_to_plan import pt_to_plan  # px, p, p_n  返回投影后的三维点


# 加载
pcd = o3d.io.read_point_cloud('../data_ply/Armadillo.ply')
pcd = pcd.dowmsample()

# 遍历

# 找到邻域

# 对每个邻域:
# 邻域局部坐标系

# 将周围的点投影到平面

# 将投影后的点旋转至z轴,得到投影后的二维点

# Delauney三角化

# 创建mesh

# 求mesh normal

# other
