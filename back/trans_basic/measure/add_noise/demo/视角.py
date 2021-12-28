from o3d_impl import *

import matplotlib.pyplot as plt


# 可视化
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([
    # mesh_gt,
    # mesh_noise,
    # kpts_gt,
    # kpts_noise,
    axis_pcd,
    ],
    window_name='ANTenna3D',
    zoom=1,
    front=[0, 10, 0.01],  # 相机位置
    lookat=[0, 0, 0],  # 对准的点
    up=[0, 1, 1],  # ?  用于确定相机右x轴
    point_show_normal=True
)


# 保存图片，而不是维截图了
# vis = o3d.visualization.Visualizer()
# # image = vis.capture_screen_float_buffer(False)
#
# image = vis.capture_screen_float_buffer(False)
# plt.imshow(np.asarray(image))
# plt.show()
