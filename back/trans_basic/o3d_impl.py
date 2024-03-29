import open3d as o3d

from point_project import *  # 点投影到面
from n_pt_plane import *  #
from base_trans import *


# open3d function lib

# 从点和索引构建mesh
def get_non_manifold_vertex_mesh(verts, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    return mesh


# 自定义数据处理
# 从顶点创建mesh
# 要改名成create_mesh
def get_mesh(now_pt, vici_pts):
    # 得到邻域构成的面片 得到面片法向量  顶点法向量
    coord = get_coord(now_pt, vici_pts)  # 列向量表示三个轴
    normal = coord[:, 2]  # 第三列
    # print('coord:\n', coord)

    # 还有一步 利用法向量和中心点,得到平面方程[ABCD]
    p = get_plan(normal, now_pt)

    # * 找到拓扑结构 START
    all_pts = vstack((now_pt, vici_pts))
    # 将周围的点投影到平面
    # plan_pts = []
    # for pt in all_pts:  # 投影函数可以升级  向量化
    plan_pts = pt_to_plane(all_pts, p, normal)  # px p pn
    # plan_pts = array(plan_pts)  # 投影到平面的点

    # 将投影后的点旋转至z轴,得到投影后的二维点
    coord_inv = inv(coord)  # 反变换
    # rota_pts = dot(coord_inv, all_pts.T).T  # 将平面旋转到与z平行
    # 首先要将平面上的点平移到原点 然后再旋转  其实不平移也是可以的，只要xy平面上的结构不变

    rota_pts = dot(coord_inv, plan_pts.T).T  # 将平面旋转到与z平行

    # rota_pts[:, 2] = 0  # 已经投影到xoy(最大平面),在此消除z向轻微抖动
    pts_2d = rota_pts[:, 0:2]

    # Delauney三角化
    tri = Delaunay(pts_2d)
    tri_idx = tri.simplices  # 三角形索引
    # print(tri_idx)

    # 统计三角形的数量
    # tri_num = len(tri_idx)
    # print(tri_num)

    # 可视化二维的投影
    # plt.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices.copy())
    # plt.plot(pts_2d[:, 0], pts_2d[:, 1], 'o')
    # plt.show()
    # * 找到拓扑结构 END

    # 根据顶点和三角形索引创建mesh
    mesh = get_non_manifold_vertex_mesh(all_pts, tri_idx)

    # 求mesh normal
    mesh.compute_triangle_normals()
    mesh_normals = array(mesh.triangle_normals)
    # print(mesh_normals)

    # 法向量同相化
    for i in range(len(mesh_normals)):
        if dot(mesh_normals[i], normal) < 0:
            mesh_normals[i] = -mesh_normals[i]
    mesh.triangle_normals = o3d.utility.Vector3dVector(mesh_normals)

    return mesh, mesh_normals, normal


# 数据变换
# 将pcd格式点转换成球
# This function is only used to make the keypoints look better on the rendering

def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres


# np格式的
def keypoints_np_to_spheres(keypoints, size=0.1, color=[0, 1, 0]):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color(color)
    return spheres


def mesh2pcd(mesh_in):
    mesh_vertices = array(mesh_in.vertices)  # nx3
    # print('pcd1_num:', len(mesh_vertices))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_vertices)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=10))

    return pcd


def mesh2np(mesh_in):
    mesh_vertices = array(mesh_in.vertices)  # nx3
    return mesh_vertices


def np2pcd(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))

    return pcd


# 数据加载
def read_mesh(mesh_path, mesh_color=[0.0, 0.6, 0.1]):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(mesh_color)

    return mesh


# 可视化
# points 顶点坐标  np格式
# lines 索引  nx2索引
def draw_line(points, lines, colors):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # print('line_set:', dir(line_set))

    return line_set

# 划线测试
# a = array([[0, 0, 0], [1, 0, 0]]) * 0.01
# b = array([[0, 0, 0], [0, 1, 0]]) * 0.01
# c = array([[0, 0, 0], [0, 0, 1]]) * 0.01
# ab = r_[a, b, c]
# print(ab)
# idx = array([[0, 1], [2, 3], [4, 5]])
# color = array([0, 0, 1])
# colors = tile(color, (len(idx), 1))


def show_pcd(pcd, color=[0.8, 0.3, 0.0]):
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([
        pcd,
        # axis_pcd,
    ],
        window_name='ANTenna3D',
        # zoom=1,
        # front=[0, 10, 0.01],  # 相机位置
        # lookat=[0, 0, 0],  # 对准的点
        # up=[0, 1, 0],  # 用于确定相机右x轴

        zoom=1,  # stanford
        front=[0, -0.1, -1],  # 相机位置
        lookat=[0, 0, 0],  # 对准的点
        up=[0, 1, 0.5],  # 用于确定相机右x轴

        # point_show_normal=True
    )
