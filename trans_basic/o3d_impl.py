import open3d as o3d

from module_test.point_to_plane_test import *
from n_pt_plane import *
from base_trans import *


# open3d lib function

def get_non_manifold_vertex_mesh(verts, triangles):

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    # mesh.rotate(
    #     mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)),
    #     center=mesh.get_center(),
    # )
    return mesh


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

    # if tri_num in histo.keys():
    #     histo[tri_num] += 1
    # else:
    #     histo[tri_num] = 1

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

    return mesh, mesh_normals, normal


def get_mesh_idx(now_pt, vici_pts):
    # 得到邻域局部坐标系 得到法向量等等
    # 返回: 顶点坐标 三角形索引
    coord = get_coord(now_pt, vici_pts)  # 列向量表示三个轴
    normal = coord[:, 2]  # 第三列
    # print('coord:\n', coord)

    # 还有一步 利用法向量和中心点,得到平面方程[ABCD]
    p = get_plan(normal, now_pt)

    # * 找到拓扑结构 START
    all_pts = vstack((now_pt, vici_pts))
    # 将周围的点投影到平面
    plan_pts = []
    for pt in all_pts:  # 投影函数可以升级  向量化  map
        pt_temp = pt_to_plan(pt, p, normal)  # px p pn
        plan_pts.append(pt_temp)

    plan_pts = array(plan_pts)  # 投影到平面的点

    # 将投影后的点旋转至z轴,得到投影后的二维点
    coord_inv = inv(coord)  # 反变换

    # 首先要将平面上的点平移到原点 然后再旋转  其实不平移也是可以的，只要xy平面上的结构不变
    rota_pts = dot(coord_inv, plan_pts.T).T  # 将平面旋转到与z平行

    # rota_pts[:, 2] = 0  # 已经投影到xoy(最大平面),在此消除z向轻微抖动
    pts_2d = rota_pts[:, 0:2]

    # Delauney三角化
    tri = Delaunay(pts_2d)
    tri_idx = tri.simplices  # 三角形索引

    # 统计三角形的数量
    # tri_num = len(tri_idx)
    # print(tri_num)

    # if tri_num in histo.keys():
    #     histo[tri_num] += 1
    # else:
    #     histo[tri_num] = 1

    # 可视化二维的投影
    # plt.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices.copy())
    # plt.plot(pts_2d[:, 0], pts_2d[:, 1], 'o')
    # plt.show()

    # * 找到拓扑结构 END

    # 根据顶点和三角形索引创建mesh
    # mesh = get_non_manifold_vertex_mesh(all_pts, tri_idx)

    # 求mesh normal
    # mesh.compute_triangle_normals()
    # mesh_normals = array(mesh.triangle_normals)
    # print(mesh_normals)

    # return mesh, mesh_normals, normal
    return all_pts, tri_idx

