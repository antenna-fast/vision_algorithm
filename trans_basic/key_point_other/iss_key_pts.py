import open3d as o3d
import time


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


voxel_1 = 0.001
search_radius = 0.02
search_max_nn = 10

model_path = '../../data_ply/bun_zipper.ply'
# model_path = '../../data_ply/Armadillo.ply'

pcd = o3d.io.read_point_cloud(model_path)
pcd = pcd.voxel_down_sample(voxel_size=voxel_1)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=search_max_nn))
pcd.paint_uniform_color([0.0, 0.5, 0.1])

mesh = o3d.io.read_triangle_mesh(model_path)
mesh.compute_vertex_normals()

# Compute ISS Keypoints on Armadillo
# mesh = o3dtut.get_armadillo_mesh()
# pcd = o3d.geometry.PointCloud()
# pcd.points = mesh.vertices

tic = time.time()
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                        # salient_radius=0.005,
                                                        # non_max_radius=0.005,
                                                        # gamma_21=0.5,
                                                        # gamma_32=0.5
                                                        )

toc = 1000 * (time.time() - tic)
print("ISS Computation took {:.0f} [ms]".format(toc))

key_spher = keypoints_to_spheres(keypoints)

# mesh.compute_vertex_normals()
# mesh.paint_uniform_color([0.5, 0.5, 0.5])
keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([
        # keypoints,
        # pcd,
        mesh,
        key_spher
        ],
        # front=[0, 0, -1.0]
        )
