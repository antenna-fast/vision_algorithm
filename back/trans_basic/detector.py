from o3d_impl import *
from dist import *


# 这个是在pcd上的！
# 要把mesh转成pcd再过来

# 大于threshold设置成关键点

def detect_2_ring_kl(pcd_in, threshold=1, vici_num=7, cut_num=4):
    pts_num = len(pcd_in.points)

    pcd_in.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=10))
    pcd_in.paint_uniform_color([0.0, 0.6, 0.0])
    pcd_tree_1 = o3d.geometry.KDTreeFlann(pcd_in)  # 构建搜索树

    key_pts_buff_1 = []
    for i in range(pts_num):
        pick_idx = i

        # 一环上构造一个特征向量
        [k, idx_1, _] = pcd_tree_1.search_knn_vector_3d(pcd_in.points[pick_idx], vici_num)
        vici_idx_1 = idx_1[1:]

        now_pt_1 = array(pcd_in.points)[i]
        vici_pts_1 = array(pcd_in.points)[vici_idx_1]
        # all_pts = array(pcd.points)[idx_1]
        mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1, vici_pts_1)

        # 构建一环的特征
        n_fn_angle_1 = []
        for f_normal in mesh_normals:
            ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
            n_fn_angle_1.append(ang)

        n_fn_angle_1 = sort(array(n_fn_angle_1))[:cut_num]  # 规定长度

        # 二环
        kl_buff = []
        n_fn_angle_buff_1 = []

        for now_pt_2r in vici_idx_1:  # 比较二环与中心环的区别
            now_pt_1_2 = array(pcd_in.points[now_pt_2r])  # 每一个邻域的相对中心点

            # 搜索二环 邻域
            [k, idx_1_2, _] = pcd_tree_1.search_knn_vector_3d(pcd_in.points[now_pt_2r], vici_num)
            vici_idx_1_2 = idx_1_2[1:]
            vici_pts_1_2 = array(pcd_in.points)[vici_idx_1_2]
            all_pts = array(pcd_in.points)[idx_1_2]

            mesh1, mesh_normals, vtx_normal = get_mesh(now_pt_1_2, vici_pts_1_2)

            n_fn_angle = []
            for f_normal in mesh_normals:
                ang = get_cos_dist(f_normal, vtx_normal)  # 两个向量的余弦值
                n_fn_angle.append(ang)

            n_fn_angle_buff_1.append(n_fn_angle)

        for vic_ang_1 in n_fn_angle_buff_1:
            # kl
            vic_ang_1 = sort(vic_ang_1)[:cut_num]  # 规定长度
            kl_loss = get_KL(vic_ang_1, n_fn_angle_1, cut_num)  # vec1, vec2, vec_len

            kl_buff.append(kl_loss)

        kl_buff = array(kl_buff)
        # sum_var = var(var_buff)
        res = get_unbalance(kl_buff, threshold)  # 不平衡点

        if res:
            pcd_in.colors[pick_idx] = [1, 0, 0]  # 选一个点
            # key_pts_buff_1.append(now_pt_1)  # 关键点
            key_pts_buff_1.append(pick_idx)  # 索引

    return key_pts_buff_1
