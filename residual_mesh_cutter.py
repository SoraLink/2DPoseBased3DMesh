import os
import cv2
import numpy as np
import trimesh
import networkx as nx


class ResidualMeshCutter:
    def __init__(self, focal_length=5000.0, img_center=(128.0, 128.0)):
        """
        初始化截肢手术刀
        :param focal_length: HMR 2.0 内部默认焦距 (相对于 256 空间)
        :param img_center: HMR 2.0 内部投影中心 (256/2 = 128)
        """
        self.fx = focal_length
        self.fy = focal_length
        self.cx, self.cy = img_center
        self.cam_origin = np.array([0.0, 0.0, 0.0])

    def _apply_calibration(self, pt_2d, M_inv):
        """
        将原始图片的 2D 坐标通过仿射矩阵 M_inv 转换到 HMR 的 256 坐标系下
        M_inv 应该包含: 1. 躯干对齐的位移/缩放  2. 从原图到 256 的缩放
        """
        if M_inv is None:
            return pt_2d
        # 转换为齐次坐标 [x, y, 1]
        point = np.array([pt_2d[0], pt_2d[1], 1.0])
        # 计算变换: P_hmr = M_inv * P_orig
        new_pt = M_inv @ point
        return new_pt[:2]

    def _get_ray_direction(self, pt_2d):
        """基于校准后的 2D 点计算 3D 射线方向"""
        u, v = pt_2d
        ray_x = (u - self.cx) / self.fx
        ray_y = (v - self.cy) / self.fy
        ray_z = 1.0
        ray_dir = np.array([ray_x, ray_y, ray_z])
        return ray_dir / np.linalg.norm(ray_dir)

    def _calculate_exact_cut_proportion(self, ray_dir, bone_start, bone_end):
        """计算 2D 射线与 3D 骨骼线段之间的最近点比例 (Lambda)"""
        bone_vec = bone_end - bone_start
        v1 = ray_dir
        v2 = bone_vec
        w0 = self.cam_origin - bone_start

        a = np.dot(v1, v1)
        b = np.dot(v1, v2)
        c = np.dot(v2, v2)
        d = np.dot(v1, w0)
        e = np.dot(v2, w0)

        denominator = a * c - b * b
        if denominator < 1e-6:  # 平行情况
            return 0.5

        t_c = (a * e - b * d) / denominator
        return np.clip(t_c, 0.0, 1.0)

    def process_multiple_cuts(self, mesh_path, cut_tasks, M_inv=None):
        """
        执行多处截肢任务
        :param cut_tasks: 列表，每个元素含 {'name', 'pt_2d', 'start_3d', 'end_3d'}
        :param M_inv: 2x3 仿射变换矩阵，负责将原始坐标系对齐到 256 空间
        """
        print(f"\n🔪 [Mesh Cutter] 正在手术，目标 Mesh: {mesh_path}")
        if not cut_tasks:
            print("   -> 无切割任务，跳过。")
            return mesh_path

        mesh = trimesh.load(mesh_path, process=False)

        for task in cut_tasks:
            print(f"   -> 处理部位: {task['name']}")

            # 1. 坐标系对齐
            pt_calibrated = self._apply_calibration(task['pt_2d'], M_inv)

            # 2. 射线投影计算
            ray_dir = self._get_ray_direction(pt_calibrated)
            lambda_cut = self._calculate_exact_cut_proportion(ray_dir, task['start_3d'], task['end_3d'])

            # 3. 确定 3D 切割面 (法向量指向要切除的肢体末端)
            cut_origin = task['start_3d'] + lambda_cut * (task['end_3d'] - task['start_3d'])
            cut_normal = task['start_3d'] - task['end_3d']
            cut_normal = cut_normal / np.linalg.norm(cut_normal)

            # 4. 局部拓扑切除 (基于孤岛检测，防止误伤躯干)
            signed_dist = np.dot(mesh.vertices - cut_origin, cut_normal)
            neg_indices = np.where(signed_dist < 0)[0]

            if len(neg_indices) == 0:
                continue

            graph = mesh.vertex_adjacency_graph
            subgraph = graph.subgraph(neg_indices)
            components = list(nx.connected_components(subgraph))

            # 寻找离切口最近的肢体孤岛
            target_component = []
            min_dist = float('inf')
            for comp in components:
                comp_list = list(comp)
                dists = np.linalg.norm(mesh.vertices[comp_list] - cut_origin, axis=1)
                curr_min = np.min(dists)
                if curr_min < min_dist:
                    min_dist = curr_min
                    target_component = comp_list

            # 安全阈值：只切除离切点 15cm 以内的孤岛
            if min_dist < 0.15:
                keep_vertex_mask = np.ones(len(mesh.vertices), dtype=bool)
                keep_vertex_mask[target_component] = False

                # 更新面：只有三个点都保留的面才留下
                keep_face_mask = keep_vertex_mask[mesh.faces].all(axis=1)
                mesh.update_faces(keep_face_mask)
                mesh.remove_unreferenced_vertices()
                print(f"      ✅ 切除成功 (Lambda: {lambda_cut:.2f})")
            else:
                print(f"      ⚠️ 坐标校准后仍偏离肢体，放弃切割以保护主体。")

        # 保存结果
        output_path = mesh_path.replace(".obj", "_truncated.obj")
        mesh.export(output_path)
        print(f"✅ [Mesh Cutter] 手术结束，保存至: {output_path}")
        return mesh