import os
import cv2
import numpy as np
import trimesh
import networkx as nx
import pymeshlab

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
        执行多处截肢任务，并使用 PyMeshLab 进行 Watertight 弧度封口
        """
        print(f"\n🔪 [Mesh Cutter] 正在手术，目标 Mesh: {mesh_path}")
        if not cut_tasks:
            print("   -> 无切割任务，跳过。")
            return trimesh.load(mesh_path, process=False)

        mesh = trimesh.load(mesh_path, process=False)
        has_cut = False

        for task in cut_tasks:
            print(f"   -> 处理部位: {task['name']}")

            pt_calibrated = self._apply_calibration(task['pt_2d'], M_inv)
            ray_dir = self._get_ray_direction(pt_calibrated)
            lambda_cut = self._calculate_exact_cut_proportion(ray_dir, task['start_3d'], task['end_3d'])

            cut_origin = task['start_3d'] + lambda_cut * (task['end_3d'] - task['start_3d'])
            cut_normal = task['start_3d'] - task['end_3d']
            cut_normal = cut_normal / np.linalg.norm(cut_normal)

            signed_dist = np.dot(mesh.vertices - cut_origin, cut_normal)
            neg_indices = np.where(signed_dist < 0)[0]

            if len(neg_indices) == 0:
                continue

            graph = mesh.vertex_adjacency_graph
            subgraph = graph.subgraph(neg_indices)
            components = list(nx.connected_components(subgraph))

            target_component = []
            min_dist = float('inf')
            for comp in components:
                comp_list = list(comp)
                dists = np.linalg.norm(mesh.vertices[comp_list] - cut_origin, axis=1)
                curr_min = np.min(dists)
                if curr_min < min_dist:
                    min_dist = curr_min
                    target_component = comp_list

            if min_dist < 0.15:
                keep_vertex_mask = np.ones(len(mesh.vertices), dtype=bool)
                keep_vertex_mask[target_component] = False

                keep_face_mask = keep_vertex_mask[mesh.faces].all(axis=1)
                mesh.update_faces(keep_face_mask)
                mesh.remove_unreferenced_vertices()
                print(f"      ✅ 切除成功 (Lambda: {lambda_cut:.2f})")
                has_cut = True
            else:
                print(f"      ⚠️ 坐标校准后仍偏离肢体，放弃切割以保护主体。")

        output_path = mesh_path.replace(".obj", "_truncated.obj")

        if not has_cut:
            mesh.export(output_path)
            return mesh

        # ==========================================================
        # 🚀 PyMeshLab 后处理：Watertight 封口与平滑
        # ==========================================================
        print(f"      -> 开始拓扑重建 (Watertight 封口)...")

        # 1. 保存中间的非水密网格（带有截断面空洞）
        temp_obj_path = mesh_path.replace(".obj", "_temp_hole.obj")
        mesh.export(temp_obj_path)

        # 2. 启动 PyMeshLab 管道
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_obj_path)

        try:
            # 3. 封口滤波器
            # maxholesize=3000: 足够大，能包容大腿根部级别的空洞
            # newfaceselected=True: 神仙参数，只选中刚刚补上去的盖子面
            ms.meshing_close_holes(maxholesize=3000, newfaceselected=True)

            # 4. Laplacian 坐标平滑
            # 只针对 selected=True (刚才补的盖子) 进行平滑
            # steps=3/4 通常能拉出一个非常自然的半球状生理弧度
            ms.apply_coord_laplacian_smoothing(steps=4, selected=True)
            print("      ✅ 拓扑空洞已修复，并生成弧度端点。")

        except Exception as e:
            print(f"      ⚠️ PyMeshLab 封口处理异常: {e}")

        # 5. 保存最终 Watertight Mesh
        ms.save_current_mesh(output_path)

        # 6. 清理临时文件
        if os.path.exists(temp_obj_path):
            os.remove(temp_obj_path)

        print(f"✅ [Mesh Cutter] 手术结束，保存至: {output_path}")

        # 重新加载为 trimesh 对象返回，保证跟你之前的 Pipeline 兼容
        return trimesh.load(output_path, process=False)