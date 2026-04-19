import os
import cv2
import numpy as np
import trimesh
import trimesh.smoothing
import networkx as nx
import pymeshlab

class ResidualMeshCutter:
    def __init__(self, focal_length, img_center):
        """
        初始化截肢手术刀
        :param focal_length: HMR 2.0 内部默认焦距 (相对于 256 空间)
        :param img_center: HMR 2.0 内部投影中心 (256/2 = 128)
        """
        self.fx = focal_length
        self.fy = focal_length
        self.cx, self.cy = img_center
        self.cam_origin = np.array([0.0, 0.0, 0.0])

    def _get_ray_direction(self, pt_2d):
        """直接基于原图 2D 点计算 3D 射线方向"""
        u, v = pt_2d
        # 直接使用传入的原图坐标和绝对光心
        ray_x = (u - self.cx) / self.fx
        ray_y = (v - self.cy) / self.fy
        ray_z = 1.0
        ray_dir = np.array([ray_x, ray_y, ray_z])
        return ray_dir / np.linalg.norm(ray_dir)

    def _calculate_exact_cut_proportion_2d_driven(self, pt_2d, bone_start_3d, bone_end_3d):
        """
        🚀 博士级修复：利用 2D 投影比例来锁定 3D 切割点
        """

        # 1. 将 3D 骨头两端投影到 2D 屏幕上
        def project(p3d):
            x = self.fx * (p3d[0] / p3d[2]) + self.cx
            y = self.fy * (p3d[1] / p3d[2]) + self.cy
            return np.array([x, y])

        p_start = project(bone_start_3d)
        p_end = project(bone_end_3d)

        # 👇 加入这三行灵魂拷问
        print(f"      🔍 [2D 坐标对齐核查]")
        print(f"         - 你的标注点 (pt_2d)  : [{pt_2d[0]:.1f}, {pt_2d[1]:.1f}]")
        print(f"         - 髋部投影点 (p_start): [{p_start[0]:.1f}, {p_start[1]:.1f}]")
        print(f"         - 膝盖投影点 (p_end)  : [{p_end[0]:.1f}, {p_end[1]:.1f}]")


        # 2. 计算 2D 向量
        bone_vec_2d = p_end - p_start
        target_vec_2d = pt_2d - p_start

        # 3. 计算残肢点在 2D 骨骼线段上的比例 (点乘投影)
        denom = np.dot(bone_vec_2d, bone_vec_2d)
        if denom < 1e-6:
            return 0.5

        lambda_2d = np.dot(target_vec_2d, bone_vec_2d) / denom

        # 4. 这里的 lambda 就是你要的“正数”了！
        # 即使 Lambda 稍微出界，我们也给它一个合理的范围
        return np.clip(lambda_2d, 0.05, 0.95)

    def _calculate_exact_cut_proportion(self, ray_dir, bone_start, bone_end):
        """计算 2D 射线与 3D 骨骼线段之间的最近点比例 (Lambda)"""
        # ... (此段数学逻辑完全不变，保持原样) ...
        bone_vec = bone_end - bone_start
        v1 = ray_dir
        v2 = bone_vec
        w0 = self.cam_origin - bone_start

        a = np.dot(v1, v1)
        b = np.dot(v1, v2)
        c = np.dot(v2, v2)
        d = np.dot(v1, w0)
        e = np.dot(v2, w0)

        denom = a * c - b * b
        if denom < 1e-6:
            return 0.5

        t_r = (b * e - c * d) / denom
        t_c = (a * e - b * d) / denom

        # 🌟 关键 Debug 变量：计算 3D 空间中的最短距离
        # 射线上的最近点
        p_ray = self.cam_origin + t_r * ray_dir
        # 骨骼上的最近点 (限制在 0-1 之间)
        lambda_clamped = np.clip(t_c, 0.0, 1.0)
        p_bone = bone_start + lambda_clamped * bone_vec

        # 3D 空间中的物理距离 (米)
        miss_distance = np.linalg.norm(p_ray - p_bone)

        # 将这个值存入 self 以便后续打印
        self.last_debug_info = {
            'miss_dist': miss_distance,
            'lambda_raw': t_c,
            'ray_depth': t_r
        }

        return lambda_clamped

    def _apply_calibration(self, pt_2d, M_inv):
        if M_inv is None:
            return pt_2d
        point = np.array([pt_2d[0], pt_2d[1], 1.0])
        # P_gen = M_orig_to_gen @ P_orig
        new_pt = M_inv @ point
        return new_pt[:2]

    def _dist_to_bone_segment(self, vertices, bone_start, bone_end):
        """计算网格所有顶点到指定骨骼线段的垂直距离，用于划定专属手术区"""
        bone_vec = bone_end - bone_start
        length = np.linalg.norm(bone_vec)
        if length < 1e-6:
            return np.linalg.norm(vertices - bone_start, axis=1)

        bone_dir = bone_vec / length
        proj = np.dot(vertices - bone_start, bone_dir)
        proj = np.clip(proj, 0.0, length)

        closest_pts = bone_start + np.outer(proj, bone_dir)
        return np.linalg.norm(vertices - closest_pts, axis=1)

    def process_multiple_cuts(self, mesh_path, cut_tasks, M_inv=None):
        """
        执行多处截肢任务，Watertight 封口，并强制生成抛物线生理鼓包
        """
        print(f"\n🔪 [Mesh Cutter] 正在手术，目标 Mesh: {mesh_path}")
        if not cut_tasks:
            print("   -> 无切割任务，跳过。")
            return trimesh.load(mesh_path, process=False)

        mesh = trimesh.load(mesh_path, process=False)
        has_cut = False

        for task in cut_tasks:
            part_name = task.get('name', '未知部位')
            print(f"   -> 处理部位: {part_name}")

            # 使用你最新的 2D 投影比例算法
            lambda_cut = self._calculate_exact_cut_proportion_2d_driven(task['pt_2d'], task['start_3d'], task['end_3d'])

            cut_origin = task['start_3d'] + lambda_cut * (task['end_3d'] - task['start_3d'])
            cut_normal = task['start_3d'] - task['end_3d']
            cut_normal = cut_normal / np.linalg.norm(cut_normal)

            # 存下来给鼓包用
            task['cut_origin'] = cut_origin
            task['cut_normal'] = cut_normal

            # 1. 计算截面 (法平面以下)
            signed_dist = np.dot(mesh.vertices - cut_origin, cut_normal)

            # 2. 计算顶点到当前骨骼的距离 (专属手术区)
            dists_to_bone = self._dist_to_bone_segment(mesh.vertices, task['start_3d'], task['end_3d'])

            # 3. 自适应护盾半径 (骨头的 40% 粗细)
            bone_length = np.linalg.norm(task['end_3d'] - task['start_3d'])
            adaptive_radius = bone_length * 0.40

            # 4. 锁定要被切掉的肉 (必须在刀刃下方，且在这根骨头附近)
            cut_vertices = np.where((signed_dist < 0) & (dists_to_bone < adaptive_radius))[0]

            if len(cut_vertices) == 0:
                print(f"      ⚠️ 未命中有效网格，跳过。")
                continue

            # 5. 挖掉选中的肉
            keep_vertex_mask = np.ones(len(mesh.vertices), dtype=bool)
            keep_vertex_mask[cut_vertices] = False

            keep_face_mask = keep_vertex_mask[mesh.faces].all(axis=1)
            mesh.update_faces(keep_face_mask)
            mesh.remove_unreferenced_vertices()

            # 6. 核心修复：只保留最大的连通块 (身体)，悬空的残肢直接当垃圾丢弃
            graph_after = mesh.vertex_adjacency_graph
            components_after = list(nx.connected_components(graph_after))

            if components_after:
                largest_component = max(components_after, key=len)
                keep_body_mask = np.zeros(len(mesh.vertices), dtype=bool)
                keep_body_mask[list(largest_component)] = True

                mesh.update_faces(keep_body_mask[mesh.faces].all(axis=1))
                mesh.remove_unreferenced_vertices()

            print(f"      ✅ 切除成功 (Lambda: {lambda_cut:.2f})")
            has_cut = True

        output_path = mesh_path.replace(".obj", "_truncated.obj")

        if not has_cut:
            mesh.export(output_path)
            return mesh

        # ==========================================================
        # 第一阶段：PyMeshLab 拓扑封口与细分
        # ==========================================================
        print(f"      -> 开始拓扑重建 (Watertight 封口)...")
        temp_obj_path = mesh_path.replace(".obj", "_temp_hole.obj")
        mesh.export(temp_obj_path)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_obj_path)

        try:
            ms.meshing_close_holes(maxholesize=3000, newfaceselected=True)
            ms.meshing_surface_subdivision_midpoint(iterations=2, selected=True)
        except Exception as e:
            print(f"      ⚠️ PyMeshLab 处理异常: {e}")

        ms.save_current_mesh(output_path)
        if os.path.exists(temp_obj_path):
            os.remove(temp_obj_path)

        # ==========================================================
        # 第二阶段：Trimesh 自适应物理顶出鼓包
        # ==========================================================
        print(f"      -> 开始施加端点物理膨胀...")
        sealed_mesh = trimesh.load(output_path, process=False)

        for task in cut_tasks:
            if 'cut_origin' not in task:
                continue

            c_origin = task['cut_origin']
            c_normal = task['cut_normal']

            distances = np.linalg.norm(sealed_mesh.vertices - c_origin, axis=1)

            # 让鼓包的范围和突起程度也自适应骨骼长度
            bone_len_for_bulge = np.linalg.norm(task['end_3d'] - task['start_3d'])
            bulge_radius = bone_len_for_bulge * 0.40

            mask = distances < bulge_radius

            if np.any(mask):
                weights = np.clip(1.0 - (distances[mask] / bulge_radius) ** 2, 0.0, 1.0)

                # 鼓包突起程度 (骨头长度的 15%)
                max_bulge = bone_len_for_bulge * 0.15
                displacement = np.outer(weights * max_bulge, -c_normal)

                sealed_mesh.vertices[mask] += displacement

        trimesh.smoothing.filter_laplacian(sealed_mesh, iterations=4)
        sealed_mesh.export(output_path)
        print("      ✅ 已成功生成完美弧度残肢端点。")

        return sealed_mesh