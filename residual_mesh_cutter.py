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
            print(f"   -> 处理部位: {task['name']}")

            # pt_calibrated = self._apply_calibration(task['pt_2d'], M_inv)
            # ray_dir = self._get_ray_direction(pt_calibrated)
            lambda_cut = self._calculate_exact_cut_proportion_2d_driven(task['pt_2d'], task['start_3d'], task['end_3d'])
            # lambda_cut = self._calculate_exact_cut_proportion(ray_dir, task['start_3d'], task['end_3d'])
            # debug = self.last_debug_info

            # print(f"📊 [几何鉴定] 部位: {task['name']}")
            # print(f"   -> 射线与骨骼 3D 最短距离: {debug['miss_dist']:.4f} 米")
            # print(f"   -> 射线深度 (Ray Depth): {debug['ray_depth']:.2f} 米")
            # print(f"   -> 原始 Lambda (未 Clip): {debug['lambda_raw']:.4f}")
            cut_origin = task['start_3d'] + lambda_cut * (task['end_3d'] - task['start_3d'])
            cut_normal = task['start_3d'] - task['end_3d']
            cut_normal = cut_normal / np.linalg.norm(cut_normal)

            # 🌟 关键：把切面中心和法向量存下来，给后面的“拔高鼓包”环节使用
            task['cut_origin'] = cut_origin
            task['cut_normal'] = cut_normal

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
        # 🚀 第一阶段：PyMeshLab 拓扑封口与细分 (制造致密的平坦盖子)
        # ==========================================================
        print(f"      -> 开始拓扑重建 (Watertight 封口)...")
        temp_obj_path = mesh_path.replace(".obj", "_temp_hole.obj")
        mesh.export(temp_obj_path)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_obj_path)

        try:
            ms.meshing_close_holes(maxholesize=3000, newfaceselected=True)
            # 迭代2次细分，给盖子铺满密集的顶点
            ms.meshing_surface_subdivision_midpoint(iterations=2, selected=True)
        except Exception as e:
            print(f"      ⚠️ PyMeshLab 处理异常: {e}")

        ms.save_current_mesh(output_path)
        if os.path.exists(temp_obj_path):
            os.remove(temp_obj_path)

        # ==========================================================
        # 🚀 第二阶段：Trimesh 底层顶点干预 (真正的物理顶出鼓包)
        # ==========================================================
        print(f"      -> 开始施加端点物理膨胀...")
        sealed_mesh = trimesh.load(output_path, process=False)

        for task in cut_tasks:
            if 'cut_origin' not in task:
                continue

            c_origin = task['cut_origin']
            c_normal = task['cut_normal']

            # 计算所有顶点到这个截断面中心的距离
            distances = np.linalg.norm(sealed_mesh.vertices - c_origin, axis=1)

            # 设定残肢端点的波及半径 (这里设为 8 厘米，一般人类大腿/小腿切面差不多这么大)
            radius = 0.08
            mask = distances < radius

            if np.any(mask):
                # 核心魔法：使用抛物线方程。越靠近截面中心，向外拔出的力量越大；越靠近大腿边缘，力量越小 (平滑过渡)
                weights = np.clip(1.0 - (distances[mask] / radius) ** 2, 0.0, 1.0)

                # 🎯 想要多鼓，就调这个参数！目前是向外顶出 4 厘米 (0.04)。如果觉得不够，改成 0.06 或 0.08
                max_bulge = 0.06
                displacement = np.outer(weights * max_bulge, -c_normal)

                # 暴力干预：让这批顶点沿着法向量突围
                sealed_mesh.vertices[mask] += displacement

        # 最后，进行一次轻量级的全局网格平滑，把刚才我们手动拉扯的接缝处抹平，让半球完美融入大腿
        trimesh.smoothing.filter_laplacian(sealed_mesh, iterations=4)

        sealed_mesh.export(output_path)
        print("      ✅ 已成功生成完美弧度残肢端点。")

        return sealed_mesh