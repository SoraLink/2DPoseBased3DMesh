import os
import numpy as np
import trimesh


class ResidualMeshCutter:
    def __init__(self, focal_length=5000.0, img_center=(512, 512)):
        self.fx = focal_length
        self.fy = focal_length
        self.cx, self.cy = img_center
        self.cam_origin = np.array([0.0, 0.0, 0.0])

    def _get_ray_direction(self, pt_2d):
        u, v = pt_2d
        ray_x = (u - self.cx) / self.fx
        ray_y = (v - self.cy) / self.fy
        ray_z = 1.0
        ray_dir = np.array([ray_x, ray_y, ray_z])
        return ray_dir / np.linalg.norm(ray_dir)

    def _calculate_exact_cut_proportion(self, ray_dir, bone_start, bone_end):
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
        if denominator < 1e-6:
            return 0.5

        t_c = (a * e - b * d) / denominator
        return np.clip(t_c, 0.0, 1.0)

    def process_multiple_cuts(self, mesh_path, cut_tasks):
        import networkx as nx  # trimesh 内部的图论引擎

        print(f"\n🔪 [Mesh Cutter] 开始处理网格: {mesh_path}")
        if not cut_tasks:
            print("   -> 没有检测到有效的残肢点，跳过切割。")
            return mesh_path

        # 加载 Mesh
        mesh = trimesh.load(mesh_path, process=False)

        for task in cut_tasks:
            print(f"   -> 正在切割部位: {task['name']}")
            ray_dir = self._get_ray_direction(task['pt_2d'])
            lambda_cut = self._calculate_exact_cut_proportion(ray_dir, task['start_3d'], task['end_3d'])
            print(f"      比例 (Lambda): {lambda_cut:.4f}")

            # 算出精确的切点和法向量
            cut_origin = task['start_3d'] + lambda_cut * (task['end_3d'] - task['start_3d'])
            cut_normal = task['start_3d'] - task['end_3d']
            cut_normal = cut_normal / np.linalg.norm(cut_normal)

            # ==========================================
            # 🌟 新增：局部拓扑切割逻辑
            # ==========================================
            # 1. 计算所有顶点到平面的距离，找出所有“负半区”顶点
            signed_dist = np.dot(mesh.vertices - cut_origin, cut_normal)
            neg_indices = np.where(signed_dist < 0)[0]

            if len(neg_indices) == 0:
                print("      ⚠️ 未发现可切除顶点。")
                continue

            # 2. 构建拓扑图，并寻找这片顶点中的“独立孤岛” (Connected Components)
            graph = mesh.vertex_adjacency_graph
            subgraph = graph.subgraph(neg_indices)
            components = list(nx.connected_components(subgraph))

            # 3. 找出距离切口最近的那个孤岛 (即目标残肢)
            target_component = []
            min_dist = float('inf')

            for comp in components:
                comp_list = list(comp)
                # 计算该孤岛中所有顶点距离切割原点的最小距离
                dists = np.linalg.norm(mesh.vertices[comp_list] - cut_origin, axis=1)
                curr_min = np.min(dists)

                if curr_min < min_dist:
                    min_dist = curr_min
                    target_component = comp_list

            # 4. 安全阈值：如果这个孤岛确实就在切口附近(例如误差 < 15cm)，就删掉它
            if min_dist < 0.15:
                # 1. 制作一个布尔掩码，标记哪些顶点是要保留的
                keep_vertex_mask = np.ones(len(mesh.vertices), dtype=bool)
                keep_vertex_mask[target_component] = False

                # 2. 核心修复：过滤三角面！只有当一个面的 3 个顶点都要被保留时，这个面才留下
                keep_face_mask = keep_vertex_mask[mesh.faces].all(axis=1)

                # 3. 先更新(删除)无效的面
                mesh.update_faces(keep_face_mask)

                # 4. 最后清理掉那些没有面连接的孤立顶点 (也就是我们的残肢顶点)
                mesh.remove_unreferenced_vertices()

                print(f"      ✅ 成功安全切除局部肢体 (清理了对应的面和顶点)")
            else:
                print(f"      ⚠️ 孤岛距离切口过远 ({min_dist:.2f}m)，为防止误伤跳过切割。")
        # 统一封口
        # try:
        #     mesh.fill_holes()
        #     print("   -> 成功完成切口自动封口 (Watertight)")
        # except Exception as e:
        #     print(f"   ⚠️ 自动封口警告: {e}")

        # 保存文件
        base_dir = os.path.dirname(mesh_path)
        file_name = os.path.basename(mesh_path).split('.')[0]
        output_path = os.path.join(base_dir, f"{file_name}_truncated.obj")

        mesh.export(output_path)
        print(f"✅ [Mesh Cutter] 所有截肢任务完成，已保存至: {output_path}")

        return output_path