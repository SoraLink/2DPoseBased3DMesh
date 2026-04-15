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
        """
        在同一个 Mesh 上执行多次切割。
        :param mesh_path: 原始完整 obj 路径
        :param cut_tasks: 列表，每个元素是一个 dict，包含 {'name': str, 'pt_2d': (u,v), 'start_3d': [x,y,z], 'end_3d': [x,y,z]}
        """
        print(f"\n🔪 [Mesh Cutter] 开始处理网格: {mesh_path}")
        if not cut_tasks:
            print("   -> 没有检测到有效的残肢点，跳过切割。")
            return mesh_path

        # 1. 加载 Mesh
        mesh = trimesh.load(mesh_path, process=False)

        # 2. 依次执行所有切割任务
        for task in cut_tasks:
            print(f"   -> 正在切割部位: {task['name']}")
            ray_dir = self._get_ray_direction(task['pt_2d'])
            lambda_cut = self._calculate_exact_cut_proportion(ray_dir, task['start_3d'], task['end_3d'])
            print(f"      比例 (Lambda): {lambda_cut:.4f}")

            cut_origin = task['start_3d'] + lambda_cut * (task['end_3d'] - task['start_3d'])
            cut_normal = task['start_3d'] - task['end_3d']
            cut_normal = cut_normal / np.linalg.norm(cut_normal)

            # 连续切片
            mesh = mesh.slice_plane(plane_origin=cut_origin, plane_normal=cut_normal)

        # 3. 所有切口切完后，统一封口
        try:
            mesh.fill_holes()
            print("   -> 成功完成所有切口的自动封口 (Watertight)")
        except Exception as e:
            print(f"   ⚠️ 自动封口警告: {e}")

        # 4. 保存文件
        base_dir = os.path.dirname(mesh_path)
        file_name = os.path.basename(mesh_path).split('.')[0]
        output_path = os.path.join(base_dir, f"{file_name}_truncated.obj")

        mesh.export(output_path)
        print(f"✅ [Mesh Cutter] {len(cut_tasks)} 处截肢完成，已保存至: {output_path}")

        return output_path