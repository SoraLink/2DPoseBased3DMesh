import os
from pathlib import Path

from main_pipeline import calculate_miou
from pose_extractor import read_kpts_annotation
from HSMR import ReconstructionEngine

import os
import cv2
import numpy as np
import trimesh
import trimesh.smoothing
import networkx as nx
import pymeshlab

RES_BONE_MAPPING = {
    'L-Elbow-Res-Above': ('left_shoulder', 'left_elbow'),
    'R-Elbow-Res-Above': ('right_shoulder', 'right_elbow'),
    'L-Elbow-Res-Below': ('left_elbow', 'left_wrist'),
    'R-Elbow-Res-Below': ('right_elbow', 'right_wrist'),
    'L-Knee-Res-Above': ('left_hip', 'left_knee'),
    'R-Knee-Res-Above': ('right_hip', 'right_knee'),
    'L-Knee-Res-Below': ('left_knee', 'left_ankle'),
    'R-Knee-Res-Below': ('right_knee', 'right_ankle')
}

METAINFO = {
    'dataset_name': 'ld_pros_pose',
    'classes': ('person',),
    'num_keypoints': 31,
    # === 1. 关键点定义 (完全对应截图) ===
    'keypoint_info': {
        # --- Part A: COCO 原生 17 点 (ID 0-16) ---
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle'),

        # --- Part B: 自定义残肢/假肢点 (ID 17-24) ---
        17: dict(name='L_Middle_Tip', id=17, color=[255, 0, 255], type='upper', swap='R_Middle_Tip'),
        18: dict(name='R_Middle_Tip', id=18, color=[255, 0, 255], type='upper', swap='L_Middle_Tip'),
        19: dict(name='L_Heel', id=19, color=[255, 0, 255], type='lower', swap='R_Heel'),
        20: dict(name='R_Heel', id=20, color=[255, 0, 255], type='lower', swap='L_Heel'),
        21: dict(name='L_Toe_Tip', id=21, color=[255, 0, 255], type='lower', swap='R_Toe_Tip'),
        22: dict(name='R_Toe_Tip', id=22, color=[255, 0, 255], type='lower', swap='L_Toe_Tip'),

        # 23-30: 残肢点 (Res KPs)
        23: dict(name='L-Elbow-Res-Above', id=23, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Above'),
        24: dict(name='R-Elbow-Res-Above', id=24, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Above'),
        25: dict(name='L-Elbow-Res-Below', id=25, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Below'),
        26: dict(name='R-Elbow-Res-Below', id=26, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Below'),
        27: dict(name='L-Knee-Res-Above', id=27, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Above'),
        28: dict(name='R-Knee-Res-Above', id=28, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Above'),
        29: dict(name='L-Knee-Res-Below', id=29, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Below'),
        30: dict(name='R-Knee-Res-Below', id=30, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Below'),
    },

    # === 2. 骨架连接 (根据 ID 和解剖逻辑推导) ===
    'skeleton_info': {
        # --- Part A: 基础连线 (0-16 为原生或基础结构) ---
        0: dict(link=('nose', 'left_eye'), id=0, color=[51, 153, 255]),
        1: dict(link=('nose', 'right_eye'), id=1, color=[51, 153, 255]),
        2: dict(link=('left_eye', 'left_ear'), id=2, color=[51, 153, 255]),
        3: dict(link=('right_eye', 'right_ear'), id=3, color=[51, 153, 255]),
        4: dict(link=('left_shoulder', 'right_shoulder'), id=4, color=[51, 153, 255]),
        5: dict(link=('left_shoulder', 'left_elbow'), id=5, color=[0, 255, 0]),
        6: dict(link=('left_elbow', 'left_wrist'), id=6, color=[0, 255, 0]),
        7: dict(link=('right_shoulder', 'right_elbow'), id=7, color=[255, 128, 0]),
        8: dict(link=('right_elbow', 'right_wrist'), id=8, color=[255, 128, 0]),
        9: dict(link=('left_shoulder', 'left_hip'), id=9, color=[51, 153, 255]),
        10: dict(link=('right_shoulder', 'right_hip'), id=10, color=[51, 153, 255]),
        11: dict(link=('left_hip', 'right_hip'), id=11, color=[51, 153, 255]),
        12: dict(link=('left_hip', 'left_knee'), id=12, color=[0, 255, 0]),
        13: dict(link=('left_knee', 'left_ankle'), id=13, color=[0, 255, 0]),
        14: dict(link=('right_hip', 'right_knee'), id=14, color=[255, 128, 0]),
        15: dict(link=('right_knee', 'right_ankle'), id=15, color=[255, 128, 0]),

        # --- Part B: 新增点连线 (17-22: 肢体末端) ---
        16: dict(link=('left_wrist', 'L_Middle_Tip'), id=16, color=[0, 255, 255]),
        17: dict(link=('right_wrist', 'R_Middle_Tip'), id=17, color=[255, 0, 255]),
        18: dict(link=('left_ankle', 'L_Heel'), id=18, color=[0, 255, 255]),
        19: dict(link=('left_ankle', 'L_Toe_Tip'), id=19, color=[0, 255, 255]),
        20: dict(link=('right_ankle', 'R_Heel'), id=20, color=[255, 0, 255]),
        21: dict(link=('right_ankle', 'R_Toe_Tip'), id=21, color=[255, 0, 255]),

        # --- Part C: 残肢连线 (23-30: 对应 RES_KPS) ---
        22: dict(link=('left_shoulder', 'L-Elbow-Res-Above'), id=22, color=[255, 0, 0]),
        23: dict(link=('right_shoulder', 'R-Elbow-Res-Above'), id=23, color=[255, 0, 0]),
        24: dict(link=('left_elbow', 'L-Elbow-Res-Below'), id=24, color=[255, 0, 0]),
        25: dict(link=('right_elbow', 'R-Elbow-Res-Below'), id=25, color=[255, 0, 0]),
        26: dict(link=('left_hip', 'L-Knee-Res-Above'), id=26, color=[255, 0, 0]),
        27: dict(link=('right_hip', 'R-Knee-Res-Above'), id=27, color=[255, 0, 0]),
        28: dict(link=('left_knee', 'L-Knee-Res-Below'), id=28, color=[255, 0, 0]),
        29: dict(link=('right_knee', 'R-Knee-Res-Below'), id=29, color=[255, 0, 0]),
    },

    # === 3. 翻转时对应的 ID 列表 ===
    # 这个列表非常关键，MMPose 训练时 Flip 增强就是靠这个 list 知道谁和谁互换
    # 格式：[1.0] * 25
    'joint_weights': [1.] * 31,

    # Sigma (用于 OKS 计算)，给自定义点一个默认值 0.05
    'sigmas': [
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        0.089, 0.089, 0.089, 0.089, 0.089, 0.089,
        0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
    ],
}


def project_mesh_overlay(image_path, gen_image_path, mesh, global_cam, output_dir):
    """
    因为 Gen 和 Orig 已完全对齐，直接使用统一相机内参进行投影
    """
    # 1. 提取相机内参 (由于对齐，两张图公用一套参数)
    f = global_cam['focal']  # [fx, fy]
    c = global_cam['princpt']  # [cx, cy]

    # 2. 构造投影矩阵 K (3x3)
    K = np.array([
        [f[0], 0, c[0]],
        [0, f[1], c[1]],
        [0, 0, 1]
    ])

    vertices = mesh.vertices
    pts_3d = vertices.T  # (3, N)
    valid_faces = mesh.faces.astype(np.int32)

    # 3. 执行 3D -> 2D 投影
    pts_2d_homo = K @ pts_3d
    zs = pts_2d_homo[2, :]

    # 避免除以 0，并处理相机背后的点
    zs_clamped = np.maximum(zs, 1e-5)
    u = pts_2d_homo[0, :] / zs_clamped
    v = pts_2d_homo[1, :] / zs_clamped
    projected_pts = np.stack([u, v], axis=1).astype(np.int32)

    # 4. 定义通用的渲染函数，避免代码重复
    def render_overlay(img_path, suffix):
        # 读取图片 (支持中文路径)
        img_data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        # 快速渲染 Mask
        # 只渲染 Z > 0.1 的面，防止畸变
        for face in valid_faces:
            if np.all(zs[face] > 0.1):
                pts = projected_pts[face].reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)

        # 混合叠加层 (绿色半透明)
        overlay = img.copy()
        overlay_color = np.zeros_like(img)
        overlay_color[:] = [0, 255, 0]  # 绿色

        # 混合
        cv2.addWeighted(overlay_color, 0.5, overlay, 0.5, 0, dst=overlay)

        # 应用 Mask
        mask_bool = (mask > 127)
        img_out = img.copy()
        img_out[mask_bool] = overlay[mask_bool]

        # 保存图片
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{base_name}_{suffix}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imencode(".jpg", img_out)[1].tofile(out_path)

        return out_path, mask

    # 5. 分别对 Gen 图和 Orig 图生成投影
    gen_out_path, mask_gen = render_overlay(gen_image_path, "gen_projection")
    orig_out_path, mask_orig = render_overlay(image_path, "orig_projection")

    return orig_out_path, mask_orig, gen_out_path, mask_gen


import numpy as np


def calculate_2d_mpjpe(pred_joints_3d, gt_kpts_orig, global_cam, mapping_dict):
    """
    计算 2D MPJPE (平均单关节位置误差)。
    由于图像已完全对齐，无需 M_inv，直接计算投影点与 GT 点的欧氏距离。
    """
    focal = global_cam['focal']
    princpt = global_cam['princpt']

    errors = []
    for joint_name, gt_idx in mapping_dict.items():
        if joint_name not in pred_joints_3d:
            continue

        gt_x, gt_y, conf = gt_kpts_orig[gt_idx]
        # 忽略未标注或置信度不达标的点
        if conf <= 0:
            continue

        vert = pred_joints_3d[joint_name]

        # 防止 Z 轴为 0 或点在相机背面导致计算异常
        if vert[2] <= 1e-5:
            continue

        # 1. 直接投影到原图坐标系 (Gen 和 Orig 相同)
        proj_x = focal[0] * (vert[0] / vert[2]) + princpt[0]
        proj_y = focal[1] * (vert[1] / vert[2]) + princpt[1]

        # 2. 计算 2D 欧氏距离 (像素级误差)
        err = np.linalg.norm(np.array([proj_x, proj_y]) - np.array([gt_x, gt_y]))
        errors.append(err)

    # 返回所有有效关键点误差的平均值
    return float(np.mean(errors)) if errors else 0.0

class ResidualMeshCutter2:
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
        🚀 2D 驱动逻辑：因为 Gen 和 Ori 已对齐，直接在原图坐标系下投影
        """

        # 1. 将 3D 骨头两端直接投影到原图 2D 屏幕上
        def project(p3d):
            # 防止除以 0
            z = max(p3d[2], 1e-6)
            x = self.fx * (p3d[0] / z) + self.cx
            y = self.fy * (p3d[1] / z) + self.cy
            return np.array([x, y])

        p_start = project(bone_start_3d)
        p_end = project(bone_end_3d)

        print(f"      🔍 [坐标对齐核查 - 已简化]")
        print(f"         - 标注点 (pt_2d)      : [{pt_2d[0]:.1f}, {pt_2d[1]:.1f}]")
        print(f"         - 骨骼起始投影 (p_start): [{p_start[0]:.1f}, {p_start[1]:.1f}]")
        print(f"         - 骨骼末端投影 (p_end)  : [{p_end[0]:.1f}, {p_end[1]:.1f}]")

        # 2. 计算 2D 向量
        bone_vec_2d = p_end - p_start
        target_vec_2d = pt_2d - p_start

        # 3. 计算残肢点在 2D 骨骼线段上的投影比例
        denom = np.dot(bone_vec_2d, bone_vec_2d)
        if denom < 1e-6:
            return 0.5

        lambda_2d = np.dot(target_vec_2d, bone_vec_2d) / denom

        # 限制在骨骼范围内，防止把整条腿切没或者完全没切到
        return np.clip(lambda_2d, 0.05, 0.95)

    def _apply_calibration(self, pt_2d, M_inv):
        if M_inv is None:
            return pt_2d
        point = np.array([pt_2d[0], pt_2d[1], 1.0])
        # P_gen = M_orig_to_gen @ P_orig
        new_pt = M_inv @ point
        return new_pt[:2]

    def _dist_to_bone_segment(self, vertices, bone_start, bone_end):
        """计算网格顶点到指定骨骼线段的垂直距离"""
        bone_vec = bone_end - bone_start
        length = np.linalg.norm(bone_vec)
        if length < 1e-6:
            return np.linalg.norm(vertices - bone_start, axis=1)

        bone_dir = bone_vec / length
        proj = np.dot(vertices - bone_start, bone_dir)
        proj = np.clip(proj, 0.0, length)

        closest_pts = bone_start + np.outer(proj, bone_dir)
        return np.linalg.norm(vertices - closest_pts, axis=1)

    def process_multiple_cuts(self, mesh_path, cut_tasks):
        """
        执行多处截肢任务，Watertight 封口，并强制生成抛物线生理鼓包
        """
        print(f"\n🔪 [Mesh Cutter] 正在手术 (对齐模式: 直接投影)")
        if not cut_tasks:
            return trimesh.load(mesh_path, process=False)

        mesh = trimesh.load(mesh_path, process=False)
        has_cut = False

        for task in cut_tasks:
            part_name = task.get('name', '未知部位')
            print(f"   -> 处理部位: {part_name}")

            # 直接计算 Lambda，不再进行坐标转换
            lambda_cut = self._calculate_exact_cut_proportion_2d_driven(
                task['pt_2d'], task['start_3d'], task['end_3d']
            )

            cut_origin = task['start_3d'] + lambda_cut * (task['end_3d'] - task['start_3d'])
            # 法线指向要切掉的方向（指向末端）
            cut_normal = task['start_3d'] - task['end_3d']
            cut_normal = cut_normal / np.linalg.norm(cut_normal)

            task['cut_origin'] = cut_origin
            task['cut_normal'] = cut_normal

            # 过滤逻辑
            signed_dist = np.dot(mesh.vertices - cut_origin, cut_normal)
            dists_to_bone = self._dist_to_bone_segment(mesh.vertices, task['start_3d'], task['end_3d'])

            bone_length = np.linalg.norm(task['end_3d'] - task['start_3d'])
            adaptive_radius = bone_length * 0.45  # 稍微调大一点点确保覆盖肌肉

            cut_vertices = np.where((signed_dist < 0) & (dists_to_bone < adaptive_radius))[0]

            if len(cut_vertices) == 0:
                print(f"      ⚠️ 未命中有效网格。")
                continue

            keep_vertex_mask = np.ones(len(mesh.vertices), dtype=bool)
            keep_vertex_mask[cut_vertices] = False
            mesh.update_faces(keep_vertex_mask[mesh.faces].all(axis=1))
            mesh.remove_unreferenced_vertices()

            # 连通域清理：保留最大的（身体主体）
            components = list(nx.connected_components(mesh.vertex_adjacency_graph))
            if components:
                largest = max(components, key=len)
                mask = np.zeros(len(mesh.vertices), dtype=bool)
                mask[list(largest)] = True
                mesh.update_faces(mask[mesh.faces].all(axis=1))
                mesh.remove_unreferenced_vertices()

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


def load_gt_mask(image_path):
    """
    读取包含透明背景或纯黑背景的 PNG 图片，并返回二值化 Mask。
    """
    # 1. 安全读取：使用 np.fromfile 绕过 cv2.imread 对中文/日文路径的限制
    img_data = np.fromfile(image_path, dtype=np.uint8)
    if img_data.size == 0:
        raise FileNotFoundError(f"找不到文件或文件为空: {image_path}")

    # 注意：必须使用 cv2.IMREAD_UNCHANGED 保留 Alpha (透明) 通道
    img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"图片解码失败，请检查文件是否损坏: {image_path}")

    # 2. 提取 Mask
    # 情况 A: 图片是 RGBA 格式（包含透明通道）
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        # 大于 0 的地方就是人物（前景），设为 255
        _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # 情况 B: 图片是 RGB 格式（背景是纯黑）
    elif len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # 情况 C: 图片已经是单通道灰度图
    else:
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    return mask

def main(ori_image_path, gen_image_path,reconstructor, annotation_file):
    img_ori_temp = cv2.imread(ori_image_path)
    target_h, target_w = img_ori_temp.shape[:2]
    img_gen_temp = cv2.imread(gen_image_path)
    img_gen_resized = cv2.resize(img_gen_temp, (target_w, target_h), interpolation=cv2.INTER_AREA)
    temp_gen_path = gen_image_path.replace(".png", "_resized.png")
    cv2.imwrite(temp_gen_path, img_gen_resized)
    kpts_orig, kpts, types_orig = read_kpts_annotation(ori_image_path, annotation_file)
    dir_name = os.path.dirname(temp_gen_path)
    mesh_save_path = os.path.join(dir_name, "whole_body_mesh.obj")
    mesh_save_path, pred_joints_3d, pred_cam = reconstructor.predict_mesh(ori_image_path, mesh_save_path)
    cut_tasks = []
    for i in range(23, 31):
        # 判断: 只有 type == 0 才是有效残肢点，且确保坐标数组够长
        if types_orig[i] == 0:
            res_name = METAINFO['keypoint_info'][i]['name']

            # 获取 2D 坐标 (假设 kpts_orig 的格式是 [x, y, conf])
            pt_2d_orig = kpts_orig[i][0:2]
            pt_2d_orig_homo = np.array([pt_2d_orig[0], pt_2d_orig[1]])
            # 查表找到对应的 3D 骨骼起点和终点
            if res_name in RES_BONE_MAPPING:
                start_joint_name, end_joint_name = RES_BONE_MAPPING[res_name]

                start_3d = pred_joints_3d[start_joint_name]
                end_3d = pred_joints_3d[end_joint_name]

                # 组装切割任务
                cut_tasks.append({
                    'name': res_name,
                    'pt_2d': pt_2d_orig_homo,
                    'start_3d': start_3d,
                    'end_3d': end_3d
                })
    if cut_tasks:
        # 🌟 直接使用原作者推导出的官方全局相机内参
        global_focal = pred_cam['focal']
        global_cx = pred_cam['princpt'][0]
        global_cy = pred_cam['princpt'][1]

        mesh_cutter = ResidualMeshCutter2(
            focal_length=global_focal[0],
            img_center=(global_cx, global_cy)
        )
        mesh = mesh_cutter.process_multiple_cuts(
            mesh_path=mesh_save_path,
            cut_tasks=cut_tasks,
        )
        global_cam = {
            'focal': global_focal,
            'princpt': np.array([global_cx, global_cy])
        }
        orig_proj_path, pred_mask_orig, gen_proj_path, pred_mask_gen = project_mesh_overlay(ori_image_path, ori_image_path, mesh, global_cam, dir_name)
        mask_gt = load_gt_mask(ori_image_path)
        miou_score = calculate_miou(pred_mask_orig, mask_gt)
        print(f"      -> miou_score: {miou_score:.4f}")
        INTACT_MAPPING = {METAINFO['keypoint_info'][i]['name']: i for i in range(0, 17)}
        RES_MAPPING = {METAINFO['keypoint_info'][i]['name']: i for i in range(23, 31)}
        mpjpe_intact = calculate_2d_mpjpe(pred_joints_3d, kpts_orig, global_cam, INTACT_MAPPING)
        mpjpe_residual = calculate_2d_mpjpe(pred_joints_3d, kpts_orig, global_cam, RES_MAPPING)
        print(f"📊 [量化评估] 完整关节 2D MPJPE: {mpjpe_intact:.2f} pixels")
        print(f"📊 [量化评估] 残肢端点 2D MPJPE: {mpjpe_residual:.2f} pixels")
        return miou_score, mpjpe_intact, mpjpe_residual

if __name__ == "__main__":
    reconstructor = ReconstructionEngine()
    workdir = Path('./workdir1')
    dirs = workdir.glob('*')
    miou = 0
    mpjpe_intact = 0
    mpjpe_residual = 0
    for dir in dirs:
        image_folder = Path(dir)

        # --- 极简逻辑 ---
        # 1. 拿到目录下所有文件，排除 final.png
        # 只要是文件就全收进来，不管它是 .png, .jpg 还是其他
        all_files = [str(p) for p in image_folder.iterdir() if p.is_file() and p.name != 'final.png']

        if not all_files:
            raise FileNotFoundError(f"目录 {dir} 是空的，没找到素材。")

        # 2. 字母序排序
        all_files.sort()
        # 3. 取最后一张
        gen_image_path = all_files[-1]
        ori_image_path = f'./data/eval_seg_padded/{dir.name}.png'
        current_miou, current_intact, current_residual = main(ori_image_path, gen_image_path, reconstructor, annotation_file='./data/filtered_annotations_padded_png.json')
        miou += current_miou
        mpjpe_intact += current_intact
        mpjpe_residual += current_residual
    print(f"\n📈 [最终平均评估] 平均 mIoU: {miou/len(dirs):.4f}")
    print(f"📈 [最终平均评估] 平均完整关节 2D MPJPE: {mpjpe_intact/len(dirs):.2f} pixels")
    print(f"📈 [最终平均评估] 平均残肢端点 2D MPJPE: {mpjpe_residual/len(dirs):.2f} pixels")