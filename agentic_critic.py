import json
import math
import os
import re
import time

import cv2
import requests
from dashscope import MultiModalConversation
import numpy as np

from auto_param_builder import AutoParamBuilder
from image_ops import OSSProcessor
from pose_extractor import PoseExtractor

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

def debug_visualize_alignment(kpts_orig, kpts_aligned, save_dir="./debug"):
    """
    最简单的调试工具：把两组点画在同一个纯白画布上对比。
    蓝点是原始图(基准)，红点是对齐后的生成图。
    """
    os.makedirs(save_dir, exist_ok=True)
    # 创建 1000x1000 的纯白画布 (如果你的坐标范围更大，可以把 1000 改大点)
    canvas = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    # 1. 画原始点 (蓝色，稍微画大一点作为底)
    for pt in kpts_orig:
        if len(pt) >= 3 and pt[2] == 0:
            continue
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 6, (255, 0, 0), -1)

    # 2. 画对齐后的生成点 (红色，稍微画小一点，看能不能盖在蓝点上)
    for pt in kpts_aligned:
        if len(pt) >= 3 and pt[2] == 0:
            continue
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

    save_path = os.path.join(save_dir, f"align_{int(time.time())}.jpg")
    cv2.imwrite(save_path, canvas)
    print(f"👁️ [Debug] 点位对比图已保存至: {save_path} (蓝:原图, 红:对齐后)")


def draw_pose_skeleton(kpts_aligned, kpts_gen, save_dir="./debug_skeletons", img_shape=(1024, 1024, 3)):
    """
    根据 METAINFO 字典自动绘制连线和关键点。
    新增功能：自动将残肢线段按 kpts_gen 中的对应肢体长度进行拉伸。
    """
    os.makedirs(save_dir, exist_ok=True)
    canvas = np.zeros(img_shape, dtype=np.uint8)

    # 1. 建立名字到索引的映射字典
    name_to_id = {info['name']: info['id'] for info in METAINFO['keypoint_info'].values()}

    # 为了不污染传进来的原始对齐数据，我们 copy 一份用来画图
    kpts_to_draw = kpts_aligned.copy()

    # ==========================================
    # 🌟 新增：残肢拉伸逻辑
    # 映射格式：'残肢点名字': ('对应的关节点起点', '对应的关节点终点')
    # 例如：左大腿残肢点，对应的其实是左髋到左膝的长度
    # ==========================================
    RES_LENGTH_MAPPING = {
        'L-Elbow-Res-Above': ('left_shoulder', 'left_elbow'),
        'R-Elbow-Res-Above': ('right_shoulder', 'right_elbow'),
        'L-Elbow-Res-Below': ('left_elbow', 'left_wrist'),
        'R-Elbow-Res-Below': ('right_elbow', 'right_wrist'),
        'L-Knee-Res-Above': ('left_hip', 'left_knee'),
        'R-Knee-Res-Above': ('right_hip', 'right_knee'),
        'L-Knee-Res-Below': ('left_knee', 'left_ankle'),
        'R-Knee-Res-Below': ('right_knee', 'right_ankle')
    }

    def is_visible(pt):
        # 辅助函数：判断点是否有效 (没有v维度，或者v!=0)
        return len(pt) < 3 or pt[2] != 0

    for res_name, (gen_start_name, gen_end_name) in RES_LENGTH_MAPPING.items():
        if res_name in name_to_id and gen_start_name in name_to_id and gen_end_name in name_to_id:
            res_id = name_to_id[res_name]
            start_id = name_to_id[gen_start_name]  # 锚点 (如肩膀、髋部)
            end_id = name_to_id[gen_end_name]  # 目标点 (如手肘、膝盖)

            # 确保索引不越界，且相关点都有效 (v!=0)
            if (res_id < len(kpts_to_draw) and start_id < len(kpts_to_draw) and
                    end_id < len(kpts_gen) and start_id < len(kpts_gen)):

                # 检查所有参与计算的点是否可见
                if (is_visible(kpts_to_draw[res_id]) and is_visible(kpts_to_draw[start_id]) and
                        is_visible(kpts_gen[start_id]) and is_visible(kpts_gen[end_id])):

                    # A. 计算 kpts_gen 中生成肢体的目标长度
                    gen_start_pt = kpts_gen[start_id][:2]
                    gen_end_pt = kpts_gen[end_id][:2]
                    target_length = np.linalg.norm(gen_end_pt - gen_start_pt)

                    # B. 计算 kpts_aligned 中残肢的方向向量
                    align_start_pt = kpts_to_draw[start_id][:2]
                    align_res_pt = kpts_to_draw[res_id][:2]
                    direction_vec = align_res_pt - align_start_pt
                    current_length = np.linalg.norm(direction_vec)

                    # C. 拉长坐标：起点 + 单位方向向量 * 目标长度
                    if current_length > 1e-5:  # 防止除以0
                        unit_vec = direction_vec / current_length
                        new_res_pt = align_start_pt + unit_vec * target_length

                        # 把拉长后的新坐标写回用来画图的数组中
                        kpts_to_draw[res_id][:2] = new_res_pt
    # ==========================================

    # 2. 遍历 skeleton_info，绘制连线 (注意：这里要用 kpts_to_draw 了！)
    for skel_id, skel_info in METAINFO['skeleton_info'].items():
        name1, name2 = skel_info['link']

        if name1 in name_to_id and name2 in name_to_id:
            p1_idx = name_to_id[name1]
            p2_idx = name_to_id[name2]

            if p1_idx < len(kpts_to_draw) and p2_idx < len(kpts_to_draw):
                p1, p2 = kpts_to_draw[p1_idx], kpts_to_draw[p2_idx]

                if not is_visible(p1) or not is_visible(p2):
                    continue

                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))

                r, g, b = skel_info['color']
                color_bgr = (int(b), int(g), int(r))

                cv2.line(canvas, pt1, pt2, color_bgr, 4)

    # 3. 遍历 keypoint_info，绘制点 (注意：使用 kpts_to_draw)
    for kp_id, kp_info in METAINFO['keypoint_info'].items():
        if kp_id < len(kpts_to_draw):
            pt = kpts_to_draw[kp_id]

            if not is_visible(pt):
                continue

            pos = (int(pt[0]), int(pt[1]))

            r, g, b = kp_info['color']
            color_bgr = (int(b), int(g), int(r))

            cv2.circle(canvas, pos, 6, color_bgr, -1)

    filename = f"target_skeleton_{int(time.time() * 1000)}.jpg"
    local_path = os.path.join(save_dir, filename)
    cv2.imwrite(local_path, canvas)

    return os.path.abspath(local_path)

class PoseGeometricEvaluator:
    def __init__(self, displacement_threshold=15.0, angle_threshold_deg=10.0):
        self.disp_thresh = displacement_threshold
        self.angle_thresh = angle_threshold_deg

    def _get_intersection(self, p1, p2, p3, p4):
        """计算躯体对角线交点"""
        x1, y1, _ = p1
        x2, y2, _ = p2
        x3, y3, _ = p3
        x4, y4, _ = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py])

    def align_poses(self, kpts_orig, kpts_gen, torso_indices):
        """
        使用 OpenCV 的相似仿射变换 (平移 + 旋转 + 等比例缩放) 对齐两组关键点。
        """
        # 1. 提取稳定的参考点 (例如躯干)，只拿前两维 (x, y) 坐标
        # 假设 kpts_orig 和 kpts_gen 都是 (N, 3) 的 numpy 数组
        torso_indices = list(torso_indices.values())
        src_pts = kpts_gen[torso_indices][:, :2].astype(np.float32)
        dst_pts = kpts_orig[torso_indices][:, :2].astype(np.float32)

        # 2. 计算变换矩阵 (2x3 矩阵)
        # estimateAffinePartial2D 会返回最优的 [旋转+缩放 | 平移] 矩阵
        transform_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if transform_matrix is None:
            print("⚠️ 警告: 无法计算仿射变换矩阵，退回简单的中心点平移。")
            # 这里可以写你之前的 translation_vector 备用逻辑，防止极少数极端情况报错
            translation_vector = np.mean(dst_pts, axis=0) - np.mean(src_pts, axis=0)
            kpts_aligned = kpts_gen.copy()
            kpts_aligned[:, :2] += translation_vector
            return kpts_aligned, translation_vector

        # 3. 将计算出的变换矩阵，应用到生成图的 **所有** 关键点上
        all_src_pts = kpts_gen[:, :2].astype(np.float32)

        # cv2.transform 要求输入的形状是 (1, N, 2)
        all_src_pts_reshaped = np.array([all_src_pts])
        aligned_pts_2d = cv2.transform(all_src_pts_reshaped, transform_matrix)[0]

        # 4. 把对齐后的 2D 坐标拼回原来的 (N, 3) 数组中 (保留原来的 Z 轴或置信度)
        kpts_aligned = kpts_gen.copy()
        kpts_aligned[:, :2] = aligned_pts_2d

        return kpts_aligned, transform_matrix

    def calculate_vector_angle(self, v1, v2):
        """计算向量夹角"""
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return math.degrees(math.acos(dot_product))

    def evaluate(self, kpts_orig, kpts_gen, stable_keys, residual_vecs_list, generated_vecs_list, torso_indices, output_dir):
        kpts_aligned, _ = self.align_poses(kpts_orig, kpts_gen, torso_indices)
        debug_visualize_alignment(kpts_orig, kpts_aligned, save_dir=output_dir)
        error_reasons = []
        correction_steps = []

        # ---------------------------------------------------------
        # 1. 检查静止点位移 (收集所有超标的关节点)
        # ---------------------------------------------------------
        displaced_joints = []
        for key in stable_keys:
            # 判断在原图和生成图中，该点是否都有效 (置信度 > 0)
            if kpts_orig[key, 2] > 0 and kpts_aligned[key, 2] > 0:
                # [注意] 必须切片 [:2] 只取 X 和 Y 计算欧氏距离
                disp = np.linalg.norm(kpts_orig[key][:2] - kpts_aligned[key][:2])
                if disp > self.disp_thresh:
                    displaced_joints.append((key, disp))

        # 如果有静止点发生偏移
        if displaced_joints:
            # 仅作控制台日志用
            joint_names = ", ".join([str(k) for k, d in displaced_joints])
            error_reasons.append(f"Joints offset > {self.disp_thresh}px: {joint_names}")

        # ---------------------------------------------------------
        # 2. 检查残肢生成角度 (带相对方向提示)
        # ---------------------------------------------------------
        for res_vec_keys, gen_vec_keys in zip(residual_vecs_list, generated_vecs_list):
            v_res = kpts_orig[res_vec_keys[1]][:2] - kpts_orig[res_vec_keys[0]][:2]
            v_gen = kpts_aligned[gen_vec_keys[1]][:2] - kpts_aligned[gen_vec_keys[0]][:2]
            angle_diff = self.calculate_vector_angle(v_res, v_gen)

            if angle_diff > self.angle_thresh:
                error_reasons.append(f"Angle error {angle_diff:.1f}° on limb {gen_vec_keys}")

            # 返回结果 (干掉 correction)
        if not error_reasons:
            return {"passed": True, "reason": "Geometric alignment perfect."}
        else:
            formatted_reasons = " | ".join(error_reasons)
            return {"passed": False, "reason": f"Errors: {formatted_reasons}"}

# ==========================================
# 2. 精细校准 Agent (专职处理第二阶段)
# ==========================================
class GeometricRefinerAgent:
    def __init__(self, pose_extractor, edit_model='qwen-image-2.0-pro', disp_thresh=15.0, angle_thresh=10.0,
                 max_iterations=3):
        self.edit_model = edit_model
        self.max_iterations = max_iterations

        self.evaluator = PoseGeometricEvaluator(disp_thresh, angle_thresh)
        self.pose_extractor = pose_extractor
        self.auto_param_builder = AutoParamBuilder()

        # 实例化你的 OSS 处理器
        self.oss_processor = OSSProcessor()

        # 精细微调的专用基础指令（强调用第二张图作为骨架参考）
        self.refine_instruction = """
        [Task: Geometric Limb Calibration]
        Objective: adjust the posture of the newly generated limb in the main image to strictly match the provided reference skeleton image.

        [Strict Rules]
        1. Look at the SECOND image (the skeleton graph). This is your exact target pose.
        2. Adjust the angles and positions of the generated limbs in the FIRST image to perfectly align with the skeleton lines.
        3. Maintain photorealistic skin texture and clothing continuity.
        """

    def align_orig_to_gen(self, kpts_orig, kpts_gen, torso_indices):
        """
        把原始的完美目标位姿 (orig)，通过仿射变换对齐到当前生成图 (gen) 的躯干上。
        """
        # 确保提取的是干净的整数列表
        if isinstance(torso_indices, dict):
            torso_idx = list(torso_indices.values())
        else:
            torso_idx = [int(i) for i in torso_indices]

        # orig(src) -> gen(dst)
        src_pts = kpts_orig[torso_idx][:, :2].astype(np.float32)
        dst_pts = kpts_gen[torso_idx][:, :2].astype(np.float32)

        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if matrix is None:
            print("⚠️ 无法计算仿射变换，回退到平移对齐...")
            translation = np.mean(dst_pts, axis=0) - np.mean(src_pts, axis=0)
            kpts_aligned = kpts_orig.copy()
            kpts_aligned[:, :2] += translation
            return kpts_aligned

        pts_reshaped = np.array([kpts_orig[:, :2].astype(np.float32)])
        aligned_2d = cv2.transform(pts_reshaped, matrix)[0]

        kpts_aligned = kpts_orig.copy()
        kpts_aligned[:, :2] = aligned_2d

        return kpts_aligned

    def edit_image(self, image_url, prompt, skeleton_url=None, mask_url=None):
        print(f"\n🎨 [生成] 调用 {self.edit_model} (同步对话模式)...")

        # 1. 组装新版 API 要求的 messages 结构
        content_list = [{"image": image_url}]

        if skeleton_url:
            content_list.append({"image": skeleton_url})

        # if mask_url:
        #     content_list.append({"image": mask_url})

        content_list.append({"text": prompt.strip()})

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        try:
            # 2. 发起同步调用
            response = MultiModalConversation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=self.edit_model,
                messages=messages,
                stream=False,
                n=1,
                seed=42,
                guidance_scale=7.0,
                watermark=False,
                negative_prompt="shifting torso, changing joint angles, redrawing background, missing limbs",
                prompt_extend=False
            )

            # 3. 提取返回的干净 URL
            if response.status_code == 200:
                for content in response.output.choices[0].message.content:
                    if 'image' in content:
                        print("成功生成图像{}".format(content['image']))
                        return content['image']
                raise RuntimeError("❌ API 返回了 200，但未找到图片链接。")
            else:
                raise RuntimeError(f"❌ 图像生成失败: HTTP {response.status_code}, {response.message}")

        except Exception as e:
            raise RuntimeError(f"❌ 大模型 API 调用崩溃: {str(e)}")

    def run(self, kpts_orig, initial_gen_url, output_dir, mask_url=None):
        print("\n" + "=" * 50)
        print(f"🔬 启动第二阶段: 几何精细校准 Agent")
        print("=" * 50)
        try:
            response = requests.get(initial_gen_url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"❌ 无法下载图像 URL: {e}")

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        current_url = initial_gen_url
        generated_image_urls = [current_url]

        eval_params = self.auto_param_builder.infer_params(kpts_orig)

        for i in range(1, self.max_iterations + 1):
            print(f"\n⚙️ 几何校准轮次 {i}/{self.max_iterations}")

            # 2. 提取当前生成图的位姿
            kpts_gen = self.pose_extractor.extract_31_keypoints(current_url)

            # 3. 执行评估
            eval_res = self.evaluator.evaluate(
                kpts_orig=kpts_orig,
                kpts_gen=kpts_gen,
                stable_keys=eval_params["stable_keys"],
                residual_vecs_list=eval_params["residual_vecs_list"],
                generated_vecs_list=eval_params["generated_vecs_list"],
                torso_indices=eval_params["torso_indices"],
                output_dir=output_dir
            )

            if eval_res["passed"]:
                print("🎯 [校准通过] 所有几何和位移误差均小于阈值！")
                return generated_image_urls
            else:
                print(f"⚠️ [校准未达标] {eval_res['reason']}")

            print("🔧 生成对齐骨架图，上传 OSS 并提交重新编辑...")
            try:
                # 3.1 产生对齐后的目标骨架点 (orig 适配到当前 gen)
                kpts_target_aligned = self.align_orig_to_gen(
                    kpts_orig,
                    kpts_gen,
                    eval_params["torso_indices"]
                )

                # 3.2 在本地画出骨架图
                # 注意：如果你的图片不是 1024x1024，建议这里动态传入 cv2.imread(本地原图).shape

                local_skeleton_path = draw_pose_skeleton(kpts_target_aligned, kpts_gen, save_dir=output_dir, img_shape=img.shape)

                # 3.3 上传到 OSS 获取 URL
                skeleton_oss_url = self.oss_processor.upload_and_get_url(local_file_path=local_skeleton_path)
                print(f"☁️ 骨架图已上传至 OSS: {skeleton_oss_url}")

                # 3.4 组装 prompt，附带 correction 信息
                current_prompt = self.refine_instruction
                # 3.5 调用大模型 (传入原图 + OSS骨架图)
                time.sleep(3)
                current_url = self.edit_image(current_url, current_prompt, skeleton_url=skeleton_oss_url,
                                              mask_url=mask_url)
                generated_image_urls.append(current_url)

            except Exception as e:
                print(f"❌ 微调中断: {e}")
                raise e
            else:
                print("🛑 已达最大校准次数，返回当前最优微调结果。")

        return generated_image_urls

class AgenticImageEditor:
    def __init__(self, edit_model='qwen-image-2.0-pro', eval_model='qwen3.6-plus'):
        self.edit_model = edit_model
        self.eval_model = eval_model
        self.max_iterations = 3
        self.base_instruction = """
        [Task: Semantic-Level Limb Completion]
        Objective: Perform local inpainting and completion on the subject's missing limb parts to generate a person with four intact limbs. All four limbs including hands and feets must be clearly visible.
        
        [Constraints]:
        1. It is strictly prohibited to alter the original torso, head, and any existing intact limbs. These parts must remain exactly as they are.
        2. When generating the missing limbs, you must strictly follow the direction of the existing stump. Do not cause any joint angles to change after the completion.
        3. The entire person must be completely within the image; no part of the body should fall outside the frame.
        
        [Allowed Actions]
        1. Change the background
        """

        self.constraints = """
        [Constraints]:
        1. It is strictly prohibited to alter the original torso, head, and any existing intact limbs. These parts must remain exactly as they are.
        2. When generating the missing limbs, you must strictly follow the direction of the existing stump. Do not cause any joint angles to change after the completion.
        3. The entire person must be completely within the image; no part of the body should fall outside the frame.
        
        [Allowed Actions]
        1. Change the background
        """

    def edit_image(self, image_url, action_prompt, base_instruction, mask_url=None):
        print(f"\n🎨 [生成] 调用 {self.edit_model} (同步对话模式)...")

        # 1. 组装符合新版 API 要求的 messages 结构
        prompt = f"""
        {base_instruction}

        =========================
        [Specific Action for This Step]:
        {action_prompt.strip()}
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": prompt.strip()}
                ]
            }
        ]

        # 如果有 mask_url，也塞进 content 列表里
        if mask_url:
            messages[0]["content"].insert(1, {"image": mask_url})

        try:
            # 2. 发起同步调用 (程序会在这里耐心等待，直到阿里云把图画完)
            response = MultiModalConversation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=self.edit_model,
                messages=messages,
                stream=False,
                n=1,
                watermark=False,
                seed=42,
                guidance_scale=7.0,
                # 你的负向提示词完整保留
                negative_prompt="deformed torso, changing existing limb angles, distorted joints, redrawing entire person, extra fingers, anatomical nonsense, missing limbs",
                # 强烈建议设为 False，防止模型自己乱加词破坏你的严苛医学约束
                prompt_extend=False
            )

            # 3. 解析并返回干净的 URL
            if response.status_code == 200:
                for content in response.output.choices[0].message.content:
                    if 'image' in content:
                        result_url = content['image']
                        print(f"✅ 生成成功，新图像 URL: {result_url}")
                        return result_url
                raise RuntimeError("❌ API 返回了 200 成功，但内容里没有图片链接。")
            else:
                error_msg = f"HTTP返回码：{response.status_code}, 错误信息：{response.message}"
                raise RuntimeError(f"❌ 图像生成失败: {error_msg}")

        except Exception as e:
            # 捕获底层的网络断连、超时等异常，直接抛出清晰的报错
            raise RuntimeError(f"❌ 大模型 API 调用崩溃: {str(e)}")

    def analyze_and_plan(self, image_url, base_instruction, current_step, total_steps, previous_feedback=None):
        """
        🧠 Thinking Module: Deep thinking and planning before image generation.
        """
        print(f"\n🧠 [Thinking Module] 正在进行合规审查与下一步规划...")

        # 💡 强化：让反思上下文带上警告语气，引起大模型的高度重视
        reflection_context = ""
        if previous_feedback:
            reflection_context = f"\n[Reflection Context]: Pay critical attention to the issues raised from the last step: {previous_feedback}. You MUST fix this."

        think_prompt = f"""You are a Visual Task Decomposer. 

        [OVERALL OBJECTIVE]
        {base_instruction}

        [PROGRESS & REFLECTION CONTEXT]
        Current Progress: Step {current_step} of {total_steps}.
        {reflection_context}

        Your job is to evaluate the current image by combining the [OVERALL OBJECTIVE] with the insights from the 
        [PROGRESS & REFLECTION CONTEXT]. By comparing the current image against both the original image and the 
        historical feedback, identify what is missing or flawed, and output an action-oriented prompt to generate the 
        NEXT part.

        [STRATEGIC PACING]
        Do NOT attempt to restore all missing limbs at once. Focus on generating or fixing ONLY ONE limb per step to 
        ensure high-quality generation. You will fix the rest in subsequent steps.

        [LIMB GENERATION PROTOCOL]
        When addressing a missing limb, you must strictly follow this workflow:
        1. Observe: Closely inspect the targeted residual limb (stump) in the current image.
        2. Analyze Orientation (in thought_process): Estimate and state the general spatial direction the stump is 
        pointing (e.g., 'pointing downward and slightly to the left', 'extending straight down', or 'facing forward').
        3. Guide Generation (in edit_prompt): Incorporate this general directional cue into your action prompt so the 
        new limb extends naturally along the stump's trajectory.

        You always need to pay attention to the constraints and enforce them in your plan.

        Output ONLY a JSON object:
        {{
          "thought_process": "Step-by-step evaluation. First, synthesize the [OVERALL OBJECTIVE] with the 
          [REFLECTION CONTEXT] to define the current priority. Second, select ONE specific limb to work on. 
          Third, state its general spatial direction based on the stump and decide the adjustments needed.",
          "edit_prompt": "An ACTION-FOCUSED prompt describing ONLY how to inpaint or adjust the specific 
          missing limb targeted in this step. Ensure it clearly defines the general direction and position 
          extending from the stump. DO NOT include general scene descriptions 
          (e.g., do not describe the jersey, face, or background)."
        }}"""
        # 调用视觉大模型
        resp = MultiModalConversation.call(
            model=self.eval_model,
            messages=[{
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": think_prompt}
                ]
            }],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )

        content = resp.output.choices[0].message.content
        if isinstance(content, list):
            text_parts = [item['text'] for item in content if 'text' in item]
            content = "".join(text_parts)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

        return json.loads(content)

    def evaluate_image(self, original_url, current_url, original_prompt):
        print(f"\n🔍 [评估] 调用 {self.eval_model} 进行视觉审视...")

        eval_prompt = f"""You are a strict Image Quality Auditor. Compare the ORIGINAL and EDITED images.
        Verify if the output achieve the goal and meets these requirements: {original_prompt}

        CRITICAL INSPECTION POINTS:
        1. Are all 4 limbs (2 arms, 2 legs) present and complete?
        2. Is the original torso and existing limbs UNCHANGED?
        3. Does the new limb follow the natural direction of the stump without changing joint angles?

        Output ONLY a valid JSON object:
        {{
          "passed": true/false,
          "reason": "Detailed explanation of violations in English (e.g., 'torso distorted', 'limb angle mismatch')",
        }}"""

        resp = MultiModalConversation.call(
            model=self.eval_model,
            messages=[{
                "role": "user",
                "content": [
                    {"image": original_url},
                    {"image": current_url},
                    {"text": eval_prompt}
                ]
            }],
            temperature=0.1,
            response_format={'type': 'json_object'}
        )

        if resp.status_code != 200:
            raise RuntimeError(f"❌ VLM 评估失败: {resp.message}")

        content = resp.output.choices[0].message.content

        if isinstance(content, list):
            text_parts = [item['text'] for item in content if 'text' in item]
            content = "".join(text_parts)

        # 清洗可能混入的思考标签或 Markdown
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"passed": False, "reason": "JSON_PARSE_ERROR"}

    def run(self, image_url, mask_url=None):
        generated_image_urls = []
        current_url = image_url
        previous_feedback = None
        print(f"🚀 启动 Agentic 编辑流程 | 模型: {self.edit_model} + {self.eval_model}")

        for i in range(1, self.max_iterations + 1):
            print(f"\n{'=' * 40} 第 {i}/{self.max_iterations} 轮 {'=' * 40}")

            plan_result = self.analyze_and_plan(current_url, self.base_instruction, i, self.max_iterations, previous_feedback)
            print(f"\n💭 [Agent Thinking]:\n{plan_result['thought_process']}\n")
            final_edit_prompt = plan_result["edit_prompt"] + "\n" + self.constraints
            print(f"🎯 [Underlying Edit Prompt]:\n{final_edit_prompt}\n")
            # 1. 执行编辑
            new_generated_url = self.edit_image(current_url, final_edit_prompt, self.base_instruction, mask_url)
            generated_image_urls.append(new_generated_url)

            # 2. 自我审视（最后一轮直接返回）
            if i < self.max_iterations:
                eval_res = self.evaluate_image(image_url, new_generated_url, self.base_instruction)
                print(f"📊 评估: {'✅ 通过' if eval_res['passed'] else '❌ 未通过'}")
                print(f"📝 原因: {eval_res.get('reason', '-')}")

                if eval_res["passed"]:
                    print("🎉 约束全部满足，提前结束迭代！")
                    return generated_image_urls

                # 3. 动态注入修正指令
                print("🛠️ 注入修正指令，准备下一轮生成...")
                previous_feedback = eval_res.get('reason', 'Unknown error.')
                print(f"⚠️ Flaw detected: {eval_res.get('reason', '-')}. Recorded for reflection in the next iteration.")
                current_url = new_generated_url

            else:
                print("⚠️ 已达最大迭代次数，返回当前最佳结果。")

        return generated_image_urls