import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

from main_pipeline import calculate_miou
from pose_extractor import read_kpts_annotation
from sam_3d_body_recon import ReconstructionEngine

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

RESIDUAL_TO_TERMINAL = {
    "L-Elbow-Res-Above": "L_Middle_Tip",
    "L-Elbow-Res-Below": "L_Middle_Tip",
    "R-Elbow-Res-Above": "R_Middle_Tip",
    "R-Elbow-Res-Below": "R_Middle_Tip",

    "L-Knee-Res-Above": "L_Toe_Tip",
    "L-Knee-Res-Below": "L_Toe_Tip",
    "R-Knee-Res-Above": "R_Toe_Tip",
    "R-Knee-Res-Below": "R_Toe_Tip",
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

def _collect_aa_eval_sets(pred_joints_3d, gt_3d_kpts, kpt_types_orig=None):
    """
    收集用于 AA-MPJPE / AA-PA-MPJPE 的点集
    返回:
        names, eval_pred, eval_gt, body_pred, body_gt
    """
    gt_joints_3d = normalize_gt_3d_keypoints(gt_3d_kpts)
    if gt_joints_3d is None:
        return None, None, None, None, None

    names = []
    eval_pred = []
    eval_gt = []

    body_pred = []
    body_gt = []

    # 标准 body joints: 0-16
    for idx in range(0, 17):
        name = METAINFO["keypoint_info"][idx]["name"]

        if kpt_types_orig is not None and idx < len(kpt_types_orig):
            if kpt_types_orig[idx] != 0:
                continue

        if name not in pred_joints_3d or name not in gt_joints_3d:
            continue

        p_pred = _to_vec3(pred_joints_3d[name])
        p_gt = _to_vec3(gt_joints_3d[name])

        if p_pred is None or p_gt is None:
            continue

        names.append(name)
        eval_pred.append(p_pred)
        eval_gt.append(p_gt)

        body_pred.append(p_pred)
        body_gt.append(p_gt)

    # residual endpoints: 23-30
    for idx in range(23, 31):
        name = METAINFO["keypoint_info"][idx]["name"]

        if kpt_types_orig is not None and idx < len(kpt_types_orig):
            if kpt_types_orig[idx] != 0:
                continue

        if name not in pred_joints_3d or name not in gt_joints_3d:
            continue

        p_pred = _to_vec3(pred_joints_3d[name])
        p_gt = _to_vec3(gt_joints_3d[name])

        if p_pred is None or p_gt is None:
            continue

        names.append(name)
        eval_pred.append(p_pred)
        eval_gt.append(p_gt)

    if len(eval_pred) == 0:
        return None, None, None, None, None

    return (
        names,
        np.asarray(eval_pred, dtype=np.float64),
        np.asarray(eval_gt, dtype=np.float64),
        np.asarray(body_pred, dtype=np.float64) if len(body_pred) > 0 else None,
        np.asarray(body_gt, dtype=np.float64) if len(body_gt) > 0 else None,
    )


def _dict_from_names_and_points(names, points):
    return {name: points[i] for i, name in enumerate(names)}


def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def _draw_skeleton(ax, joints_dict, color, label_prefix=""):
    """
    joints_dict: {name: np.array([x,y,z])}
    """
    # 画点
    xs, ys, zs = [], [], []
    for name, p in joints_dict.items():
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])

    ax.scatter(xs, ys, zs, c=color, s=28)

    # 画骨架连线
    for _, info in METAINFO["skeleton_info"].items():
        a, b = info["link"]
        if a in joints_dict and b in joints_dict:
            pa = joints_dict[a]
            pb = joints_dict[b]
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]],
                c=color,
                linewidth=2
            )


def _draw_named_points(ax, joints_dict, color):
    for name, p in joints_dict.items():
        if "Res" in name:
            ax.text(p[0], p[1], p[2], name, color=color, fontsize=8)


def visualize_aa_alignment(pred_joints_3d, gt_3d_kpts, kpt_types_orig, save_prefix):
    """
    生成两张图:
      1) root-aligned pred vs gt
      2) PA-aligned pred vs gt
    """
    if gt_3d_kpts is None:
        print("No 3D GT, skip visualization.")
        return

    names, eval_pred, eval_gt, body_pred, body_gt = _collect_aa_eval_sets(
        pred_joints_3d, gt_3d_kpts, kpt_types_orig
    )

    if names is None:
        print("No valid 3D joints to visualize.")
        return

    gt_joints_3d = normalize_gt_3d_keypoints(gt_3d_kpts)
    pred_root = _get_pelvis_root(pred_joints_3d)
    gt_root = _get_pelvis_root(gt_joints_3d)

    if pred_root is None or gt_root is None:
        print("No valid pelvis root, skip visualization.")
        return

    # -----------------------------
    # 1) Root-aligned visualization
    # -----------------------------
    eval_pred_rooted = eval_pred - pred_root
    eval_gt_rooted = eval_gt - gt_root

    pred_rooted_dict = _dict_from_names_and_points(names, eval_pred_rooted)
    gt_rooted_dict = _dict_from_names_and_points(names, eval_gt_rooted)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    _draw_skeleton(ax, gt_rooted_dict, color='green')
    _draw_skeleton(ax, pred_rooted_dict, color='red')
    _draw_named_points(ax, gt_rooted_dict, color='green')
    _draw_named_points(ax, pred_rooted_dict, color='red')

    ax.set_title("Root-aligned: GT (green) vs Pred (red)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    _set_axes_equal(ax)

    root_vis_path = f"{save_prefix}_root_aligned.png"
    plt.tight_layout()
    plt.savefig(root_vis_path, dpi=200)
    plt.close(fig)

    # -----------------------------
    # 2) PA-aligned visualization
    # -----------------------------
    if body_pred is not None and len(body_pred) >= 3:
        transform = _similarity_transform_from_points(body_pred, body_gt)

        if transform is not None:
            eval_pred_pa = _apply_similarity_transform(eval_pred, transform)

            pred_pa_dict = _dict_from_names_and_points(names, eval_pred_pa)
            gt_dict = _dict_from_names_and_points(names, eval_gt)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            _draw_skeleton(ax, gt_dict, color='green')
            _draw_skeleton(ax, pred_pa_dict, color='red')
            _draw_named_points(ax, gt_dict, color='green')
            _draw_named_points(ax, pred_pa_dict, color='red')

            ax.set_title("PA-aligned: GT (green) vs Pred (red)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=20, azim=-60)
            _set_axes_equal(ax)

            pa_vis_path = f"{save_prefix}_pa_aligned.png"
            plt.tight_layout()
            plt.savefig(pa_vis_path, dpi=200)
            plt.close(fig)

            print(f"✅ Saved visualization: {root_vis_path}")
            print(f"✅ Saved visualization: {pa_vis_path}")
        else:
            print("PA transform failed, only saved root-aligned visualization.")
    else:
        print("Not enough body joints for PA visualization.")

def visualize_raw_keypoints(img_path, raw_joints_2d, raw_joints_3d, global_cam, out_path, search_range=35):
    """
    不管名字，直接遍历底层数组，把 Index (0, 1, 2...) 画在图上找手腕
    """
    img = cv2.imread(img_path)
    if img is None:
        return

    focal = global_cam['focal']
    princpt = global_cam['princpt']

    # 限制画点的数量，防止 70 个点把人脸和身体全糊住
    num_points = min(search_range, len(raw_joints_2d))

    for i in range(num_points):
        # 🔴 1. 画 2D 原生点 (红色)
        pt2d = raw_joints_2d[i]
        cv2.circle(img, (int(pt2d[0]), int(pt2d[1])), 4, (0, 0, 255), -1)
        # 直接写上数字 ID
        cv2.putText(img, f"{i}", (int(pt2d[0]) - 10, int(pt2d[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # 🔵 2. 画 3D 投影点 (蓝色，用来双重确认)
        if raw_joints_3d is not None and i < len(raw_joints_3d):
            vert = raw_joints_3d[i]
            if vert[2] > 1e-5:
                proj_x = focal[0] * (vert[0] / vert[2]) + princpt[0]
                proj_y = focal[1] * (vert[1] / vert[2]) + princpt[1]
                cv2.circle(img, (int(proj_x), int(proj_y)), 4, (255, 0, 0), -1)
                cv2.putText(img, f"{i}", (int(proj_x) + 5, int(proj_y) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    # 图例
    cv2.putText(img, "Red: 2D Raw Index", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Blue: 3D Proj Index", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite(out_path, img)
    print(f"      🔎 原始索引雷达图已保存，快去抓出手腕的 ID 吧！: {out_path}")

def visualize_keypoints_comparison(img_path, pred_joints_3d, global_cam, out_path):
    """
    将 2D 预测点(红) 与 3D 投影点(蓝) 画在同一张图上，并标注 0-16 的序号
    """
    img = cv2.imread(img_path)
    if img is None:
        return

    focal = global_cam['focal']
    princpt = global_cam['princpt']

    # 严格按照 0-16 的顺序遍历 COCO 17 点
    coco_keys = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    for i, key in enumerate(coco_keys):
        # 🔵 2. 绘制 3D 坐标强行投影回 2D 的点 (蓝色)
        if key in pred_joints_3d:
            vert = pred_joints_3d[key]
            if vert[2] > 1e-5:
                proj_x = focal[0] * (vert[0] / vert[2]) + princpt[0]
                proj_y = focal[1] * (vert[1] / vert[2]) + princpt[1]
                cv2.circle(img, (int(proj_x), int(proj_y)), 4, (255, 0, 0), -1)
                # 在点的右下方写上编号，防止和红色字重叠
                cv2.putText(img, f"3D:{i}", (int(proj_x) + 5, int(proj_y) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 添加图例
    cv2.putText(img, "Red: SAM 2D Pred", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Blue: SAM 3D Proj", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite(out_path, img)
    print(f"      👀 2D/3D 关键点可视化对比已保存至: {out_path}")


def project_mesh_overlay(image_path, gen_image_path, mesh, global_cam, output_dir, is_full=False):
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
    if is_full:
        gen_out_path, mask_gen = render_overlay(gen_image_path, "gen_projection_full")
        orig_out_path, mask_orig = render_overlay(image_path, "orig_projection_full")
    else:
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

GT3D_RAW_TO_PRED_NAME = {
    "Nose": "nose",
    "L_Eye": "left_eye",
    "R_Eye": "right_eye",
    "L_Ear": "left_ear",
    "R_Ear": "right_ear",
    "L_Shoulder": "left_shoulder",
    "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow",
    "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist",
    "R_Wrist": "right_wrist",
    "L_Hip": "left_hip",
    "R_Hip": "right_hip",
    "L_Knee": "left_knee",
    "R_Knee": "right_knee",
    "L_Ankle": "left_ankle",
    "R_Ankle": "right_ankle",

    "L_Finger": "L_Middle_Tip",
    "R_Finger": "R_Middle_Tip",
    "L_Heel": "L_Heel",
    "R_Heel": "R_Heel",
    "L_Toe": "L_Toe_Tip",
    "R_Toe": "R_Toe_Tip",
}

GT3D_RESIDUAL_PAIR_TO_OUTPUT = [
    ("Residual_L_Upperarm_Front", "Residual_L_Upperarm_Back", "L-Elbow-Res-Above"),
    ("Residual_R_Upperarm_Front", "Residual_R_Upperarm_Back", "R-Elbow-Res-Above"),
    ("Residual_L_Forearm_Front", "Residual_L_Forearm_Back", "L-Elbow-Res-Below"),
    ("Residual_R_Forearm_Front", "Residual_R_Forearm_Back", "R-Elbow-Res-Below"),
    ("Residual_L_Tigh_Front", "Residual_L_Tigh_Back", "L-Knee-Res-Above"),
    ("Residual_R_Tigh_Front", "Residual_R_Tigh_Back", "R-Knee-Res-Above"),
    ("Residual_L_Calf_Front", "Residual_L_Calf_Back", "L-Knee-Res-Below"),
    ("Residual_R_Calf_Front", "Residual_R_Calf_Back", "R-Knee-Res-Below"),
]


def _to_vec3(v):
    if v is None or len(v) < 3:
        return None
    try:
        arr = np.asarray(v[:3], dtype=np.float32)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _avg_vec3(v1, v2):
    p1 = _to_vec3(v1)
    p2 = _to_vec3(v2)

    if p1 is None and p2 is None:
        return None
    if p1 is not None and p2 is None:
        return p1
    if p2 is not None and p1 is None:
        return p2
    return (p1 + p2) / 2.0


def get_output_keypoint_names():
    return [METAINFO["keypoint_info"][i]["name"] for i in range(31)]


def _flat_3d_keypoints_to_dict(keypoints_3d):
    """
    New merged format:
        keypoints_3d = [x, y, z, v, x, y, z, v, ...]
    """
    if keypoints_3d is None:
        return None

    out = {}

    for idx in range(31):
        name = METAINFO["keypoint_info"][idx]["name"]
        base = idx * 4

        if base + 3 >= len(keypoints_3d):
            continue

        x, y, z, v = keypoints_3d[base:base + 4]

        if v <= 0:
            continue

        p = np.asarray([x, y, z], dtype=np.float32)
        if np.all(np.isfinite(p)):
            out[name] = p

    return out

def normalize_gt_3d_keypoints(gt_3d_kpts):
    if gt_3d_kpts is None:
        return None

    # New merged annotation format
    if isinstance(gt_3d_kpts, dict) and "keypoints_3d" in gt_3d_kpts:
        return _flat_3d_keypoints_to_dict(gt_3d_kpts["keypoints_3d"])

    # Old raw global_3d_keypoints.json format
    out = {}

    for raw_name, pred_name in GT3D_RAW_TO_PRED_NAME.items():
        p = _to_vec3(gt_3d_kpts.get(raw_name))
        if p is not None:
            out[pred_name] = p

    for front_name, back_name, out_name in GT3D_RESIDUAL_PAIR_TO_OUTPUT:
        p = _avg_vec3(gt_3d_kpts.get(front_name), gt_3d_kpts.get(back_name))
        if p is not None:
            out[out_name] = p

    return out

def _get_pelvis_root(joints_dict):
    """
    Root joint = average(left_hip, right_hip).
    """
    if joints_dict is None:
        return None

    left = _to_vec3(joints_dict.get("left_hip"))
    right = _to_vec3(joints_dict.get("right_hip"))

    if left is None or right is None:
        return None

    return (left + right) / 2.0


def _similarity_transform_from_points(src, dst):
    """
    Estimate similarity transform from src to dst using Procrustes alignment.

    src: (N, 3), predicted points
    dst: (N, 3), GT points

    Return:
        scale, R, t
    where:
        aligned = scale * (R @ src.T).T + t
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        return None

    if src.shape[0] < 3:
        return None

    src_t = src.T  # 3 x N
    dst_t = dst.T  # 3 x N

    mu_src = np.mean(src_t, axis=1, keepdims=True)
    mu_dst = np.mean(dst_t, axis=1, keepdims=True)

    src_centered = src_t - mu_src
    dst_centered = dst_t - mu_dst

    var_src = np.sum(src_centered ** 2)
    if var_src < 1e-12:
        return None

    K = src_centered @ dst_centered.T
    U, s, Vt = np.linalg.svd(K)
    V = Vt.T

    Z = np.eye(3)
    if np.linalg.det(V @ U.T) < 0:
        Z[-1, -1] = -1

    R = V @ Z @ U.T
    scale = np.sum(s * np.diag(Z)) / var_src

    t = mu_dst.squeeze() - scale * (R @ mu_src).squeeze()

    return scale, R, t


def _apply_similarity_transform(points, transform):
    scale, R, t = transform
    points = np.asarray(points, dtype=np.float64)
    return scale * (R @ points.T).T + t

def rotation_matrix_to_angle_axis(R):
    """
    Return rotation angle in degrees and axis.
    """
    R = np.asarray(R, dtype=np.float64)

    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)

    if abs(angle) < 1e-8:
        axis = np.array([0.0, 0.0, 0.0])
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * np.sin(angle))

    return float(np.degrees(angle)), axis

def _rigid_rotation_transform_from_points(src, dst):
    """
    Estimate rotation + translation only.
    No scale alignment.

    aligned = (R @ src.T).T + t
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_dst - (R @ mu_src)

    return R, t


def _apply_rigid_rotation_transform(points, transform):
    R, t = transform
    points = np.asarray(points, dtype=np.float64)
    return (R @ points.T).T + t

def debug_print_pa_transform(transform, sample_name=""):
    """
    Print PA similarity transform:
        aligned = scale * (R @ pred.T).T + t
    """
    if transform is None:
        print(f"[PA DEBUG] {sample_name}: transform is None")
        return

    scale, R, t = transform
    angle_deg, axis = rotation_matrix_to_angle_axis(R)

    print("\n================ PA Transform Debug ================")
    print(f"Sample: {sample_name}")
    print(f"Scale: {scale:.6f}")
    print(f"Rotation angle: {angle_deg:.2f} degrees")
    print(f"Rotation axis: {axis}")
    print(f"det(R): {np.linalg.det(R):.6f}")
    print("R =")
    print(np.array2string(R, precision=4, suppress_small=True))
    print("t =")
    print(np.array2string(t, precision=4, suppress_small=True))
    print("====================================================\n")

def calculate_aa_3d_metrics(
    pred_joints_3d,
    gt_3d_kpts,
    kpt_types_orig=None,
    unit_scale=1000.0,
):
    """
    Compute AA-MPJPE and AA-PA-MPJPE.

    Return:
        aa_mpjpe, aa_pa_mpjpe

    If gt_3d_kpts is None, return (None, None).

    AA-MPJPE:
        root-aligned amputation-aware MPJPE.
        It includes valid standard body joints and existing residual endpoints.

    AA-PA-MPJPE:
        Procrustes-aligned amputation-aware MPJPE.
        The similarity transform is estimated using valid standard body joints,
        then applied to both body joints and residual endpoints.

    unit_scale:
        1000.0 means meter -> mm.
        If your coordinates are already in mm, set unit_scale=1.0.
    """
    if gt_3d_kpts is None:
        return None, None, None

    gt_joints_3d = normalize_gt_3d_keypoints(gt_3d_kpts)
    if gt_joints_3d is None:
        return None, None, None

    pred_root = _get_pelvis_root(pred_joints_3d)
    gt_root = _get_pelvis_root(gt_joints_3d)

    if pred_root is None or gt_root is None:
        return None, None, None

    eval_pred = []
    eval_gt = []

    body_pred = []
    body_gt = []

    # Standard body joints: 0-16
    for idx in range(0, 17):
        name = METAINFO["keypoint_info"][idx]["name"]

        # If keypoint type exists, only evaluate normal body joints.
        # type=1 prosthetic, type=2 absent should be skipped.
        if kpt_types_orig is not None and idx < len(kpt_types_orig):
            if kpt_types_orig[idx] != 0:
                continue

        if name not in pred_joints_3d or name not in gt_joints_3d:
            continue

        p_pred = _to_vec3(pred_joints_3d[name])
        p_gt = _to_vec3(gt_joints_3d[name])

        if p_pred is None or p_gt is None:
            continue

        eval_pred.append(p_pred)
        eval_gt.append(p_gt)

        body_pred.append(p_pred)
        body_gt.append(p_gt)

    # Residual endpoints: 23-30
    for idx in range(23, 31):
        name = METAINFO["keypoint_info"][idx]["name"]

        # Only existing residual endpoints are evaluated.
        if kpt_types_orig is not None and idx < len(kpt_types_orig):
            if kpt_types_orig[idx] != 0:
                continue

        if name not in gt_joints_3d:
            continue

        if name in pred_joints_3d:
            pred_name = name
        else:
            # Direct complete-body HMR baseline: use terminal landmark surrogate.
            pred_name = RESIDUAL_TO_TERMINAL.get(name)

        if pred_name is None or pred_name not in pred_joints_3d:
            continue

        p_pred = _to_vec3(pred_joints_3d[pred_name])
        p_gt = _to_vec3(gt_joints_3d[name])

        if p_pred is None or p_gt is None:
            continue

        eval_pred.append(p_pred)
        eval_gt.append(p_gt)

    if len(eval_pred) == 0:
        return None, None, None

    eval_pred = np.asarray(eval_pred, dtype=np.float64)
    eval_gt = np.asarray(eval_gt, dtype=np.float64)

    # -----------------------------
    # AA-MPJPE: root-aligned
    # -----------------------------
    eval_pred_rooted = eval_pred - pred_root
    eval_gt_rooted = eval_gt - gt_root

    aa_mpjpe = float(np.mean(np.linalg.norm(eval_pred_rooted - eval_gt_rooted, axis=1)) * unit_scale)

    # -----------------------------
    # AA-PA-MPJPE: Procrustes aligned
    # -----------------------------
    aa_pa_mpjpe = None
    aa_r_mpjpe = None
    if len(body_pred) >= 3:
        body_pred = np.asarray(body_pred, dtype=np.float64)
        body_gt = np.asarray(body_gt, dtype=np.float64)

        rigid_transform = _rigid_rotation_transform_from_points(body_pred, body_gt)
        eval_pred_rigid = _apply_rigid_rotation_transform(eval_pred, rigid_transform)

        aa_r_mpjpe = float(
            np.mean(np.linalg.norm(eval_pred_rigid - eval_gt, axis=1)) * unit_scale
        )

        transform = _similarity_transform_from_points(body_pred, body_gt)

        if transform is not None:
            # debug_print_pa_transform(transform)

            eval_pred_aligned = _apply_similarity_transform(eval_pred, transform)
            aa_pa_mpjpe = float(np.mean(np.linalg.norm(eval_pred_aligned - eval_gt, axis=1)) * unit_scale)

    return aa_mpjpe, aa_r_mpjpe, aa_pa_mpjpe


def load_3d_gt_json(gt_3d_json_path):
    """
    Load global_3d_keypoints.json.
    If path is None or file does not exist, return None.
    """
    if gt_3d_json_path is None:
        return None

    gt_3d_json_path = Path(gt_3d_json_path)
    if not gt_3d_json_path.exists():
        return None

    with open(gt_3d_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_frame_id_from_name(name):
    """
    Examples:
        Camera_View_00_frame_0000 -> 0000
        frame_0010 -> 0010
        0010 -> 0010
    """
    stem = Path(str(name)).stem

    m = re.search(r"frame_(\d+)", stem)
    if m:
        return m.group(1)

    m = re.search(r"(\d{4,})$", stem)
    if m:
        return m.group(1)

    return stem


def get_gt_3d_for_sample(gt_3d_data, sample_name):
    """
    Return one frame's 3D GT dict from global_3d_keypoints.json.
    If no matched GT, return None.
    """
    if gt_3d_data is None:
        return None

    frame_id = infer_frame_id_from_name(sample_name)
    return gt_3d_data.get(frame_id, None)

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

        # print(f"      🔍 [坐标对齐核查 - 已简化]")
        # print(f"         - 标注点 (pt_2d)      : [{pt_2d[0]:.1f}, {pt_2d[1]:.1f}]")
        # print(f"         - 骨骼起始投影 (p_start): [{p_start[0]:.1f}, {p_start[1]:.1f}]")
        # print(f"         - 骨骼末端投影 (p_end)  : [{p_end[0]:.1f}, {p_end[1]:.1f}]")

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

    def _get_closest_point_ray_to_bone(self, pt_2d, bone_start_3d, bone_end_3d):
        """
        🚀 射线-骨骼求交法 (Ray-Casting)
        计算 2D 像素射线与 3D 骨骼线段之间的最近点（公垂线点），并强制 Clip 在线段内。
        """
        # 1. 构建从相机光心穿过 2D 标注点的 3D 射线 (Ray)
        u, v = pt_2d
        ray_dir = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        ray_origin = self.cam_origin  # np.array([0.0, 0.0, 0.0])

        # 2. 骨骼线段方向 (Line Segment)
        bone_vec = bone_end_3d - bone_start_3d

        # 3. 计算 3D 空间中两条直线的最短距离点 (数学公垂线公式)
        w0 = bone_start_3d - ray_origin
        a = np.dot(ray_dir, ray_dir)  # 恒为 1.0
        b = np.dot(ray_dir, bone_vec)
        c = np.dot(bone_vec, bone_vec)
        d = np.dot(ray_dir, w0)
        e = np.dot(bone_vec, w0)

        denom = a * c - b * b

        if denom < 1e-6:
            # 射线和骨骼平行（极小概率事件），默认取中间
            t = 0.5
        else:
            # t 是落在 3D 骨骼线段上的比例参数
            t = (a * e - b * d) / denom

        # 4. 🛡️ 强制安全锁：Clip 限制
        # 预留 5% 缓冲区，防止 t=0 或 1 时刚好切在关节点上导致 Mesh 破洞或撕裂
        t_clipped = np.clip(t, 0.05, 0.95)

        # 5. 计算出绝对安全的 3D 切点
        safe_cut_origin = bone_start_3d + t_clipped * bone_vec

        # 可选：计算射线和骨骼的最短物理误差距离
        # s 是落在射线上的参数
        s = (b * e - c * d) / denom if denom >= 1e-6 else 0
        point_on_ray = ray_origin + s * ray_dir
        physical_error_dist = np.linalg.norm(safe_cut_origin - point_on_ray)

        return safe_cut_origin, physical_error_dist

    def process_multiple_cuts(self, mesh_path, cut_tasks):
        """
        执行多处截肢任务，Watertight 封口，并强制生成抛物线生理鼓包
        """
        # print(f"\n🔪 [Mesh Cutter] 正在手术 (对齐模式: 直接投影)")
        if not cut_tasks:
            return trimesh.load(mesh_path, process=False)

        mesh = trimesh.load(mesh_path, process=False)
        has_cut = False

        for task in cut_tasks:
            part_name = task.get('name', '未知部位')
            # print(f"   -> 处理部位: {part_name}")

            # 直接计算 Lambda，不再进行坐标转换
            # cut_origin= self._calculate_exact_cut_proportion_2d_driven(
            #     task['pt_2d'], task['start_3d'], task['end_3d']
            # )

            # if error_dist > 0.2:
            #     print(f"⚠️ 警告: 部位 {part_name} HMR 预测空间偏差较大 (距离: {error_dist:.2f}m)，已强制对其执行安全截断。")

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
        # print(f"      -> 开始拓扑重建 (Watertight 封口)...")
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
        # print(f"      -> 开始施加端点物理膨胀...")
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
        # print("      ✅ 已成功生成完美弧度残肢端点。")

        return sealed_mesh

def load_merged_3d_annotations(ann3d_path):
    with open(ann3d_path, "r", encoding="utf-8") as f:
        ann3d = json.load(f)

    return {
        ann["id"]: ann
        for ann in ann3d.get("annotations", [])
    }


def load_2d_image_to_gt3d_map(ann2d_path):
    """
    Build mapping:
        image stem -> gt3d_id

    Example image stem:
        demo16__Camera_View_07__frame_0130
    """
    with open(ann2d_path, "r", encoding="utf-8") as f:
        ann2d = json.load(f)

    mapping = {}

    for img_info in ann2d.get("images", []):
        stem = Path(img_info["file_name"]).stem
        mapping[stem] = img_info.get("gt3d_id")

    return mapping


def get_gt_3d_for_sample_from_merged(sample_name, image_to_gt3d, ann3d_by_id):
    sample_stem = Path(str(sample_name)).stem

    gt3d_id = image_to_gt3d.get(sample_stem)
    if gt3d_id is None:
        print(f"⚠️ No gt3d_id found for sample: {sample_stem}")
        return None

    gt3d_ann = ann3d_by_id.get(gt3d_id)
    if gt3d_ann is None:
        print(f"⚠️ No 3D annotation found for gt3d_id={gt3d_id}")
        return None

    return gt3d_ann

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

def main(ori_image_path, gen_image_path, reconstructor, annotation_file, gt_3d_kpts=None):
    kpts_orig, kpts, types_orig = read_kpts_annotation(ori_image_path, annotation_file)

    for i in range(len(kpts_orig)):
        kpts_orig[i][0] /= 3.0  # x 坐标缩小
        kpts_orig[i][1] /= 3.0  # y 坐标缩小

    # ============================================================
    # 1. 提取真实的 GT Mask (基于原图 Alpha 通道或灰度图) 算 mIoU 用
    # ============================================================
    img_ori_raw = cv2.imread(ori_image_path, cv2.IMREAD_UNCHANGED)
    if img_ori_raw is None:
        return 0, 0, 0, None, None, None

    target_h, target_w = int(img_ori_raw.shape[0] / 3), int(img_ori_raw.shape[1] / 3)

    if len(img_ori_raw.shape) == 3 and img_ori_raw.shape[2] == 4:
        alpha_channel = img_ori_raw[:, :, 3]
        _, raw_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img_ori_raw, cv2.COLOR_BGR2GRAY)
        _, raw_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    mask_gt_resized = cv2.resize(raw_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # ============================================================
    # 2. 制作白底的原图 (Ori)
    # ============================================================
    if len(img_ori_raw.shape) == 3 and img_ori_raw.shape[2] == 4:
        alpha_norm = img_ori_raw[:, :, 3] / 255.0
        white_bg = np.ones_like(img_ori_raw[:, :, :3]) * 255
        img_ori_white = (img_ori_raw[:, :, :3] * alpha_norm[:, :, np.newaxis] +
                         white_bg * (1 - alpha_norm[:, :, np.newaxis])).astype(np.uint8)
    else:
        img_ori_white = cv2.cvtColor(img_ori_raw, cv2.COLOR_GRAY2BGR) if len(img_ori_raw.shape) == 2 else img_ori_raw[
            :, :, :3]

    image_ori_resized = cv2.resize(img_ori_white, (target_w, target_h), interpolation=cv2.INTER_AREA)
    base_name_ori, _ = os.path.splitext(ori_image_path)
    temp_ori_path = f"{base_name_ori}_white_resized.jpg"
    cv2.imwrite(temp_ori_path, image_ori_resized)
    ori_image_path = temp_ori_path

    # ============================================================
    # 3. 🌟 新增：制作白底的生成图 (Gen)
    # ============================================================
    img_gen_raw = cv2.imread(gen_image_path, cv2.IMREAD_UNCHANGED)
    if img_gen_raw is not None and len(img_gen_raw.shape) == 3 and img_gen_raw.shape[2] == 4:
        alpha_norm_gen = img_gen_raw[:, :, 3] / 255.0
        white_bg_gen = np.ones_like(img_gen_raw[:, :, :3]) * 255
        img_gen_white = (img_gen_raw[:, :, :3] * alpha_norm_gen[:, :, np.newaxis] +
                         white_bg_gen * (1 - alpha_norm_gen[:, :, np.newaxis])).astype(np.uint8)
    elif img_gen_raw is not None:
        img_gen_white = cv2.cvtColor(img_gen_raw, cv2.COLOR_GRAY2BGR) if len(img_gen_raw.shape) == 2 else img_gen_raw[
            :, :, :3]
    else:
        # 兜底
        img_gen_white = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    img_gen_resized = cv2.resize(img_gen_white, (target_w, target_h), interpolation=cv2.INTER_AREA)
    base_name_gen, _ = os.path.splitext(gen_image_path)
    temp_gen_path = f"{base_name_gen}_white_resized.jpg"  # 强制存为 JPG 抹除透明度
    cv2.imwrite(temp_gen_path, img_gen_resized)

    # ============================================================
    # 4. 模型推理与后续逻辑
    # ============================================================
    dir_name = os.path.dirname(temp_gen_path)
    mesh_save_path = os.path.join(dir_name, "whole_body_mesh.obj")

    try:
        mesh_save_path, pred_joints_3d, pred_cam, mesh = reconstructor.predict_mesh(
            ori_image_path, mesh_save_path)
    except SystemExit as e:
        print(e)
        return 0, 0, 0, None, None, None

    whole_mesh = mesh.copy()
    global_focal = pred_cam['focal']
    global_cx = pred_cam['princpt'][0]
    global_cy = pred_cam['princpt'][1]
    global_cam = {
        'focal': global_focal,
        'princpt': np.array([global_cx, global_cy])
    }

    # 绘制可视化对比图 (现在原图和生成图都是干净的白底了)
    vis_save_path = os.path.join(dir_name, "keypoints_comparison.jpg")
    visualize_keypoints_comparison(ori_image_path, pred_joints_3d, global_cam, vis_save_path)
    project_mesh_overlay(ori_image_path, temp_gen_path, mesh, global_cam, dir_name)
    cut_tasks = []
    for i in range(23, 31):
        if types_orig[i] == 0:
            res_name = METAINFO['keypoint_info'][i]['name']
            pt_2d_orig = kpts_orig[i][0:2]
            pt_2d_orig_homo = np.array([pt_2d_orig[0], pt_2d_orig[1]])
            if res_name in RES_BONE_MAPPING:
                start_joint_name, end_joint_name = RES_BONE_MAPPING[res_name]
                start_3d = pred_joints_3d[start_joint_name]
                end_3d = pred_joints_3d[end_joint_name]

                cut_tasks.append({
                    'name': res_name,
                    'pt_2d': pt_2d_orig_homo,
                    'start_3d': start_3d,
                    'end_3d': end_3d
                })

    if cut_tasks:
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
        for task in cut_tasks:
            if 'cut_origin' in task:
                pred_joints_3d[task['name']] = task['cut_origin']

        global_cam = {
            'focal': global_focal,
            'princpt': np.array([global_cx, global_cy])
        }

        orig_proj_path, pred_mask_orig, gen_proj_path, pred_mask_gen = project_mesh_overlay(ori_image_path,
                                                                                            temp_gen_path, mesh,
                                                                                            global_cam, dir_name)

        paper_vis_dir = os.path.join(dir_name, "paper_visualizations")
        os.makedirs(paper_vis_dir, exist_ok=True)

        paper_paths = reconstructor.render_paper_projections(
            image_path=ori_image_path,
            out_dir=paper_vis_dir,
            whole_mesh=whole_mesh,
            cut_mesh=mesh,
            pred_cam=pred_cam,
        )

        print("Whole-body projection:", paper_paths["whole"])
        print("Cut-mesh projection:", paper_paths["cut"])
        # 🚨 使用一开始保留下来的真实 Mask，防止 mIoU 受白底影响变 0
        miou_score = calculate_miou(pred_mask_orig, mask_gt_resized)
        print(f"      -> miou_score: {miou_score:.4f}")

        INTACT_MAPPING = {METAINFO['keypoint_info'][i]['name']: i for i in range(0, 17)}
        RES_MAPPING = {METAINFO['keypoint_info'][i]['name']: i for i in range(23, 31)}
        mpjpe_intact = calculate_2d_mpjpe(pred_joints_3d, kpts_orig, global_cam, INTACT_MAPPING)
        mpjpe_residual = calculate_2d_mpjpe(pred_joints_3d, kpts_orig, global_cam, RES_MAPPING)

        print(f"📊 [量化评估] 完整关节 2D MPJPE: {mpjpe_intact:.2f} pixels")
        print(f"📊 [量化评估] 残肢端点 2D MPJPE: {mpjpe_residual:.2f} pixels")

        aa_mpjpe, aa_r_mpjpe, aa_pa_mpjpe = calculate_aa_3d_metrics(
            pred_joints_3d=pred_joints_3d,
            gt_3d_kpts=gt_3d_kpts,
            kpt_types_orig=types_orig,
            unit_scale=1000.0,  # meter -> mm
        )
        if gt_3d_kpts is not None:
            vis_prefix = os.path.join(dir_name, "aa_metric_vis")
            visualize_aa_alignment(
                pred_joints_3d=pred_joints_3d,
                gt_3d_kpts=gt_3d_kpts,
                kpt_types_orig=types_orig,
                save_prefix=vis_prefix
            )

        if aa_mpjpe is not None:
            print(f"📊 [3D评估] AA-MPJPE: {aa_mpjpe:.2f} mm")
        else:
            print("📊 [3D评估] AA-MPJPE: skipped")

        if aa_pa_mpjpe is not None:
            print(f"📊 [3D评估] AA-PA-MPJPE: {aa_pa_mpjpe:.2f} mm")
        else:
            print("📊 [3D评估] AA-PA-MPJPE: skipped")

        if aa_r_mpjpe is not None:
            print(f"📊 [3D评估] AA-R-MPJPE: {aa_r_mpjpe:.2f} mm")
        else:
            print("📊 [3D评估] AA-R-MPJPE: skipped")

        return miou_score, mpjpe_intact, mpjpe_residual, aa_mpjpe, aa_r_mpjpe, aa_pa_mpjpe

    # 如果没有 cut tasks，随便返回个值或者 0
    return 0, 0, 0, None, None, None


if __name__ == "__main__":
    reconstructor = ReconstructionEngine()
    workdir = Path('./workdir1')

    ann2d_path = "./data/filtered_annotations.json"
    # ann3d_path = "./3D_data/annotations_3d_propose.json"

    # image_to_gt3d = load_2d_image_to_gt3d_map(ann2d_path)
    # ann3d_by_id = load_merged_3d_annotations(ann3d_path)
    # 提前转为 list
    dirs = list(workdir.glob('*'))

    miou = 0
    mpjpe_intact = 0
    mpjpe_residual = 0
    valid_count = 0  # 🌟 必须加上有效计数器
    aa_mpjpe_sum = 0.0
    aa_pa_mpjpe_sum = 0.0
    aa_r_mpjpe_sum = 0.0
    valid_3d_count = 0
    bad_images = []

    for dir_path in dirs:
        image_folder = Path(dir_path)

        all_files = [str(p) for p in image_folder.iterdir() if p.is_file() and p.name.startswith('compositing')]

        if not all_files:
            continue

        all_files.sort()
        gen_image_path = all_files[-1]
        ori_image_path = f'./data/eval/{dir_path.name}.jpg'
        print(f'start to analyse image {gen_image_path}')
        # gt_3d_kpts = get_gt_3d_for_sample_from_merged(
        #     sample_name=dir_path.name,
        #     image_to_gt3d=image_to_gt3d,
        #     ann3d_by_id=ann3d_by_id,
        # )
        result = main(
            ori_image_path,
            gen_image_path,
            reconstructor,
            annotation_file=ann2d_path,
            gt_3d_kpts=None,
        )

        # 🌟 跳过失败的预测，防止 0 误差污染平均值
        if result[0] == 0 and result[1] == 0 and result[2] == 0:
            print(f"❌ {dir_path.name} 预测失败，跳过统计。")
            continue

        current_miou, current_intact, current_residual, current_aa_mpjpe, current_aa_r_mpjpe, current_aa_pa_mpjpe = result

        if current_miou < 0.7:
            bad_images.append(dir_path.name)
        if current_miou < 0.1:
            print(f"❌ {dir_path.name} 预测失败，跳过统计。")
            continue
        miou += current_miou
        mpjpe_intact += current_intact
        mpjpe_residual += current_residual
        valid_count += 1

        if (
                current_aa_mpjpe is not None
                and current_aa_r_mpjpe is not None
                and current_aa_pa_mpjpe is not None
        ):
            aa_mpjpe_sum += current_aa_mpjpe
            aa_pa_mpjpe_sum += current_aa_pa_mpjpe
            aa_r_mpjpe_sum += current_aa_r_mpjpe
            valid_3d_count += 1

        # 🌟 用 valid_count 算平均值
    if valid_count > 0:
        print(f"\n📈 [最终平均评估] (有效样本: {valid_count}/{len(dirs)})")
        print(f"   平均 mIoU: {miou / valid_count:.4f}")
        print(f"   平均完整关节 2D MPJPE: {mpjpe_intact / valid_count:.2f} pixels")
        print(f"   平均残肢端点 2D MPJPE: {mpjpe_residual / valid_count:.2f} pixels")
        print(f"\n🚨 Bad images (mIoU < 0.7): {bad_images}")
    else:
        print("\n💥 整个数据集处理失败。")

    if valid_3d_count > 0:
        print(f"   平均 AA-MPJPE: {aa_mpjpe_sum / valid_3d_count:.2f} mm")
        print(f"   平均 AA-PA-MPJPE: {aa_pa_mpjpe_sum / valid_3d_count:.2f} mm")
        print(f"  平均 AA-R-MPJPE: {aa_r_mpjpe_sum / valid_3d_count:.2f} mm")
    else:
        print("   平均 3D metrics: skipped, no 3D GT found")