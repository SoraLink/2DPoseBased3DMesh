import sys
import os
import cv2
import numpy as np
import trimesh
import torch

# ==========================================
# 1. 暴力注入 Meta 仓库路径（解决瞎子问题）
# ==========================================
REPO_ROOT = '/home/sora/workspace/sam-3d-body'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ==========================================
# 2. 从核心包直接导入 API（避开 utils 命名冲突）
# ==========================================
from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator


class ReconstructionEngine:
    def __init__(self, hf_repo_id="facebook/sam-3d-body-dinov3", device='cuda'):
        """
        初始化 SAM 3D Body 引擎
        """
        print(f">>> 正在加载 Meta SAM 3D Body 模型 ({hf_repo_id})...")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载核心模型
        model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

        # 初始化 Estimator
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None
        )
        self.device = device
        print(">>> 🚀 SAM 3D Body 模型加载完成！")

    def predict_mesh(self, image_path: str, save_path: str):
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise ValueError(f"❌ 无法读取图像: {image_path}")
        height, width = img_cv2.shape[:2]

        outputs = self.estimator.process_one_image(image_path)
        if not outputs or len(outputs) == 0:
            raise ValueError("❌ 预测失败，未检测到人物。")

        person_data = outputs[0]

        # ==========================================
        # 🚨 关键修复：提取相机平移参数 cam_t
        # ==========================================
        cam_t = person_data.get("pred_cam_t")
        if cam_t is not None and hasattr(cam_t, 'cpu'):
            cam_t = cam_t.detach().cpu().numpy()

        # 1. 提取 Mesh 顶点，并立刻加上 cam_t (将局部坐标转换到相机物理坐标)
        vertices = person_data["pred_vertices"]
        if hasattr(vertices, 'cpu'):
            vertices = vertices.detach().cpu().numpy()

        if cam_t is not None:
            vertices = vertices + cam_t  # 🌟 加上偏移！

        faces = self.estimator.faces
        if hasattr(faces, 'cpu'):
            faces = faces.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)

        # 2. 提取并更新相机内参 (下游用不到 cam_t 了，因为已经加过了)
        focal_length = float(person_data["focal_length"])
        global_cam = {
            'focal': np.array([focal_length, focal_length]),
            'princpt': np.array([width / 2.0, height / 2.0]),
            'cam_t': cam_t
        }

        joints_3d = person_data.get("pred_keypoints_3d")
        if joints_3d is not None and hasattr(joints_3d, 'cpu'):
            joints_3d = joints_3d.detach().cpu().numpy()

        if joints_3d is not None and cam_t is not None:
            joints_3d = joints_3d + cam_t  # 🌟 加上偏移，转换到相机绝对坐标系

        pred_joints_dict = {}
        if joints_3d is not None:
            # 🌟 核心修复：SAM 3D Body 的 MHR70 格式，前 17 个点就是完美的 COCO 17！
            # 顺序和你代码里的 METAINFO 'keypoint_info' 做到 100% 绝对对齐。
            pred_joints_dict = {
                'nose': joints_3d[0],
                'left_eye': joints_3d[1],
                'right_eye': joints_3d[2],
                'left_ear': joints_3d[3],
                'right_ear': joints_3d[4],
                'left_shoulder': joints_3d[5],
                'right_shoulder': joints_3d[6],
                'left_elbow': joints_3d[7],
                'right_elbow': joints_3d[8],
                'left_wrist': joints_3d[9],
                'right_wrist': joints_3d[10],
                'left_hip': joints_3d[11],
                'right_hip': joints_3d[12],
                'left_knee': joints_3d[13],
                'right_knee': joints_3d[14],
                'left_ankle': joints_3d[15],
                'right_ankle': joints_3d[16],

                # SMPLest-X 可能还需要这两个基础骨骼点，用中点生成即可
                'pelvis': (joints_3d[11] + joints_3d[12]) / 2.0,
                'neck': (joints_3d[5] + joints_3d[6]) / 2.0
            }

        joints_2d = person_data.get("pred_keypoints_2d")
        if joints_2d is not None and hasattr(joints_2d, 'cpu'):
            joints_2d = joints_2d.detach().cpu().numpy()

        joints_2d_raw = person_data.get("pred_keypoints_2d")
        if joints_2d_raw is not None and hasattr(joints_2d_raw, 'cpu'):
            joints_2d_raw = joints_2d_raw.detach().cpu().numpy()

        joints_3d_raw = person_data.get("pred_keypoints_3d")
        if joints_3d_raw is not None and hasattr(joints_3d_raw, 'cpu'):
            joints_3d_raw = joints_3d_raw.detach().cpu().numpy()

        if joints_3d_raw is not None and cam_t is not None:
            joints_3d_raw = joints_3d_raw + cam_t  # 🌟 保持相机绝对坐标系转换

        pred_joints_2d_dict = {}
        if joints_2d is not None:
            pred_joints_2d_dict = {
                'nose': joints_2d[0], 'left_eye': joints_2d[1], 'right_eye': joints_2d[2],
                'left_ear': joints_2d[3], 'right_ear': joints_2d[4], 'left_shoulder': joints_2d[5],
                'right_shoulder': joints_2d[6], 'left_elbow': joints_2d[7], 'right_elbow': joints_2d[8],
                'left_wrist': joints_2d[9], 'right_wrist': joints_2d[10], 'left_hip': joints_2d[11],
                'right_hip': joints_2d[12], 'left_knee': joints_2d[13], 'right_knee': joints_2d[14],
                'left_ankle': joints_2d[15], 'right_ankle': joints_2d[16]
            }
        return save_path, pred_joints_dict, global_cam, mesh, joints_2d_raw, joints_3d_raw