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

        # 3. 提取 3D 关节点，并立刻加上 cam_t
        joints_3d = person_data.get("pred_keypoints_3d")
        if joints_3d is not None and hasattr(joints_3d, 'cpu'):
            joints_3d = joints_3d.detach().cpu().numpy()

        if joints_3d is not None and cam_t is not None:
            joints_3d = joints_3d + cam_t  # 🌟 加上偏移！

        pred_joints_dict = {}
        if joints_3d is not None:
            is_smpl = len(joints_3d) >= 24
            pred_joints_dict = {
                'pelvis': joints_3d[0],
                'left_hip': joints_3d[1] if is_smpl else joints_3d[11],
                'right_hip': joints_3d[2] if is_smpl else joints_3d[12],
                'left_knee': joints_3d[4] if is_smpl else joints_3d[13],
                'right_knee': joints_3d[5] if is_smpl else joints_3d[14],
                'left_ankle': joints_3d[7] if is_smpl else joints_3d[15],
                'right_ankle': joints_3d[8] if is_smpl else joints_3d[16],
                'neck': joints_3d[12] if is_smpl else joints_3d[0],
                'left_shoulder': joints_3d[16] if is_smpl else joints_3d[5],
                'right_shoulder': joints_3d[17] if is_smpl else joints_3d[6],
                'left_elbow': joints_3d[18] if is_smpl else joints_3d[7],
                'right_elbow': joints_3d[19] if is_smpl else joints_3d[8],
                'left_wrist': joints_3d[20] if is_smpl else joints_3d[9],
                'right_wrist': joints_3d[21] if is_smpl else joints_3d[10],
            }

        # 五官映射 (由于 vertices 已经加上了 cam_t，这里直接取顶点依然是正确的绝对物理坐标)
        if len(vertices) > 10000:
            pred_joints_dict.update({
                'nose': vertices[9120], 'left_eye': vertices[9448], 'right_eye': vertices[9929],
                'left_ear': vertices[6], 'right_ear': vertices[616],
            })
        else:
            pred_joints_dict.update({
                'nose': vertices[331], 'left_eye': vertices[2802], 'right_eye': vertices[6262],
                'left_ear': vertices[3489], 'right_ear': vertices[3990],
            })

        return save_path, pred_joints_dict, global_cam, mesh